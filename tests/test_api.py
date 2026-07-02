import os
import pytest
from fastapi.testclient import TestClient

# Set environment variables for testing before importing the app
# Use a file-based SQLite db to share database state across TestClient request connections.
DB_FILE = "test_verity.db"
os.environ["DATABASE_URL"] = f"sqlite:///{DB_FILE}"
os.environ["JWT_SECRET"] = "test_super_secret"
os.environ["JWT_ALGORITHM"] = "HS256"
os.environ["JWT_EXPIRE_MINUTES"] = "60"
os.environ["DB_PASSWORD"] = "test_password"
os.environ["ALLOWED_ORIGINS"] = "http://localhost:4200,https://myproductiondomain.com"
os.environ["SENTRY_DSN"] = ""  # Disable Sentry in tests
os.environ["MAX_REQUEST_SIZE_BYTES"] = "2048"  # 2KB limit for easy testing

from api.server import app
from api.database import Base, engine
from api.rate_limit import login_limiter


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    # Create the tables at the start of the test session
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    # Clean up the test database file after the session finishes
    Base.metadata.drop_all(bind=engine)
    if os.path.exists(DB_FILE):
        try:
            os.remove(DB_FILE)
        except Exception:
            pass


@pytest.fixture(autouse=True)
def clean_db_tables():
    # Re-create all tables for a clean slate before each test case
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    # Clear rate limiter state
    with login_limiter.lock:
        login_limiter.requests.clear()


def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["status"] == "running"


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["database"] == "up"
        assert data["service"] == "verity"


def test_health_endpoint_db_failure(monkeypatch):
    from sqlalchemy.orm import Session
    # Force a failure on DB execution to simulate DB connection issue
    def mock_execute(*args, **kwargs):
        raise Exception("DB is down")
    monkeypatch.setattr(Session, "execute", mock_execute)

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 503
        assert "Database connection failed" in response.json()["detail"]


def test_cors_headers():
    with TestClient(app) as client:
        # Test an allowed origin
        response = client.options("/health", headers={
            "Origin": "https://myproductiondomain.com",
            "Access-Control-Request-Method": "GET"
        })
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "https://myproductiondomain.com"

        # Test a disallowed origin
        response = client.options("/health", headers={
            "Origin": "https://maliciousdomain.com",
            "Access-Control-Request-Method": "GET"
        })
        assert response.headers.get("access-control-allow-origin") is None


def test_input_size_limit_middleware():
    with TestClient(app) as client:
        # Create a payload slightly larger than 2KB limit (e.g. 3000 chars)
        large_body = "x" * 3000
        response = client.post("/auth/register", content=large_body, headers={"Content-Type": "application/json"})
        assert response.status_code == 413
        assert response.json()["detail"] == "Payload too large"


def test_pydantic_field_limits():
    with TestClient(app) as client:
        # Username too long (> 50 chars)
        long_username = "u" * 51
        response = client.post("/auth/register", json={
            "email": "test@example.com",
            "username": long_username,
            "password": "validpassword"
        })
        assert response.status_code == 422

        # Email too long (> 255 chars)
        long_email = ("e" * 250) + "@example.com"
        response = client.post("/auth/register", json={
            "email": long_email,
            "username": "user",
            "password": "validpassword"
        })
        assert response.status_code == 422


def test_rate_limiting_login():
    with TestClient(app) as client:
        # Make 5 failed logins (allowed limit is 5)
        for _ in range(5):
            response = client.post("/auth/login", json={
                "email": "test@example.com",
                "password": "wrongpassword"
            })
            assert response.status_code == 401

        # The 6th request should be rate limited (429)
        response = client.post("/auth/login", json={
            "email": "test@example.com",
            "password": "wrongpassword"
        })
        assert response.status_code == 429
        assert "Too many login attempts" in response.json()["detail"]


def test_verify_async_flow():
    with TestClient(app) as client:
        # Register and get access token
        reg_response = client.post("/auth/register", json={
            "email": "user@example.com",
            "username": "user123",
            "password": "securepassword"
        })
        assert reg_response.status_code == 200
        token = reg_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Call /verify using the mock provider
        verify_response = client.post("/verify", json={
            "answer": "The company raised $12 million. It is located in Paris.",
            "context": "The company raised a $12 million Series A. Paris is where the company is headquartered.",
            "provider": "mock",
            "extractor": "rule_based"
        }, headers=headers)

        assert verify_response.status_code == 200
        data = verify_response.json()
        assert "run_id" in data
        assert data["status"] in ("pending", "processing", "completed")

        # Verify details of the run (CELERY_ALWAYS_EAGER ensures task runs synchronously during .delay())
        run_id = data["run_id"]
        detail_response = client.get(f"/runs/{run_id}", headers=headers)
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["status"] == "completed"
        assert len(detail["claims"]) > 0
        assert detail["stats"]["total"] == len(detail["claims"])

        # Verify it was saved to the database (check /runs)
        runs_response = client.get("/runs", headers=headers)
        assert runs_response.status_code == 200
        runs = runs_response.json()
        assert len(runs) == 1
        assert runs[0]["provider"] == "mock"


def test_quota_limits():
    with TestClient(app) as client:
        # 1. Register a user
        reg_response = client.post("/auth/register", json={
            "email": "quota_test@example.com",
            "username": "quota_user",
            "password": "securepassword"
        })
        assert reg_response.status_code == 200
        token = reg_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 2. Get user info, quota should be 10
        me_response = client.get("/auth/me", headers=headers)
        assert me_response.status_code == 200
        assert me_response.json()["remaining_quota"] == 10

        # 3. Trigger 10 verification runs
        for i in range(10):
            verify_response = client.post("/verify", json={
                "answer": "The company raised $12 million.",
                "context": "The company raised a $12 million Series A.",
                "provider": "mock",
                "extractor": "rule_based"
            }, headers=headers)
            assert verify_response.status_code == 200
            assert verify_response.json()["remaining_quota"] == 9 - i

        # 4. Trigger the 11th run, it should fail with 403 Forbidden
        verify_response = client.post("/verify", json={
            "answer": "The company raised $12 million.",
            "context": "The company raised a $12 million Series A.",
            "provider": "mock",
            "extractor": "rule_based"
        }, headers=headers)
        assert verify_response.status_code == 403
        assert "Daily verification limit" in verify_response.json()["detail"]

        # 5. Check me endpoint again, remaining quota should be 0
        me_response = client.get("/auth/me", headers=headers)
        assert me_response.status_code == 200
        assert me_response.json()["remaining_quota"] == 0


def test_get_runs_endpoints():
    with TestClient(app) as client:
        # Register a user
        reg_response = client.post("/auth/register", json={
            "email": "history_test@example.com",
            "username": "history_user",
            "password": "securepassword"
        })
        assert reg_response.status_code == 200
        token = reg_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Create a verification run
        client.post("/verify", json={
            "answer": "The company raised $12 million.",
            "context": "The company raised a $12 million Series A.",
            "provider": "mock",
            "extractor": "rule_based"
        }, headers=headers)

        # Query GET /runs
        runs_response = client.get("/runs", headers=headers)
        assert runs_response.status_code == 200
        runs = runs_response.json()
        assert len(runs) == 1
        run_id = runs[0]["id"]
        assert runs[0]["provider"] == "mock"
        assert "total" in runs[0]["stats"]

        # Query GET /runs/{id}
        detail_response = client.get(f"/runs/{run_id}", headers=headers)
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["id"] == run_id
        assert detail["answer"] == "The company raised $12 million."
        assert len(detail["claims"]) == 1


def test_api_key_auth_and_management():
    with TestClient(app) as client:
        # 1. Register a user
        reg_response = client.post("/auth/register", json={
            "email": "key_user@example.com",
            "username": "key_user",
            "password": "securepassword"
        })
        assert reg_response.status_code == 200
        token = reg_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 2. Create API key
        create_response = client.post("/auth/keys", json={"name": "My Server Key"}, headers=headers)
        assert create_response.status_code == 200
        key_data = create_response.json()
        assert "api_key" in key_data
        assert key_data["name"] == "My Server Key"
        assert key_data["prefix"].startswith("vt_live_")

        raw_key = key_data["api_key"]
        key_id = key_data["id"]

        # 3. List API keys
        list_response = client.get("/auth/keys", headers=headers)
        assert list_response.status_code == 200
        keys = list_response.json()
        assert len(keys) == 1
        assert keys[0]["id"] == key_id
        assert "api_key" not in keys[0]  # Raw key should not be returned in list

        # 4. Authenticate using X-API-Key header
        api_headers = {"X-API-Key": raw_key}
        me_response = client.get("/auth/me", headers=api_headers)
        assert me_response.status_code == 200
        assert me_response.json()["email"] == "key_user@example.com"

        # 5. Call verify endpoint using API Key
        verify_response = client.post("/verify", json={
            "answer": "The company raised $12 million.",
            "context": "The company raised a $12 million Series A.",
            "provider": "mock",
            "extractor": "rule_based"
        }, headers=api_headers)
        assert verify_response.status_code == 200
        verify_data = verify_response.json()
        assert "run_id" in verify_data

        # Verify details of that run using the API Key
        run_id = verify_data["run_id"]
        detail_response = client.get(f"/runs/{run_id}", headers=api_headers)
        assert detail_response.status_code == 200
        assert detail_response.json()["status"] == "completed"

        # 6. Revoke API key
        delete_response = client.delete(f"/auth/keys/{key_id}", headers=headers)
        assert delete_response.status_code == 204

        # 7. Authenticate again using revoked X-API-Key header (should fail)
        bad_me_response = client.get("/auth/me", headers=api_headers)
        assert bad_me_response.status_code == 401
        assert "Invalid or inactive API key" in bad_me_response.json()["detail"]


def test_dashboard_stats():
    with TestClient(app) as client:
        # 1. Register
        reg_response = client.post("/auth/register", json={
            "email": "dash_user@example.com",
            "username": "dash_user",
            "password": "securepassword"
        })
        token = reg_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 2. Get stats when empty
        stats_response = client.get("/dashboard/stats", headers=headers)
        assert stats_response.status_code == 200
        stats = stats_response.json()
        assert stats["total_runs"] == 0
        assert stats["average_accuracy"] == 0.0
        assert stats["activity_history"] == []

        # 3. Create run (RuleBasedMockProvider: 1 supported claim, 1 contradicted claim, 1 unsupported claim)
        verify_response = client.post("/verify", json={
            "answer": "The company raised $12 million. The company raised $50 million. The company plans to expand to Europe next year.",
            "context": "The company raised a $12 million Series A led by Greenfield Ventures.",
            "provider": "mock",
            "extractor": "rule_based"
        }, headers=headers)
        assert verify_response.status_code == 200
        run_id = verify_response.json()["run_id"]

        # 4. Get stats again
        stats_response = client.get("/dashboard/stats", headers=headers)
        assert stats_response.status_code == 200
        stats = stats_response.json()
        assert stats["total_runs"] == 1
        # Claims breakdown
        breakdown = stats["hallucinations_breakdown"]
        assert breakdown["supported"] == 1
        assert breakdown["contradicted"] == 1
        assert breakdown["unsupported"] == 1
        # Accuracy: 1 / 3 = 33.3%
        assert stats["average_accuracy"] == 33.3
        assert len(stats["activity_history"]) == 1
        assert stats["activity_history"][0]["id"] == run_id
        assert stats["activity_history"][0]["claims_count"] == 3
        assert stats["activity_history"][0]["accuracy"] == 33.3


def test_pdf_export():
    with TestClient(app) as client:
        # Register
        reg_response = client.post("/auth/register", json={
            "email": "pdf_user@example.com",
            "username": "pdf_user",
            "password": "securepassword"
        })
        token = reg_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Create run
        verify_response = client.post("/verify", json={
            "answer": "The company raised $12 million.",
            "context": "The company raised a $12 million Series A.",
            "provider": "mock",
            "extractor": "rule_based"
        }, headers=headers)
        assert verify_response.status_code == 200
        run_id = verify_response.json()["run_id"]

        # Download PDF
        pdf_response = client.get(f"/runs/{run_id}/pdf", headers=headers)
        assert pdf_response.status_code == 200
        assert pdf_response.headers.get("content-type") == "application/pdf"
        assert len(pdf_response.content) > 0
        # Check PDF signature bytes (%PDF-)
        assert pdf_response.content.startswith(b"%PDF-")
