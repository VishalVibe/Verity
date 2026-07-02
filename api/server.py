"""
Verity — FastAPI verification server.

Exposes endpoints:
  POST /auth/register  { email, username, password }
  POST /auth/login     { email, password }
  GET  /auth/me        → current user
  POST /verify         { answer, context, provider } → requires auth
  GET  /runs           → user's verification history
  GET  /runs/{id}      → single run detail

Run from the project root:
  uvicorn api.server:app --reload --port 8000
"""

import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

# Initialize Sentry for error tracking if DSN is set
sentry_dsn = os.environ.get("SENTRY_DSN")
if sentry_dsn:
    import sentry_sdk
    sentry_sdk.init(
        dsn=sentry_dsn,
        traces_sample_rate=1.0,
        profiles_sample_rate=1.0,
    )

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.database import engine, get_db
from api import models
from api.auth import get_current_user
from api.auth_routes import router as auth_router
from api.api_key_routes import router as api_key_router
from api.schemas import RunDetail, RunSummary, DashboardStatsResponse
from api.tasks import verify_claims_task
from api.pdf_generator import generate_verification_pdf

from src.chunking import chunk_document
from src.retrieval.retrieval import TfidfRetriever
from src.verification.verify_llm import (
    AnthropicProvider,
    GeminiProvider,
    GroqProvider,
    OpenAIProvider,
    RuleBasedMockProvider,
    verify_claim_llm,
)
from src.extraction.extract_llm import extract_claims_llm, MockProvider
from src.extraction.extract_rule_based import extract_claims_rule_based

class LimitRequestSizeMiddleware(BaseHTTPMiddleware):
    """Global middleware to restrict the maximum allowed request body size."""
    def __init__(self, app, max_size_bytes: int):
        super().__init__(app)
        self.max_size_bytes = max_size_bytes

    async def dispatch(self, request: Request, call_next):
        if request.method in ("POST", "PUT", "PATCH"):
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    if int(content_length) > self.max_size_bytes:
                        return JSONResponse(
                            status_code=413,
                            content={"detail": "Payload too large"}
                        )
                except ValueError:
                    return JSONResponse(
                        status_code=400,
                        content={"detail": "Invalid Content-Length"}
                    )
        return await call_next(request)

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Verity", version="1.0.0")

# Register request size limiter middleware
max_request_size = int(os.environ.get("MAX_REQUEST_SIZE_BYTES", 2 * 1024 * 1024))
app.add_middleware(LimitRequestSizeMiddleware, max_size_bytes=max_request_size)

# Setup CORS with dynamic origins from the environment
allowed_origins_str = os.environ.get("ALLOWED_ORIGINS", "http://localhost:4200")
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(api_key_router)


# ── request / response models ────────────────────────────────────────────────

class VerifyRequest(BaseModel):
    answer: str = Field(..., min_length=10, max_length=50000, description="AI-generated answer to verify")
    context: str = Field(..., min_length=10, max_length=500000, description="Source document to verify against")
    provider: str = Field("groq", max_length=50, description="LLM provider: groq | anthropic | openai | gemini | mock")
    extractor: str = Field("rule_based", max_length=50, description="Claim extractor: rule_based | llm")


class ClaimResult(BaseModel):
    claim: str
    label: str
    evidence: str
    reasoning: str
    confidence: str


class VerifyResponse(BaseModel):
    run_id: int
    status: str
    error: str | None = None
    claims: list[ClaimResult] | None = None
    stats: dict | None = None
    remaining_quota: int


# ── provider factory ─────────────────────────────────────────────────────────

def _get_provider(name: str):
    name = name.lower()
    if name == "groq":
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise HTTPException(400, "GROQ_API_KEY not set")
        return GroqProvider()
    if name == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise HTTPException(400, "ANTHROPIC_API_KEY not set")
        return AnthropicProvider()
    if name == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise HTTPException(400, "OPENAI_API_KEY not set")
        return OpenAIProvider()
    if name == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise HTTPException(400, "GEMINI_API_KEY not set")
        return GeminiProvider()
    if name == "mock":
        return RuleBasedMockProvider()
    raise HTTPException(400, f"Unknown provider: {name}. Use groq | anthropic | openai | gemini | mock")


# ── endpoints ────────────────────────────────────────────────────────────────

from sqlalchemy import text

@app.get("/")
def root():
    return {
        "message": "Welcome to the Verity API. Visit /docs for interactive API documentation.",
        "status": "running"
    }

@app.get("/health")

def health(db: Session = Depends(get_db)):
    try:
        # Ping the database
        db.execute(text("SELECT 1"))
        return {"status": "ok", "database": "up", "service": "verity"}
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database connection failed: {str(e)}"
        )


def _verify_single_claim_sync(claim_text: str, chunks: list, retriever: TfidfRetriever, provider) -> ClaimResult:
    top = retriever.retrieve(claim_text, chunks, top_k=1)
    evidence_text = top[0].chunk.text if top else ""
    retrieval_score = top[0].score if top else 0.0

    try:
        result = verify_claim_llm(claim_text, evidence_text, provider=provider)
        label = result.label.value
        reasoning = result.reasoning
    except Exception as e:
        label = "unsupported"
        reasoning = f"Verification failed: {e}"

    if retrieval_score > 0.4:
        confidence = "high"
    elif retrieval_score > 0.15:
        confidence = "medium"
    else:
        confidence = "low"

    return ClaimResult(
        claim=claim_text,
        label=label,
        evidence=evidence_text,
        reasoning=reasoning,
        confidence=confidence,
    )


def _save_run_sync(db: Session, run: models.VerificationRun):
    db.add(run)
    db.commit()


@app.post("/verify", response_model=VerifyResponse)
async def verify(
    req: VerifyRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    # 0. Check usage limits (10 verifications / day)
    from datetime import datetime, timedelta
    twenty_four_hours_ago = datetime.utcnow() - timedelta(hours=24)
    run_count = db.query(models.VerificationRun).filter(
        models.VerificationRun.user_id == current_user.id,
        models.VerificationRun.created_at >= twenty_four_hours_ago
    ).count()
    if run_count >= 10:
        raise HTTPException(
            status_code=403,
            detail="Daily verification limit of 10 reached. Please try again tomorrow."
        )

    # 1. Create a verification run entry with pending status
    run = models.VerificationRun(
        user_id=current_user.id,
        provider=req.provider,
        answer=req.answer,
        context=req.context,
        claims=[],
        stats={},
        status="pending",
        error=None
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    # 2. Trigger background task via FastAPI BackgroundTasks
    background_tasks.add_task(verify_claims_task, run.id, req.extractor)


    # 3. Compute remaining quota
    remaining = max(0, 10 - (run_count + 1))
    return VerifyResponse(
        run_id=run.id,
        status=run.status,
        claims=[],
        stats={},
        remaining_quota=remaining
    )


@app.get("/runs", response_model=list[RunSummary])
def get_runs(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    return db.query(models.VerificationRun)\
        .filter(models.VerificationRun.user_id == current_user.id)\
        .order_by(models.VerificationRun.created_at.desc())\
        .all()


@app.get("/runs/{run_id}", response_model=RunDetail)
def get_run(
    run_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    run = db.query(models.VerificationRun)\
        .filter(
            models.VerificationRun.id == run_id,
            models.VerificationRun.user_id == current_user.id,
        ).first()
    if not run:
        raise HTTPException(404, "Run not found")
    return run


@app.get("/runs/{run_id}/pdf")
def get_run_pdf(
    run_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    from fastapi import Response
    run = db.query(models.VerificationRun).filter(
        models.VerificationRun.id == run_id,
        models.VerificationRun.user_id == current_user.id,
    ).first()
    if not run:
        raise HTTPException(404, "Run not found")
    if run.status != "completed":
        raise HTTPException(400, f"Run PDF is not ready (status: {run.status})")

    pdf_bytes = generate_verification_pdf(run)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=verity_report_{run_id}.pdf"}
    )


@app.get("/dashboard/stats", response_model=DashboardStatsResponse)
def get_dashboard_stats(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    # Fetch all completed runs for user
    runs = db.query(models.VerificationRun).filter(
        models.VerificationRun.user_id == current_user.id,
        models.VerificationRun.status == "completed"
    ).all()

    total_runs = len(runs)
    total_claims = 0
    supported_claims = 0
    contradicted_claims = 0
    unsupported_claims = 0

    for r in runs:
        stats = r.stats or {}
        total_claims += stats.get("total", 0)
        supported_claims += stats.get("supported", 0)
        contradicted_claims += stats.get("contradicted", 0)
        unsupported_claims += stats.get("unsupported", 0)

    avg_accuracy = (supported_claims / total_claims) if total_claims > 0 else 0.0

    breakdown = {
        "supported": supported_claims,
        "contradicted": contradicted_claims,
        "unsupported": unsupported_claims
    }

    # Chronological history of recent 10 completed runs
    # Format dates as e.g. "Jun 26"
    sorted_runs = sorted(runs, key=lambda x: x.created_at or datetime.min)
    recent_runs = sorted_runs[-10:]

    history_list = []
    for r in recent_runs:
        r_total = r.stats.get("total", 0)
        r_supp = r.stats.get("supported", 0)
        accuracy = (r_supp / r_total * 100) if r_total > 0 else 0.0

        history_list.append({
            "id": r.id,
            "date": r.created_at.strftime("%b %d, %H:%M") if r.created_at else "",
            "accuracy": round(accuracy, 1),
            "claims_count": r_total
        })

    return DashboardStatsResponse(
        total_runs=total_runs,
        average_accuracy=round(avg_accuracy * 100, 1),
        hallucinations_breakdown=breakdown,
        activity_history=history_list
    )