import time
from collections import defaultdict
from threading import Lock
from fastapi import Request, HTTPException, status

class InMemoryRateLimiter:
    """Thread-safe sliding window rate limiter."""
    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
        self.lock = Lock()

    def check_limit(self, key: str) -> bool:
        now = time.time()
        with self.lock:
            # Keep only the timestamps within the current sliding window
            self.requests[key] = [
                t for t in self.requests[key]
                if now - t < self.window_seconds
            ]
            if len(self.requests[key]) >= self.limit:
                return False
            self.requests[key].append(now)
            return True

# Allow 5 login attempts per 60 seconds per IP address
login_limiter = InMemoryRateLimiter(limit=5, window_seconds=60)

def get_client_ip(request: Request) -> str:
    """Extract real client IP address, checking proxy forwarding headers."""
    # Check X-Forwarded-For (list of IPs: client, proxy1, proxy2, ...)
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    # Check X-Real-IP
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip
        
    return request.client.host if request.client else "unknown"

async def rate_limit_login(request: Request):
    """FastAPI dependency to rate limit requests on login endpoint."""
    # In some testing environments, we might want to skip rate limits.
    # We can handle that via a config check if needed, but standard unit tests can run normally.
    ip = get_client_ip(request)
    if not login_limiter.check_limit(ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts. Please try again later."
        )
