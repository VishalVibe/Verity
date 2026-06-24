"""FastAPI wrapper around the Verity pipeline."""

from fastapi import FastAPI
from pydantic import BaseModel

from main import run_pipeline

app = FastAPI(title="Verity", version="1.0.0")


class VerifyRequest(BaseModel):
    answer: str
    source: str


class VerifyResponse(BaseModel):
    results: list[dict]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/verify", response_model=VerifyResponse)
def verify(request: VerifyRequest) -> VerifyResponse:
    return VerifyResponse(results=run_pipeline(request.answer, request.source))
