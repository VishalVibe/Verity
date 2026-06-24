"""
Verity — FastAPI verification server.

Exposes one endpoint:
  POST /verify   { answer: str, context: str, provider?: str }
  -> { claims: [...], stats: { total, supported, contradicted, unsupported } }

Run from the project root:
  uvicorn api.server:app --reload --port 8000

CORS is open for localhost:4200 (Angular dev server).
"""
from dotenv import load_dotenv
load_dotenv()

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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

app = FastAPI(title="Verity", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── request / response models ────────────────────────────────────────────────

class VerifyRequest(BaseModel):
    answer: str = Field(..., min_length=10, description="AI-generated answer to verify")
    context: str = Field(..., min_length=10, description="Source document to verify against")
    provider: str = Field("groq", description="LLM provider: groq | anthropic | openai | gemini | mock")
    extractor: str = Field("rule_based", description="Claim extractor: rule_based | llm")


class ClaimResult(BaseModel):
    claim: str
    label: str          # supported | contradicted | unsupported
    evidence: str
    reasoning: str
    confidence: str     # high | medium | low  (derived from label + retrieval score)


class VerifyResponse(BaseModel):
    claims: list[ClaimResult]
    stats: dict


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

@app.get("/health")
def health():
    return {"status": "ok", "service": "verity"}


@app.post("/verify", response_model=VerifyResponse)
def verify(req: VerifyRequest):
    # 1. extract claims
    if req.extractor == "llm":
        raw_claims = extract_claims_llm(req.answer, provider=MockProvider())
        claim_texts = [c.text for c in raw_claims]
    else:
        result = extract_claims_rule_based(req.answer)
        claim_texts = [c.text for c in result.claims]

    if not claim_texts:
        raise HTTPException(422, "No claims could be extracted from the answer.")

    # 2. chunk the source context
    chunks = chunk_document(req.context)
    if not chunks:
        raise HTTPException(422, "Could not chunk the source context.")

    # 3. retrieve + verify each claim
    retriever = TfidfRetriever()
    provider = _get_provider(req.provider)

    results: list[ClaimResult] = []
    label_counts = {"supported": 0, "contradicted": 0, "unsupported": 0}

    for claim_text in claim_texts:
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

        # derive a simple confidence signal from retrieval score + label
        if retrieval_score > 0.4:
            confidence = "high"
        elif retrieval_score > 0.15:
            confidence = "medium"
        else:
            confidence = "low"

        label_counts[label] = label_counts.get(label, 0) + 1
        results.append(ClaimResult(
            claim=claim_text,
            label=label,
            evidence=evidence_text,
            reasoning=reasoning,
            confidence=confidence,
        ))

    stats = {
        "total": len(results),
        "supported": label_counts["supported"],
        "contradicted": label_counts["contradicted"],
        "unsupported": label_counts["unsupported"],
    }

    return VerifyResponse(claims=results, stats=stats)