import json
from api.celery_app import celery_app
from api.database import SessionLocal
from api import models

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


def _get_provider(name: str):
    name = name.lower()
    if name == "groq":
        return GroqProvider()
    if name == "anthropic":
        return AnthropicProvider()
    if name == "openai":
        return OpenAIProvider()
    if name == "gemini":
        return GeminiProvider()
    return RuleBasedMockProvider()


@celery_app.task(name="api.tasks.verify_claims_task")
def verify_claims_task(run_id: int, extractor: str):
    db = SessionLocal()
    try:
        run = db.query(models.VerificationRun).filter(models.VerificationRun.id == run_id).first()
        if not run:
            return

        run.status = "processing"
        db.commit()

        # 1. extract claims
        if extractor == "llm":
            raw_claims = extract_claims_llm(run.answer, provider=MockProvider())
            claim_texts = [c.text for c in raw_claims]
        else:
            result = extract_claims_rule_based(run.answer)
            claim_texts = [c.text for c in result.claims]

        if not claim_texts:
            run.status = "failed"
            run.error = "No claims could be extracted from the answer."
            db.commit()
            return

        # 2. chunk the source context
        chunks = chunk_document(run.context)
        if not chunks:
            run.status = "failed"
            run.error = "Could not chunk the source context."
            db.commit()
            return

        # 3. retrieve + verify each claim
        retriever = TfidfRetriever()
        provider = _get_provider(run.provider)

        results = []
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

            if retrieval_score > 0.4:
                confidence = "high"
            elif retrieval_score > 0.15:
                confidence = "medium"
            else:
                confidence = "low"

            label_counts[label] = label_counts.get(label, 0) + 1
            results.append({
                "claim": claim_text,
                "label": label,
                "evidence": evidence_text,
                "reasoning": reasoning,
                "confidence": confidence,
            })

        stats = {
            "total": len(results),
            "supported": label_counts.get("supported", 0),
            "contradicted": label_counts.get("contradicted", 0),
            "unsupported": label_counts.get("unsupported", 0),
        }

        # Save results back to run
        run.claims = results
        run.stats = stats
        run.status = "completed"
        db.commit()

    except Exception as e:
        run.status = "failed"
        run.error = str(e)
        db.commit()
    finally:
        db.close()
