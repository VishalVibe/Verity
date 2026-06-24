"""Smoke tests for the assembled pipeline."""

from main import run_pipeline
from src.extraction.extract_rule_based import extract_claims_rule_based


def test_rule_based_extraction_smoke() -> None:
    result = extract_claims_rule_based("Paris is the capital of France.")
    assert len(result.claims) == 1
    assert result.claims[0].text == "Paris is the capital of France."


def test_pipeline_smoke() -> None:
    results = run_pipeline(
        answer="Paris is the capital of France.",
        source="Paris is the capital and largest city of France.",
    )
    assert len(results) == 1
    assert results[0]["label"] in {"supported", "contradicted", "unsupported"}
