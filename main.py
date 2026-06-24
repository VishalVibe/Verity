"""Command-line entry point for the Verity claim-verification pipeline."""

import argparse
import json
from dataclasses import asdict

from src.chunking import chunk_document
from src.extraction.extract_rule_based import extract_claims_rule_based
from src.retrieval.retrieval import TfidfRetriever
from src.verification.verify_llm import verify_claim_llm


def run_pipeline(answer: str, source: str) -> list[dict]:
    """Extract claims, retrieve evidence, and verify each claim."""
    claims = extract_claims_rule_based(answer).claims
    chunks = chunk_document(source)
    retriever = TfidfRetriever()
    results = []

    for claim in claims:
        matches = retriever.retrieve(claim.text, chunks, top_k=1)
        evidence = matches[0].chunk.text if matches else ""
        verification = verify_claim_llm(claim.text, evidence)
        item = asdict(verification)
        item["label"] = verification.label.value
        item["start_char"] = claim.start_char
        item["end_char"] = claim.end_char
        results.append(item)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify claims in an answer against a source.")
    parser.add_argument("--answer", required=True, help="AI-generated answer to check")
    parser.add_argument("--source", required=True, help="Source text used as evidence")
    args = parser.parse_args()
    print(json.dumps(run_pipeline(args.answer, args.source), indent=2))


if __name__ == "__main__":
    main()
