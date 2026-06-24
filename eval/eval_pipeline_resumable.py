"""Resumable full pipeline eval that writes per-claim checkpoint results.

This is meant for real API-backed providers where quota/rate limits can
interrupt a long run. Each completed claim is saved immediately to
eval/checkpoints/results_<provider>.json, so a later run resumes where the
previous one stopped.
"""

import json
from collections import Counter
from pathlib import Path

from eval.eval_pipeline import _load_eval_set, _provider_from_env
from src.chunking import chunk_document
from src.retrieval.retrieval import TfidfRetriever
from src.verification.verify_llm import verify_claim_llm

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


def _load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _save_checkpoint(path: Path, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    with open(temp_path, "w") as f:
        json.dump(results, f, indent=2)
    temp_path.replace(path)


def main() -> None:
    eval_set = _load_eval_set()
    provider, provider_label = _provider_from_env()
    provider_name = provider_label.replace("Provider", "").replace("RuleBasedMock", "mock").lower()
    checkpoint_path = CHECKPOINT_DIR / f"results_{provider_name}.json"
    results = _load_checkpoint(checkpoint_path)

    retriever = TfidfRetriever()
    total_claims = sum(len(ex["claims"]) for ex in eval_set)

    print(f"=== Resumable eval ({provider_label}) ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Already completed: {len(results)}/{total_claims}")

    for ex in eval_set:
        chunks = chunk_document(ex["source"])
        for claim_index, claim in enumerate(ex["claims"]):
            result_id = f"{ex['id']}:{claim_index}"
            if result_id in results:
                continue

            top_evidence = retriever.retrieve(claim["text"], chunks, top_k=1)
            evidence_text = top_evidence[0].chunk.text if top_evidence else ""

            try:
                result = verify_claim_llm(claim["text"], evidence_text, provider=provider)
            except Exception:
                _save_checkpoint(checkpoint_path, results)
                print(f"Stopped after {len(results)}/{total_claims}; checkpoint preserved.")
                raise

            results[result_id] = {
                "id": ex["id"],
                "claim": claim["text"],
                "true_label": claim["label"],
                "predicted_label": result.label.value,
                "evidence_used": evidence_text,
                "reasoning": result.reasoning,
            }
            _save_checkpoint(checkpoint_path, results)

            if len(results) % 5 == 0:
                print(f"Processed {len(results)}/{total_claims} claims...", flush=True)

    confusion: Counter = Counter(
        (r["true_label"], r["predicted_label"]) for r in results.values()
    )
    correct = sum(1 for r in results.values() if r["true_label"] == r["predicted_label"])
    total = len(results)
    accuracy = correct / total if total else 0.0

    print(f"\nCompleted: {correct}/{total} = {accuracy:.1%}")
    print("Confusion matrix (true -> predicted):")
    labels = ["supported", "contradicted", "unsupported"]
    header = f"{'':<15}" + "".join(f"{p:<15}" for p in labels)
    print(header)
    for true_label in labels:
        row = f"{true_label:<15}"
        for predicted_label in labels:
            row += f"{confusion.get((true_label, predicted_label), 0):<15}"
        print(row)


if __name__ == "__main__":
    main()
