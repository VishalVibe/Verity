"""
Full pipeline eval: for every (question, source, answer) in the eval set,
extract ground-truth claims (we use the LABELED claims directly here, not
the extractor's output, to isolate verification quality from extraction
quality -- see note below), retrieve evidence with TF-IDF, verify with the
LLM judge, and compare predicted labels against ground truth.

WHY we use ground-truth claims instead of extractor output for this eval:
this script's job is to measure RETRIEVAL + VERIFICATION accuracy in
isolation. If we fed it the rule-based extractor's output instead, a
verification error and an extraction error would be tangled together and
we wouldn't know which stage to blame for a wrong final label. Once all
three stages exist, a separate full-pipeline-from-scratch eval (using
extracted, not labeled, claims) is the right way to measure true end-to-end
performance -- noted as remaining work in the README.

This script uses RuleBasedMockProvider for the LLM judge by default (free,
runnable here) -- swap in AnthropicProvider/OpenAIProvider for the real
number. The mock numbers below are a lower-bound sanity check, NOT the
real result -- treat any "good" mock score with suspicion, since the mock
is deliberately crude.
"""

import json
from collections import Counter
from pathlib import Path

from src.chunking import chunk_document
from src.retrieval.retrieval import TfidfRetriever
from src.verification.verify_llm import (
    AnthropicProvider,
    GeminiProvider,
    OpenAIProvider,
    GroqProvider, 
    verify_claim_llm,
)

DATA_PATH_CANDIDATES = [
    Path(__file__).parent / "eval_set.json",
]


def _load_eval_set() -> list[dict]:
    for path in DATA_PATH_CANDIDATES:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    searched = ", ".join(str(path) for path in DATA_PATH_CANDIDATES)
    raise FileNotFoundError(f"Could not find eval_set.json. Searched: {searched}")


def _provider_from_env():
    provider_name = __import__("os").environ.get("VERIFY_PROVIDER", "mock").lower()
    if provider_name == "anthropic":
        return AnthropicProvider(), "AnthropicProvider"
    if provider_name == "openai":
        return OpenAIProvider(), "OpenAIProvider"
    if provider_name == "gemini":
        return GeminiProvider(), "GeminiProvider"
    if provider_name == "groq":                          # ← add this block
        from src.verification.verify_llm import GroqProvider
        return GroqProvider(), "GroqProvider"
    if provider_name == "mock":
        return None, "RuleBasedMockProvider"
    raise ValueError("VERIFY_PROVIDER must be one of: mock, anthropic, openai, gemini, groq")


def main() -> None:
    eval_set = _load_eval_set()
    provider, provider_label = _provider_from_env()

    retriever = TfidfRetriever()

    correct = 0
    total = 0
    confusion: Counter = Counter()  # (true_label, predicted_label) -> count
    errors_by_label: dict[str, list] = {"supported": [], "contradicted": [], "unsupported": []}
    total_claims = sum(len(ex["claims"]) for ex in eval_set)

    for ex in eval_set:
        chunks = chunk_document(ex["source"])
        for claim in ex["claims"]:
            true_label = claim["label"]

            top_evidence = retriever.retrieve(claim["text"], chunks, top_k=1)
            evidence_text = top_evidence[0].chunk.text if top_evidence else ""

            result = verify_claim_llm(claim["text"], evidence_text, provider=provider)
            predicted_label = result.label.value

            total += 1
            if provider_label != "RuleBasedMockProvider" and total % 5 == 0:
                print(f"Processed {total}/{total_claims} claims...", flush=True)
            confusion[(true_label, predicted_label)] += 1
            if predicted_label == true_label:
                correct += 1
            else:
                errors_by_label[true_label].append(
                    {
                        "id": ex["id"],
                        "claim": claim["text"],
                        "true": true_label,
                        "predicted": predicted_label,
                        "evidence_used": evidence_text,
                    }
                )

    accuracy = correct / total
    print(f"=== End-to-end pipeline eval (TF-IDF retrieval + {provider_label}) ===\n")
    print(f"Overall accuracy: {correct}/{total} = {accuracy:.1%}\n")

    print("Confusion matrix (true -> predicted):")
    labels = ["supported", "contradicted", "unsupported"]
    header = f"{'':<15}" + "".join(f"{p:<15}" for p in labels)
    print(header)
    for t in labels:
        row = f"{t:<15}"
        for p in labels:
            row += f"{confusion.get((t, p), 0):<15}"
        print(row)

    print("\nPer-label recall (of the claims with this TRUE label, % predicted correctly):")
    for label in labels:
        label_total = sum(v for (t, p), v in confusion.items() if t == label)
        label_correct = confusion.get((label, label), 0)
        recall = label_correct / label_total if label_total else 0.0
        print(f"  {label:<15}: {label_correct}/{label_total} = {recall:.1%}")

    print(
        "\nThis is the MOCK baseline (crude word-overlap heuristics, not a real LLM). "
        "Expected to be mediocre, especially on 'unsupported' vs 'contradicted' "
        "discrimination, which requires real semantic understanding the mock doesn't "
        "have. Set VERIFY_PROVIDER=anthropic, VERIFY_PROVIDER=openai, or VERIFY_PROVIDER=gemini for the real "
        "LLM judge number."
        if provider_label == "RuleBasedMockProvider"
        else "\nThis is the REAL LLM judge result. Compare it against the 59.8% mock floor."
    )

    print(f"\nSample errors (first 3 per true-label category):")
    for label, errs in errors_by_label.items():
        print(f"\n  -- true label: {label} --")
        for e in errs[:3]:
            print(f"     [{e['id']}] claim={e['claim']!r}")
            print(f"          predicted={e['predicted']}, evidence_used={e['evidence_used']!r}")


if __name__ == "__main__":
    main()
