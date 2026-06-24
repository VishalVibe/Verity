"""
Run the rule-based extractor across the full eval set and report how many
claims it produces per example, compared to the ground-truth claim count.

This isn't a full accuracy eval (that requires matching extracted claims
back to ground-truth claims, which is non-trivial -- see notes at bottom).
It's a first sanity check: is the extractor in the right ballpark, or wildly
over/under-segmenting?
"""

import json
from pathlib import Path

from src.extraction.extract_rule_based import extract_claims_rule_based

DATA_PATH = Path(__file__).parent / "eval_set.json"


def main() -> None:
    with open(DATA_PATH) as f:
        eval_set = json.load(f)

    total_ground_truth = 0
    total_extracted = 0
    exact_matches = 0

    print(f"{'ID':<5}{'Ground Truth':<15}{'Extracted':<12}Match?")
    print("-" * 45)

    for ex in eval_set:
        gt_count = len(ex["claims"])
        result = extract_claims_rule_based(ex["answer"])
        extracted_count = len(result)

        total_ground_truth += gt_count
        total_extracted += extracted_count
        match = gt_count == extracted_count
        exact_matches += match

        print(f"{ex['id']:<5}{gt_count:<15}{extracted_count:<12}{'✓' if match else '✗'}")

    print("-" * 45)
    print(f"Total ground-truth claims: {total_ground_truth}")
    print(f"Total extracted claims:    {total_extracted}")
    print(f"Examples with exact count match: {exact_matches}/{len(eval_set)}")
    print(
        "\nNote: matching COUNT is a weak proxy -- the extractor could get the "
        "right number of claims but split them in the wrong places. A proper "
        "eval needs claim-level alignment, which we'll add once the LLM-based "
        "extractor exists to compare against."
    )


if __name__ == "__main__":
    main()
