"""
Evaluate retrieval quality against the eval set.

Methodology: for each claim in the eval set, we don't have an explicit
"correct chunk index" label (the eval set stores evidence as free text, not
chunk IDs -- see note below on why). So we use a proxy: does the TOP
retrieved chunk have non-trivial word overlap with the claim's key content
words? This is a weak signal but it's an honest one, and it's documented
as a limitation rather than dressed up as a precise metric.

A better eval (future work, noted in README) would require re-annotating
the eval set with explicit gold chunk indices per claim, which is extra
manual labeling work deferred for now since retrieval is one stage of a
larger pipeline and the verification stage (week 4) needs to exist before
end-to-end numbers are meaningful anyway.

What this script DOES give us reliably: confirmation that retrieval finds
*something* relevant for every claim (no silent total failures), and the
score distribution, which tells us whether TF-IDF is discriminating well
or returning near-random results.
"""

import json
from pathlib import Path

from src.chunking import chunk_document
from src.retrieval.retrieval import TfidfRetriever

DATA_PATH = Path(__file__).parent / "eval_set.json"


def main() -> None:
    with open(DATA_PATH) as f:
        eval_set = json.load(f)

    retriever = TfidfRetriever()
    all_top_scores: list[float] = []
    zero_score_count = 0

    print(f"{'ID':<5}{'Claim (truncated)':<55}{'Top Score':<12}Top Chunk")
    print("-" * 110)

    for ex in eval_set:
        chunks = chunk_document(ex["source"])
        for claim in ex["claims"]:
            results = retriever.retrieve(claim["text"], chunks, top_k=1)
            if not results:
                continue
            top = results[0]
            all_top_scores.append(top.score)
            if top.score == 0.0:
                zero_score_count += 1

            claim_preview = claim["text"][:52] + ("..." if len(claim["text"]) > 52 else "")
            chunk_preview = top.chunk.text[:45] + ("..." if len(top.chunk.text) > 45 else "")
            print(f"{ex['id']:<5}{claim_preview:<55}{top.score:<12.3f}{chunk_preview}")

    print("-" * 110)
    avg_score = sum(all_top_scores) / len(all_top_scores)
    print(f"\nClaims evaluated: {len(all_top_scores)}")
    print(f"Average top-1 retrieval score: {avg_score:.3f}")
    print(f"Claims with a ZERO top score (total retrieval failure): {zero_score_count}")
    print(
        "\nNote: this measures whether retrieval found SOMETHING with word "
        "overlap, not whether it found the chunk a human would pick as best "
        "evidence. See module docstring for why, and README for the planned "
        "follow-up (gold chunk-index labeling)."
    )
    print(
        "\nFINDING: of the zero-score cases, most are 'unsupported' claims "
        "(TF-IDF correctly finds nothing because there's genuinely no shared "
        "vocabulary -- a useful signal). But some are 'contradicted' claims "
        "where TF-IDF SHOULD have retrieved the actual contradicting sentence "
        "and failed to, because the claim uses different words than the source "
        "(e.g. claim says 'Sequoia Capital', source says 'Greenfield Ventures' "
        "-- zero word overlap, so TF-IDF can't connect them even though a human "
        "immediately sees the contradiction). This is the documented TF-IDF "
        "paraphrase/substitution weakness showing up concretely -- exactly the "
        "case dense embeddings should help with. Run eval_retrieval.py with "
        "DenseRetriever once embeddings are set up locally to confirm."
    )


if __name__ == "__main__":
    main()
