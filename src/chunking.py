"""
Chunk source documents into retrievable units.

Decision: chunk at the SENTENCE level, not paragraph level.

Our eval set sources are short single paragraphs (3-5 sentences). If we
chunked at the paragraph level, retrieval would be trivial -- there'd be
only one chunk to "retrieve", so the retrieval step wouldn't be doing real
work. Sentence-level chunking forces retrieval to actually discriminate
between multiple candidate chunks per source, which is a more honest test
of the pipeline and closer to how chunking works in practice on longer
documents anyway (you rarely want to hand an LLM judge a whole multi-page
document as "evidence" for one claim).

Trade-off: sentence-level chunks can lose cross-sentence context (e.g. a
pronoun in sentence 3 referring to a noun in sentence 1). We accept this
for now and note it as a limitation -- a production system would likely
use a sliding window of 1-2 sentences with overlap instead of single
sentences. Left as a documented future improvement, not silently ignored.
"""

import re

from src.retrieval.retrieval_models import Chunk

# Reuse the same sentence-boundary heuristic as the rule-based claim
# extractor, kept here as its own copy since chunking source documents and
# extracting claims from answers are conceptually different jobs even
# though the underlying regex is the same.
_SENTENCE_BOUNDARY = re.compile(r"(?<!\b[A-Z])(?<=[.!?])\s+(?=[A-Z])")


def chunk_document(source_text: str) -> list[Chunk]:
    """Split a source document into sentence-level chunks.

    Returns chunks with their original character offsets in `source_text`,
    so later pipeline stages can point back to the exact source span used
    as evidence for a claim.
    """
    chunks: list[Chunk] = []
    pos = 0
    chunk_id = 0

    for part in _SENTENCE_BOUNDARY.split(source_text):
        part = part.strip()
        if not part:
            continue
        start = source_text.index(part, pos)
        end = start + len(part)
        pos = end
        chunks.append(Chunk(text=part, chunk_id=chunk_id, start_char=start, end_char=end))
        chunk_id += 1

    return chunks


if __name__ == "__main__":
    example = (
        "The Eiffel Tower was constructed between 1887 and 1889 as the entrance "
        "arch for the 1889 World's Fair in Paris. Designed by engineer Gustave "
        "Eiffel's company, it stands 330 meters tall. The tower is built from "
        "wrought iron and weighs approximately 10,100 tonnes."
    )
    chunks = chunk_document(example)
    print(f"Produced {len(chunks)} chunks:\n")
    for c in chunks:
        print(f"  [{c.chunk_id}] {c.text}")
