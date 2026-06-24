"""
Shared data structures for the claim extraction pipeline.

Keeping this in its own module means both extractor implementations
(rule-based and LLM-based) return the same shape, so the rest of the
pipeline (retrieval, verification) doesn't need to know which one was used.
"""

from dataclasses import dataclass, field

# Shared by both extraction implementations.


@dataclass
class Claim:
    """A single atomic, checkable claim extracted from an AI-generated answer."""

    text: str
    # Character offsets of this claim's source span in the original answer.
    # Useful later for highlighting the claim in a UI.
    start_char: int
    end_char: int

    def __repr__(self) -> str:
        return f"Claim({self.text!r})"


@dataclass
class ExtractionResult:
    """Output of running a claim extractor over one answer."""

    answer: str
    claims: list[Claim] = field(default_factory=list)
    method: str = "unknown"  # "rule_based" or "llm"

    def __len__(self) -> int:
        return len(self.claims)
