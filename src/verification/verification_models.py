"""Shared data structures for verification."""

from dataclasses import dataclass
from enum import Enum

# Shared by LLM and NLI verification implementations.


class VerificationLabel(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"


@dataclass
class VerificationResult:
    """Output of a verifier for one claim/evidence pair."""

    claim_text: str
    evidence_text: str
    label: VerificationLabel
    reasoning: str
    method: str
    confidence: float | None = None
