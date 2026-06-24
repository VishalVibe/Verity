"""Shared data structures for retrieval."""

from dataclasses import dataclass

# Shared by sparse and dense retrievers.


@dataclass
class Chunk:
    """A retrievable source-document span."""

    text: str
    chunk_id: int
    start_char: int
    end_char: int


@dataclass
class RetrievalResult:
    """A retrieved chunk and its similarity score."""

    chunk: Chunk
    score: float
