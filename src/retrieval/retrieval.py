"""
Retrieval: given a claim and a chunked source document, find the chunk(s)
most likely to contain evidence for or against the claim.

Two retrieval methods, same provider-abstraction pattern as extract_llm.py:

  - TfidfRetriever: classical sparse retrieval (term frequency weighting).
    Free, deterministic, no model download, no API call. This is a REAL
    baseline used in production systems for keyword-heavy matching, not a
    placeholder -- it's known to struggle with paraphrase (e.g. claim says
    "constructed", source says "built") since it only matches surface
    word forms.

  - DenseRetriever: embedding-based semantic search. Handles paraphrase
    much better than TF-IDF because it compares meaning, not just word
    overlap, but requires a model (local sentence-transformers or an API
    embedding call) and is non-trivial to run in environments without
    package/network access.

Both implement the same `retrieve(claim_text, chunks, top_k)` interface so
the rest of the pipeline doesn't care which one is in use, and so week 5's
comparison eval can swap between them with one line changed.
"""

from abc import ABC, abstractmethod

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.retrieval.retrieval_models import Chunk, RetrievalResult


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, claim_text: str, chunks: list[Chunk], top_k: int = 2) -> list[RetrievalResult]:
        """Return the top_k chunks most relevant to claim_text, scored and sorted descending."""
        raise NotImplementedError


class TfidfRetriever(Retriever):
    """Sparse retrieval using TF-IDF + cosine similarity.

    Fits a fresh TF-IDF vocabulary per call, scoped to just this document's
    chunks plus the claim. This is the right scope for our use case (we're
    always retrieving within one document's chunks, not a large corpus) and
    avoids needing a pre-built global vocabulary.
    """

    def retrieve(self, claim_text: str, chunks: list[Chunk], top_k: int = 2) -> list[RetrievalResult]:
        if not chunks:
            return []

        corpus = [c.text for c in chunks] + [claim_text]
        vectorizer = TfidfVectorizer(stop_words="english")

        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
        except ValueError:
            # Happens if the claim/chunks share zero vocabulary after
            # stop-word removal (e.g. all-stopword claim) -- fall back to
            # uniform low scores rather than crashing.
            return [RetrievalResult(chunk=c, score=0.0) for c in chunks[:top_k]]

        claim_vector = tfidf_matrix[-1]
        chunk_vectors = tfidf_matrix[:-1]
        similarities = cosine_similarity(claim_vector, chunk_vectors)[0]

        ranked_indices = np.argsort(similarities)[::-1][:top_k]
        return [
            RetrievalResult(chunk=chunks[i], score=float(similarities[i]))
            for i in ranked_indices
        ]


class EmbeddingProvider(ABC):
    """Interface for turning text into a dense vector."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n_texts, dim) array of embeddings."""
        raise NotImplementedError


class SentenceTransformerProvider(EmbeddingProvider):
    """Local, free, offline embeddings via sentence-transformers.

    Requires: pip install sentence-transformers
    Not installed in this sandboxed environment (no network access to
    download the model), so this is provided for you to run on your own
    machine -- see README for setup instructions.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """API-based embeddings via OpenAI. Requires OPENAI_API_KEY and `pip install openai`."""

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model

    def embed(self, texts: list[str]) -> np.ndarray:
        import os

        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.embeddings.create(model=self.model, input=texts)
        return np.array([item.embedding for item in response.data])


class DenseRetriever(Retriever):
    """Semantic retrieval using dense embeddings + cosine similarity."""

    def __init__(self, provider: EmbeddingProvider):
        self.provider = provider

    def retrieve(self, claim_text: str, chunks: list[Chunk], top_k: int = 2) -> list[RetrievalResult]:
        if not chunks:
            return []

        chunk_texts = [c.text for c in chunks]
        chunk_embeddings = self.provider.embed(chunk_texts)
        claim_embedding = self.provider.embed([claim_text])

        similarities = cosine_similarity(claim_embedding, chunk_embeddings)[0]
        ranked_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            RetrievalResult(chunk=chunks[i], score=float(similarities[i]))
            for i in ranked_indices
        ]


if __name__ == "__main__":
    from src.chunking import chunk_document

    source = (
        "The Eiffel Tower was constructed between 1887 and 1889 as the entrance "
        "arch for the 1889 World's Fair in Paris. Designed by engineer Gustave "
        "Eiffel's company, it stands 330 meters tall. The tower is built from "
        "wrought iron and weighs approximately 10,100 tonnes."
    )
    chunks = chunk_document(source)

    claim = "The Eiffel Tower is 324 meters tall."  # deliberately wrong height
    retriever = TfidfRetriever()
    results = retriever.retrieve(claim, chunks, top_k=2)

    print(f"Claim: {claim}\n")
    print("Top retrieved chunks:")
    for r in results:
        print(f"  score={r.score:.3f}  {r.chunk.text!r}")
