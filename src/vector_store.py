"""
A thin, dependency-minimal wrapper around a FAISS index.

FAISS stores only vectors + integer ids — it knows nothing about our text
or metadata. So this class keeps a parallel Python list (`_records`) that
maps each vector's position back to its original chunk text, page number,
and source file. That mapping is the whole trick to making a vector DB
"queryable" in a useful way.
"""

from dataclasses import dataclass

import faiss
import numpy as np


@dataclass
class SearchResult:
    text: str
    page: int
    source: str
    score: float


class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        # Inner product on normalized vectors == cosine similarity.
        self.index = faiss.IndexFlatIP(dimension)
        self._records: list[dict] = []

    def add(self, embeddings: np.ndarray, records: list[dict]) -> None:
        """embeddings: (n, dim) float32 array. records: parallel list of
        {"text": ..., "page": ..., "source": ...} dicts."""
        if len(records) != embeddings.shape[0]:
            raise ValueError("embeddings and records length mismatch")
        self.index.add(embeddings)
        self._records.extend(records)

    def search(self, query_embedding: np.ndarray, top_k: int, min_score: float = 0.0) -> list[SearchResult]:
        if self.index.ntotal == 0:
            return []
        query = query_embedding.reshape(1, -1).astype("float32")
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < min_score:
                continue
            record = self._records[idx]
            results.append(SearchResult(
                text=record["text"],
                page=record["page"],
                source=record["source"],
                score=float(score),
            ))
        return results

    @property
    def is_empty(self) -> bool:
        return self.index.ntotal == 0

    def reset(self) -> None:
        self.index = faiss.IndexFlatIP(self.dimension)
        self._records = []
