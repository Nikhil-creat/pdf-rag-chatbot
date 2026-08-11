"""
Embedding model wrapper.

Uses sentence-transformers so the whole "understand meaning" half of the
pipeline runs locally, offline, and free — only the final answer-generation
call touches a paid API. This is a deliberate cost/architecture choice
worth mentioning in interviews: embeddings are the highest-volume calls
in a RAG system (every chunk, every query), so keeping them local avoids
both latency and API cost scaling with corpus size.
"""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts into L2-normalized embeddings.

        Normalizing lets us use a plain inner-product FAISS index and have
        it behave as cosine similarity, which is simpler and faster than
        computing cosine distance manually at query time.
        """
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype("float32")

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str) -> EmbeddingModel:
    """Cached so Streamlit reruns don't reload the model from disk every time."""
    return EmbeddingModel(model_name)
