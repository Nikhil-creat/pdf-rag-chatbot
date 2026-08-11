"""
Central configuration for the AI PDF Chatbot (RAG).

All values can be overridden via environment variables (see .env.example).
Keeping every tunable in one place makes it easy to explain design
decisions in an interview: "why 800-token chunks?", "why top-k=4?", etc.
"""

import os
from dataclasses import dataclass


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    # --- Chunking ---
    # Character-based chunking (not token-based) keeps the pipeline
    # dependency-free. ~800 chars ≈ 150-200 tokens, a good balance between
    # retrieval precision (small chunks) and context per chunk (large chunks).
    chunk_size: int = _get_int("CHUNK_SIZE", 800)
    chunk_overlap: int = _get_int("CHUNK_OVERLAP", 150)

    # --- Embeddings ---
    # all-MiniLM-L6-v2 runs locally (no API key, no per-call cost), is fast
    # on CPU, and produces 384-dim vectors that are strong enough for
    # sentence/paragraph-level semantic search.
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # --- Retrieval ---
    top_k: int = _get_int("TOP_K", 4)
    # Chunks scoring below this cosine similarity are dropped even if
    # they're in the top-k, so the model isn't fed irrelevant context.
    similarity_threshold: float = _get_float("SIMILARITY_THRESHOLD", 0.2)

    # --- Generation (Anthropic) ---
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    # Haiku is cheap and fast, which suits a RAG setup where the model's
    # job is mostly to synthesize retrieved text rather than reason deeply.
    llm_model: str = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
    max_tokens: int = _get_int("MAX_TOKENS", 1024)
    temperature: float = _get_float("TEMPERATURE", 0.2)

    # --- Storage ---
    index_dir: str = os.getenv("INDEX_DIR", "storage")


settings = Settings()
