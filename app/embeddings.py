"""
OpenAI-backed embeddings, with disk caching for the (fixed) corpus and a
graceful-degradation path for the one call that has to happen live: turning
the user's query into a vector.

Design choices, and why:
- The corpus is small and fixed, so its embeddings are computed once
  (see scripts/build_embeddings.py) and cached to cache/corpus_embeddings.npy.
  The live demo path never has to wait on N embedding calls -- only one.
- If the query-time embedding call fails (bad key, network issue, timeout),
  callers get an EmbeddingUnavailable exception. main.py catches this and
  degrades the whole request to lexical-only search rather than erroring out,
  and tells the user that's what happened (see main.py / index.html).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from openai import OpenAI

from .config import settings


class EmbeddingUnavailable(RuntimeError):
    """Raised when we could not get an embedding back from the API."""


def _client() -> OpenAI:
    if not settings.openai_api_key:
        raise EmbeddingUnavailable("OPENAI_API_KEY is not set")
    kwargs = {"api_key": settings.openai_api_key}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return OpenAI(**kwargs)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts. Raises EmbeddingUnavailable on any failure so
    callers can decide how to degrade -- this module never silently returns
    zeros or fake vectors."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    try:
        client = _client()
        response = client.embeddings.create(
            model=settings.embedding_model_name,
            input=texts,
            timeout=settings.request_timeout_seconds,
        )
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any failure degrades the same way
        raise EmbeddingUnavailable(str(exc)) from exc

    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """Embed a single query string. This is the one live embedding call in
    the request path -- corpus embeddings are precomputed (see below)."""
    vectors = embed_texts([text])
    return vectors[0]


def cosine_similarity_matrix(query_vector: np.ndarray, corpus_vectors: np.ndarray) -> np.ndarray:
    """Cosine similarity between one query vector and every row of
    corpus_vectors. Returns a 1D array, one score per corpus document."""
    if corpus_vectors.size == 0:
        return np.zeros((0,), dtype=np.float32)

    query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
    corpus_norms = corpus_vectors / (np.linalg.norm(corpus_vectors, axis=1, keepdims=True) + 1e-8)
    return corpus_norms @ query_norm


# --------------------------------------------------------------------------
# Corpus embedding cache
# --------------------------------------------------------------------------


def _corpus_hash(doc_texts: list[str]) -> str:
    joined = "\n".join(doc_texts).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def _cache_paths() -> tuple[Path, Path]:
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    return (
        settings.cache_dir / "corpus_embeddings.npy",
        settings.cache_dir / "corpus_embeddings.meta.json",
    )


def load_cached_corpus_embeddings(doc_texts: list[str]) -> np.ndarray | None:
    """Returns cached embeddings if a cache exists AND matches the current
    corpus content exactly. Returns None otherwise (caller should compute
    fresh embeddings, e.g. via scripts/build_embeddings.py)."""
    vectors_path, meta_path = _cache_paths()
    if not vectors_path.exists() or not meta_path.exists():
        return None

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("hash") != _corpus_hash(doc_texts):
        return None  # corpus changed since the cache was built

    return np.load(vectors_path)


def save_corpus_embeddings(doc_texts: list[str], vectors: np.ndarray) -> None:
    vectors_path, meta_path = _cache_paths()
    np.save(vectors_path, vectors)
    meta_path.write_text(
        json.dumps({"hash": _corpus_hash(doc_texts), "count": len(doc_texts)}),
        encoding="utf-8",
    )


def build_and_cache_corpus_embeddings(doc_texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Computes embeddings for every document, in batches, and caches the
    result. Intended to be run offline (scripts/build_embeddings.py), not on
    every app startup."""
    all_vectors: list[np.ndarray] = []
    for start in range(0, len(doc_texts), batch_size):
        batch = doc_texts[start : start + batch_size]
        all_vectors.append(embed_texts(batch))
    vectors = np.vstack(all_vectors) if all_vectors else np.zeros((0, 0), dtype=np.float32)
    save_corpus_embeddings(doc_texts, vectors)
    return vectors
