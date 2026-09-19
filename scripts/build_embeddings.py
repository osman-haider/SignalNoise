"""
Precomputes and caches embeddings for the (fixed, small) demo corpus.

Run this once, offline, before starting the server:

    python scripts/build_embeddings.py

Corpus embeddings are cached to cache/corpus_embeddings.npy, keyed by a hash
of the corpus text (see app/embeddings.py). If you edit data/corpus.json,
just re-run this script -- the cache will be rebuilt automatically because
the hash will no longer match.

Doing this offline (rather than embedding all ~100 docs on every app
startup) keeps the live demo path fast and means the only network-dependent
call during a live search is the single query embedding.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import data  # noqa: E402
from app.embeddings import EmbeddingUnavailable, build_and_cache_corpus_embeddings  # noqa: E402


def main() -> None:
    texts = data.corpus_texts()
    print(f"Embedding {len(texts)} corpus documents...")
    try:
        vectors = build_and_cache_corpus_embeddings(texts)
    except EmbeddingUnavailable as exc:
        print(f"ERROR: could not build corpus embeddings: {exc}")
        print("Check OFFICIAL_OPENAI_API_KEY / OFFICIAL_OPENAI_BASE_URL / OFFICIAL_EMBEDDING_MODEL_NAME in your .env.")
        raise SystemExit(1)

    print(f"Done. Cached {vectors.shape[0]} vectors of dimension {vectors.shape[1]} "
          f"to cache/corpus_embeddings.npy")


if __name__ == "__main__":
    main()
