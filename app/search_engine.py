"""
The actual "naive vs hybrid" search logic the whole demo is about.

- mode="naive"  -> plain BM25 over the raw query. This is the baseline that
  reproduces the documented "ambiguous keyword returns noise" problem.
- mode="hybrid" -> LLM disambiguates the query -> BM25 runs against the
  query *plus* the model's expansion terms -> the query is embedded and
  compared against precomputed corpus embeddings -> the two scores are
  combined into a single ranking.

Degradation is explicit, not silent: if disambiguation or embeddings fail
for any reason, the response says so (`degraded=True`, `degraded_reason=...`)
and still returns a usable ranking rather than a 500 error.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from . import data
from .bm25 import BM25, tokenize
from .config import settings
from .embeddings import EmbeddingUnavailable, cosine_similarity_matrix, embed_query, load_cached_corpus_embeddings
from .llm_disambiguate import DisambiguationResult, DisambiguationUnavailable, disambiguate


@dataclass
class ScoredDoc:
    doc_id: str
    text: str
    term: str
    topic: str
    score: float
    bm25_score: float
    semantic_score: float | None = None


@dataclass
class SearchResponse:
    mode: str
    query: str
    results: list[ScoredDoc]
    disambiguation: dict | None = None
    degraded: bool = False
    degraded_reason: str | None = None


def _min_max_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


@lru_cache(maxsize=1)
def _bm25_index() -> tuple[BM25, list[str]]:
    docs = data.load_corpus()
    tokenized = [tokenize(doc.text) for doc in docs]
    return BM25(tokenized), [doc.id for doc in docs]


def _corpus_embeddings() -> np.ndarray | None:
    """Returns cached corpus embeddings if they exist, else None (caller
    should treat that as "semantic scoring unavailable")."""
    texts = data.corpus_texts()
    return load_cached_corpus_embeddings(texts)


def _bm25_scores_for_query(query_text: str) -> dict[str, float]:
    bm25, doc_ids = _bm25_index()
    scores = bm25.get_scores(tokenize(query_text))
    return dict(zip(doc_ids, scores))


def _build_scored_docs(doc_ids_in_order: list[str], scores: dict[str, float], semantic: dict[str, float] | None = None) -> list[ScoredDoc]:
    docs_by_id = data.docs_by_id()
    scored = []
    for doc_id in doc_ids_in_order:
        doc = docs_by_id[doc_id]
        scored.append(
            ScoredDoc(
                doc_id=doc_id,
                text=doc.text,
                term=doc.term,
                topic=doc.topic,
                score=scores[doc_id],
                bm25_score=scores[doc_id],
                semantic_score=(semantic.get(doc_id) if semantic else None),
            )
        )
    return scored


def search_naive(query: str, k: int = 10) -> SearchResponse:
    bm25_scores = _bm25_scores_for_query(query)
    ranked_ids = sorted(bm25_scores, key=lambda doc_id: bm25_scores[doc_id], reverse=True)[:k]
    results = _build_scored_docs(ranked_ids, bm25_scores)
    return SearchResponse(mode="naive", query=query, results=results)


def search_hybrid(query: str, k: int = 10) -> SearchResponse:
    disambiguation_dict: dict | None = None
    degraded = False
    degraded_reason = None

    # Step 1: disambiguate the query (LLM call).
    augmented_query = query
    disambiguation_result: DisambiguationResult | None = None
    try:
        disambiguation_result = disambiguate(query)
        disambiguation_dict = disambiguation_result.to_dict()
        if disambiguation_result.expansion_terms:
            augmented_query = query + " " + " ".join(disambiguation_result.expansion_terms)
    except DisambiguationUnavailable as exc:
        degraded = True
        degraded_reason = f"Query disambiguation unavailable ({exc}); showing lexical-only results."

    # Step 2: BM25 against the (possibly expanded) query -- this alone
    # already benefits from disambiguation even if embeddings fail later.
    bm25_scores = _bm25_scores_for_query(augmented_query)

    # Step 3: semantic re-ranking, if corpus embeddings exist AND the
    # live query embedding call succeeds. Either failure degrades to
    # BM25-only (still using the expanded query from step 1/2).
    corpus_vectors = _corpus_embeddings()
    doc_ids = list(bm25_scores.keys())
    semantic_scores: dict[str, float] | None = None

    if corpus_vectors is None:
        degraded = True
        degraded_reason = degraded_reason or (
            "Corpus embeddings not built yet (run scripts/build_embeddings.py); "
            "showing lexical-only results."
        )
    else:
        try:
            query_vector = embed_query(augmented_query)
            sims = cosine_similarity_matrix(query_vector, corpus_vectors)
            semantic_scores = dict(zip(doc_ids, sims.tolist()))
        except EmbeddingUnavailable as exc:
            degraded = True
            degraded_reason = degraded_reason or (
                f"Semantic re-ranking unavailable ({exc}); showing lexical-only results."
            )

    # Step 4: combine scores (or just use BM25 if semantic scoring degraded).
    if semantic_scores is not None:
        bm25_norm = dict(zip(doc_ids, _min_max_normalize([bm25_scores[d] for d in doc_ids])))
        sem_norm = dict(zip(doc_ids, _min_max_normalize([semantic_scores[d] for d in doc_ids])))
        combined = {
            d: settings.hybrid_bm25_weight * bm25_norm[d] + settings.hybrid_semantic_weight * sem_norm[d]
            for d in doc_ids
        }
    else:
        combined = bm25_scores

    ranked_ids = sorted(combined, key=lambda doc_id: combined[doc_id], reverse=True)[:k]
    results = _build_scored_docs(ranked_ids, combined, semantic_scores)

    return SearchResponse(
        mode="hybrid",
        query=query,
        results=results,
        disambiguation=disambiguation_dict,
        degraded=degraded,
        degraded_reason=degraded_reason,
    )


def search(query: str, mode: str, k: int = 10) -> SearchResponse:
    if mode == "naive":
        return search_naive(query, k=k)
    if mode == "hybrid":
        return search_hybrid(query, k=k)
    raise ValueError(f"Unknown search mode: {mode!r} (expected 'naive' or 'hybrid')")
