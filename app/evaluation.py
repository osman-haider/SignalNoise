"""
Standard IR evaluation metrics: Precision@k, Recall@k, NDCG@k.

Deliberately implemented from scratch and kept small -- these are the exact
metrics named in the job posting this demo is built around, so the point is
to show working knowledge of them, not to hide them behind a library.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class QueryResult:
    """One query's ranked list of doc ids, in ranked (best-first) order."""

    query_id: str
    ranked_doc_ids: list[str]


@dataclass
class QueryJudgment:
    """Ground truth for one query: which doc ids are relevant, and at what
    graded relevance level (0 = not relevant, higher = more relevant)."""

    query_id: str
    relevant_doc_ids: set[str]
    grade: int = 2  # binary-graded relevance is enough for this demo


def precision_at_k(ranked_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    top_k = ranked_doc_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant_doc_ids)
    return hits / len(top_k)


def recall_at_k(ranked_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    if not relevant_doc_ids:
        return 0.0
    top_k = ranked_doc_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_doc_ids)
    return hits / len(relevant_doc_ids)


def _dcg_at_k(relevances: list[int], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        if rel:
            dcg += rel / math.log2(i + 1)
    return dcg


def ndcg_at_k(ranked_doc_ids: list[str], relevant_doc_ids: set[str], k: int, grade: int = 2) -> float:
    relevances = [grade if doc_id in relevant_doc_ids else 0 for doc_id in ranked_doc_ids[:k]]
    dcg = _dcg_at_k(relevances, k)

    ideal_relevances = sorted(relevances, reverse=True)
    # Ideal ranking would put every relevant doc first, up to k or the total
    # number of relevant docs, whichever is smaller.
    n_relevant_in_top_k = min(len(relevant_doc_ids), k)
    ideal_relevances = [grade] * n_relevant_in_top_k + [0] * (k - n_relevant_in_top_k)
    idcg = _dcg_at_k(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0.0


@dataclass
class AggregateMetrics:
    k: int
    mean_precision: float
    mean_recall: float
    mean_ndcg: float
    per_query: list[dict] = field(default_factory=list)


def evaluate(results: list[QueryResult], judgments: dict[str, QueryJudgment], k: int = 10) -> AggregateMetrics:
    """Average Precision@k / Recall@k / NDCG@k across all queries in `results`."""
    per_query = []
    for result in results:
        judgment = judgments.get(result.query_id)
        if judgment is None:
            continue
        p = precision_at_k(result.ranked_doc_ids, judgment.relevant_doc_ids, k)
        r = recall_at_k(result.ranked_doc_ids, judgment.relevant_doc_ids, k)
        n = ndcg_at_k(result.ranked_doc_ids, judgment.relevant_doc_ids, k, judgment.grade)
        per_query.append(
            {
                "query_id": result.query_id,
                "precision_at_k": round(p, 4),
                "recall_at_k": round(r, 4),
                "ndcg_at_k": round(n, 4),
            }
        )

    if not per_query:
        return AggregateMetrics(k=k, mean_precision=0.0, mean_recall=0.0, mean_ndcg=0.0, per_query=[])

    mean_p = sum(row["precision_at_k"] for row in per_query) / len(per_query)
    mean_r = sum(row["recall_at_k"] for row in per_query) / len(per_query)
    mean_n = sum(row["ndcg_at_k"] for row in per_query) / len(per_query)

    return AggregateMetrics(
        k=k,
        mean_precision=round(mean_p, 4),
        mean_recall=round(mean_r, 4),
        mean_ndcg=round(mean_n, 4),
        per_query=per_query,
    )
