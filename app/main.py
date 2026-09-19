"""
Signal/Noise -- FastAPI backend.

Endpoints:
  GET  /api/corpus         -> list every document in the demo corpus
  GET  /api/eval-queries   -> the curated evaluation query set + ground truth
  POST /api/search         -> run one query in "naive" or "hybrid" mode
  POST /api/disambiguate   -> just the disambiguation step, standalone
  POST /api/evaluate       -> run the full eval set through both modes and
                              compare Precision@k / Recall@k / NDCG@k

Static frontend is served from /static and mounted at the site root.
"""
from __future__ import annotations

import dataclasses

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import data
from .config import settings
from .evaluation import AggregateMetrics, QueryJudgment, QueryResult, evaluate
from .llm_disambiguate import DisambiguationUnavailable, disambiguate as disambiguate_query
from .search_engine import search as run_search

app = FastAPI(title="Signal/Noise", version="0.1.0")

# Permissive CORS: this is a small local/demo project, not a multi-tenant
# production service, so a wide-open policy keeps local frontend/backend
# development friction-free without adding real risk.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Request/response schemas
# --------------------------------------------------------------------------


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=200)
    mode: str = Field(..., pattern="^(naive|hybrid)$")
    k: int = Field(default=10, ge=1, le=50)


class DisambiguateRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=200)


class EvaluateRequest(BaseModel):
    k: int = Field(default=10, ge=1, le=50)


def _asdict(obj):
    return dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else obj


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@app.get("/api/corpus")
def get_corpus():
    docs = data.load_corpus()
    return {"count": len(docs), "documents": [dataclasses.asdict(d) for d in docs]}


@app.get("/api/eval-queries")
def get_eval_queries():
    queries = data.load_eval_queries()
    return {"count": len(queries), "queries": [dataclasses.asdict(q) for q in queries]}


@app.post("/api/search")
def post_search(req: SearchRequest):
    try:
        response = run_search(req.query, mode=req.mode, k=req.k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _asdict(response)


@app.post("/api/disambiguate")
def post_disambiguate(req: DisambiguateRequest):
    try:
        result = disambiguate_query(req.query)
        return {"query": req.query, "degraded": False, **result.to_dict()}
    except DisambiguationUnavailable as exc:
        return {
            "query": req.query,
            "degraded": True,
            "degraded_reason": str(exc),
            "senses": [],
            "selected_sense": req.query,
            "expansion_terms": [],
            "confidence": 0.0,
        }


@app.post("/api/evaluate")
def post_evaluate(req: EvaluateRequest):
    """Runs every query in the curated eval set through both naive and
    hybrid mode, and returns the aggregate Precision/Recall/NDCG@k for each
    side by side, plus a per-query breakdown."""
    eval_queries = data.load_eval_queries()
    judgments = {
        q.id: QueryJudgment(query_id=q.id, relevant_doc_ids=set(q.relevant_doc_ids), grade=q.grade)
        for q in eval_queries
    }

    def run_mode(mode: str) -> AggregateMetrics:
        results = []
        degraded_any = False
        for q in eval_queries:
            resp = run_search(q.query, mode=mode, k=req.k)
            degraded_any = degraded_any or resp.degraded
            results.append(QueryResult(query_id=q.id, ranked_doc_ids=[r.doc_id for r in resp.results]))
        metrics = evaluate(results, judgments, k=req.k)
        metrics_dict = dataclasses.asdict(metrics)
        metrics_dict["degraded"] = degraded_any
        return metrics_dict

    naive_metrics = run_mode("naive")
    hybrid_metrics = run_mode("hybrid")

    # Attach the query text + kind to each per-query row for a readable UI table.
    query_lookup = {q.id: q for q in eval_queries}
    for metrics in (naive_metrics, hybrid_metrics):
        for row in metrics["per_query"]:
            q = query_lookup[row["query_id"]]
            row["query_text"] = q.query
            row["kind"] = q.kind

    return {"k": req.k, "naive": naive_metrics, "hybrid": hybrid_metrics}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "openai_configured": bool(settings.openai_api_key),
        "embedding_model": settings.embedding_model_name,
        "chat_model": settings.openai_model_name,
    }


# --------------------------------------------------------------------------
# Static frontend
# --------------------------------------------------------------------------

app.mount("/", StaticFiles(directory=str(settings.static_dir), html=True), name="static")
