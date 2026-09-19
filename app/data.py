"""Loads the (already-generated) corpus and eval query set from data/*.json.

Run scripts/generate_corpus.py first if these files don't exist yet.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

from .config import settings


@dataclass(frozen=True)
class Document:
    id: str
    term: str
    topic: str
    is_military_relevant: bool
    text: str


@dataclass(frozen=True)
class EvalQuery:
    id: str
    query: str
    kind: str
    relevant_doc_ids: list[str]
    grade: int


def _corpus_path():
    return settings.data_dir / "corpus.json"


def _eval_queries_path():
    return settings.data_dir / "eval_queries.json"


@lru_cache(maxsize=1)
def load_corpus() -> list[Document]:
    path = _corpus_path()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/generate_corpus.py` first."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Document(**doc) for doc in raw]


@lru_cache(maxsize=1)
def load_eval_queries() -> list[EvalQuery]:
    path = _eval_queries_path()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/generate_corpus.py` first."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [EvalQuery(**q) for q in raw]


def corpus_texts() -> list[str]:
    return [doc.text for doc in load_corpus()]


def corpus_ids() -> list[str]:
    return [doc.id for doc in load_corpus()]


def docs_by_id() -> dict[str, Document]:
    return {doc.id: doc for doc in load_corpus()}
