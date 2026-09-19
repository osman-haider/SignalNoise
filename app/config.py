"""
Central place that reads configuration from the environment (via .env).

Nothing else in the app should call os.environ directly -- import Settings
from here instead, so there's exactly one place that knows about env var
names and defaults.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env once, on first import, from the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    openai_base_url: str | None
    openai_model_name: str
    embedding_model_name: str
    analyst_context: str
    hybrid_bm25_weight: float
    hybrid_semantic_weight: float
    request_timeout_seconds: float
    data_dir: Path
    cache_dir: Path
    static_dir: Path


def load_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OFFICIAL_OPENAI_API_KEY"),
        openai_base_url=os.getenv("OFFICIAL_OPENAI_BASE_URL") or None,
        openai_model_name=os.getenv("OFFICIAL_OPENAI_MODEL_NAME", "gpt-4o-mini"),
        embedding_model_name=os.getenv("OFFICIAL_EMBEDDING_MODEL_NAME", "text-embedding-3-small"),
        # Fixed, short "who is searching and why" string fed to the disambiguation
        # prompt so the model has a consistent frame of reference. Kept simple and
        # constant on purpose -- see the demo spec for why.
        analyst_context=os.getenv(
            "ANALYST_CONTEXT",
            "This search is for an open-source military and foreign-affairs "
            "intelligence monitoring workflow, over a mixed corpus that also "
            "contains unrelated everyday and pop-culture content.",
        ),
        hybrid_bm25_weight=float(os.getenv("HYBRID_BM25_WEIGHT", "0.4")),
        hybrid_semantic_weight=float(os.getenv("HYBRID_SEMANTIC_WEIGHT", "0.6")),
        request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "8")),
        data_dir=_PROJECT_ROOT / "data",
        cache_dir=_PROJECT_ROOT / "cache",
        static_dir=_PROJECT_ROOT / "static",
    )


settings = load_settings()
