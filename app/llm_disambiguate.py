"""
Query disambiguation via a single OpenAI chat-completion call.

Given a query and a fixed "analyst context" string, the model is asked to:
  1. identify whether the query term has more than one common sense,
  2. pick the sense that best matches the analyst context,
  3. suggest a short list of expansion terms for that sense (used to bias
     the lexical/BM25 side of hybrid search, not just the semantic side).

The model is asked for strict JSON so the backend never has to guess-parse
prose. If the call fails for any reason (bad key, network, malformed
response), callers get a DisambiguationUnavailable exception -- main.py
catches this and falls back to lexical-only search, same as the embeddings
failure path.
"""
from __future__ import annotations

import json

from openai import OpenAI

from .config import settings

_SYSTEM_PROMPT = """You help disambiguate short search queries for an intelligence \
analysis search tool. The person searching is doing open-source monitoring; some \
words in their query may have both a defense/military-relevant sense and a common \
everyday or pop-culture sense.

Given the query and the analyst's stated context, respond with STRICT JSON only \
(no prose, no markdown fences) matching this exact shape:

{
  "senses": ["short label for sense 1", "short label for sense 2", ...],
  "selected_sense": "the single sense label that best matches the analyst context",
  "expansion_terms": ["term1", "term2", "term3"],
  "confidence": 0.0
}

Rules:
- List at most 4 senses. If the query is not actually ambiguous, return a single
  sense and expansion_terms that are simply close synonyms/related terms for it.
- expansion_terms should be 2-5 short words or phrases that would help a keyword
  search find documents about the selected sense specifically (not the other senses).
- confidence is your rough confidence (0.0-1.0) that selected_sense is correct given
  the analyst context.
"""


class DisambiguationUnavailable(RuntimeError):
    """Raised when we could not get a usable disambiguation result."""


class DisambiguationResult:
    def __init__(self, senses: list[str], selected_sense: str, expansion_terms: list[str], confidence: float):
        self.senses = senses
        self.selected_sense = selected_sense
        self.expansion_terms = expansion_terms
        self.confidence = confidence

    def to_dict(self) -> dict:
        return {
            "senses": self.senses,
            "selected_sense": self.selected_sense,
            "expansion_terms": self.expansion_terms,
            "confidence": self.confidence,
        }


def _client() -> OpenAI:
    if not settings.openai_api_key:
        raise DisambiguationUnavailable("OFFICIAL_OPENAI_API_KEY is not set")
    kwargs = {"api_key": settings.openai_api_key}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return OpenAI(**kwargs)


def _extract_json(raw_text: str) -> dict:
    """Best-effort JSON extraction: try a direct parse first, then fall back
    to pulling out the first {...} block in case the model wrapped its
    answer in prose or a markdown fence despite instructions."""
    raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise DisambiguationUnavailable(f"Could not find JSON in model response: {raw_text[:200]!r}")
    return json.loads(raw_text[start : end + 1])


def disambiguate(query: str) -> DisambiguationResult:
    client = _client()
    user_prompt = (
        f'Analyst context: "{settings.analyst_context}"\n'
        f'Query: "{query}"'
    )

    try:
        response = client.chat.completions.create(
            model=settings.openai_model_name,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            timeout=settings.request_timeout_seconds,
        )
        raw_text = response.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001 - any failure degrades the same way
        raise DisambiguationUnavailable(str(exc)) from exc

    try:
        parsed = _extract_json(raw_text)
        return DisambiguationResult(
            senses=list(parsed.get("senses", [])),
            selected_sense=str(parsed.get("selected_sense", query)),
            expansion_terms=list(parsed.get("expansion_terms", [])),
            confidence=float(parsed.get("confidence", 0.0)),
        )
    except (ValueError, TypeError, DisambiguationUnavailable) as exc:
        raise DisambiguationUnavailable(f"Could not parse model response: {exc}") from exc
