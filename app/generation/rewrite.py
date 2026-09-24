"""Conditional query rewriting.

A cheap deterministic gate decides *whether* to rewrite; only then do we pay
for an LLM call. Rewriting a clear query is not free: it adds latency and can
drift away from the user's wording, which hurts BM25.
"""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel

from app.core.errors import LLMError
from app.core.llm import LLMClient, LLMResponse, parse_json_model
from app.generation.prompts import REWRITE_SYSTEM
from app.schemas.query import ChatTurn

logger = logging.getLogger(__name__)

_REFERENCE_WORDS = re.compile(
    r"\b(it|its|this|that|these|those|they|them|their|he|she|his|her|above|previous|former|latter|same|"
    r"the (?:first|second|third|last) one)\b",
    re.IGNORECASE,
)
_FILLER = re.compile(
    r"^\s*(hi|hey|hello|ok(ay)?|so|um+|please|can you|could you|would you|tell me|i want to know|"
    r"i was wondering|i'd like to know|do you know)\b",
    re.IGNORECASE,
)


class RewriteDecision(BaseModel):
    should_rewrite: bool
    reason: str


class _RewriteOut(BaseModel):
    query: str


def decide_rewrite(query: str, history: list[ChatTurn]) -> RewriteDecision:
    if history and _REFERENCE_WORDS.search(query):
        return RewriteDecision(should_rewrite=True, reason="follow-up with unresolved reference")
    if _FILLER.search(query):
        return RewriteDecision(should_rewrite=True, reason="conversational phrasing")
    return RewriteDecision(should_rewrite=False, reason="query is already standalone")


def rewrite_query(llm: LLMClient, query: str, history: list[ChatTurn], *, model: str) -> tuple[str, LLMResponse | None]:
    """Return `(retrieval_query, llm_response)`. Falls back to the original query on any failure."""
    convo = "\n".join(f"{t.role}: {t.content}" for t in history[-6:])
    user = (f"Conversation so far:\n{convo}\n\n" if convo else "") + f"Latest user message: {query}"
    try:
        resp = llm.complete(
            [{"role": "system", "content": REWRITE_SYSTEM}, {"role": "user", "content": user}],
            purpose="rewrite",
            model=model,
            temperature=0.0,
            max_tokens=200,
            json_mode=True,
        )
        rewritten = parse_json_model(resp.text, _RewriteOut).query.strip()
    except LLMError as exc:
        logger.warning("query rewrite failed; using original query", extra={"error": exc.message})
        return query, None
    # Guardrails: an empty or runaway rewrite is worse than none.
    if not rewritten or len(rewritten) > max(300, 4 * len(query)):
        return query, resp
    return rewritten, resp
