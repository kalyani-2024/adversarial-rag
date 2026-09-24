"""Structured backend log records for a completed query.

The chat UI shows only the answer and its citations. Everything needed to
debug or audit an answer (retrieval scores, judge verdicts, critiques, timings,
tokens) is emitted here as JSON log lines keyed by `request_id`.

Free text (the query, unsupported-claim excerpts, critique items) is logged only
when LOG_CONTENT=true (the default, for local debugging). Disable it wherever
logs must not contain user content.
"""

from __future__ import annotations

import logging

from app.schemas.query import QueryResponse

logger = logging.getLogger("rag.query")

_MAX_TEXT = 200


def _clip(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _MAX_TEXT else text[: _MAX_TEXT - 1] + "…"


def log_query(r: QueryResponse, *, include_content: bool) -> None:
    rid = r.request_id
    base = {"request_id": rid}

    request = {**base, "mode": r.mode, "query_rewritten": r.query_rewritten}
    if include_content:
        request["query"] = _clip(r.query)
        if r.query_rewritten:
            request["retrieval_query"] = _clip(r.retrieval_query)
    logger.info("query received", extra=request)

    logger.info(
        "retrieval",
        extra={
            **base,
            "chunks": [
                {
                    "rank": c.final_rank, "chunk_id": c.chunk_id, "doc": c.document_name, "page": c.page,
                    "dense": _round(c.dense_score), "dense_rank": c.dense_rank,
                    "bm25": _round(c.bm25_score), "bm25_rank": c.bm25_rank,
                    "rrf": _round(c.rrf_score, 4), "rerank": _round(c.rerank_score),
                }
                for c in r.retrieved_chunks
            ],
        },
    )

    for attempt in r.reliability.attempts:
        record = {**base, "attempt": attempt.attempt}
        if attempt.judge:
            j = attempt.judge
            record |= {"verdict": j.verdict, "faithfulness": j.faithfulness, "relevance": j.relevance,
                       "completeness": j.completeness, "failed_checks": j.failed_checks,
                       "unsupported_claims_count": len(j.unsupported_claims)}
            if include_content:
                record["unsupported_claims"] = [_clip(c) for c in j.unsupported_claims]
                record["judge_reason"] = _clip(j.reason)
        if attempt.critique:
            crit = attempt.critique
            record["critique_issue_counts"] = {
                k: len(getattr(crit, k)) for k in
                ("unsupported_claims", "contradictions", "missing_evidence", "weak_reasoning", "irrelevant_content")
            }
            if include_content:
                record["critique_instructions"] = _clip(crit.instructions)
        logger.info("reliability attempt", extra=record)

    t, rel = r.trace, r.reliability
    logger.info(
        "query completed",
        extra={
            **base, "status": r.status, "mode": r.mode,
            "final_verdict": rel.final.verdict if rel.final else None,
            "initial_verdict": rel.initial.verdict if rel.initial else None,
            "retries": rel.retries, "improved": rel.improved, "judge_error": rel.judge_error,
            "citations": [c.label for c in r.citations],
            "total_ms": t.total_ms, "retrieval_ms": t.retrieval_ms, "rerank_ms": t.rerank_ms,
            "generation_ms": t.generation_ms, "evaluation_ms": t.evaluation_ms,
            "throttle_ms": t.throttle_ms, "llm_calls": t.llm_calls, "llm_retries": t.llm_retries,
            "prompt_tokens": t.prompt_tokens, "completion_tokens": t.completion_tokens,
            "estimated_cost_usd": t.estimated_cost_usd,
        },
    )


def _round(x: float | None, digits: int = 3) -> float | None:
    return None if x is None else round(x, digits)
