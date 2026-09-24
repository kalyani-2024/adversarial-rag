"""Reliability judge: LLM scores, verdict decided in code.

The LLM only produces *measurements* (faithfulness / relevance / completeness
in [0, 1]). The PASS/FAIL decision is a deterministic threshold comparison, so
it is unit-testable, tunable per request, and cannot be talked into a PASS by
a persuasive answer.
"""

from __future__ import annotations

from app.core.llm import LLMClient, LLMResponse, parse_json_model
from app.generation.generator import format_context
from app.generation.prompts import JUDGE_SYSTEM, JUDGE_USER
from app.schemas.reliability import JudgeResult, JudgeScores, Thresholds
from app.schemas.retrieval import RetrievedChunk


def apply_thresholds(scores: JudgeScores, thresholds: Thresholds) -> JudgeResult:
    failed = [
        name
        for name in ("faithfulness", "relevance", "completeness")
        if getattr(scores, name) < getattr(thresholds, name)
    ]
    return JudgeResult(**scores.model_dump(), verdict="FAIL" if failed else "PASS", failed_checks=failed)


def judge_answer(
    llm: LLMClient,
    question: str,
    chunks: list[RetrievedChunk],
    answer: str,
    thresholds: Thresholds,
    *,
    model: str,
) -> tuple[JudgeResult, LLMResponse]:
    """Raises `LLMError` (incl. `LLMOutputError`) if no valid judgement can be obtained."""
    resp = llm.complete(
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_USER.format(context=format_context(chunks), question=question, answer=answer)},
        ],
        purpose="judge",
        model=model,
        temperature=0.0,
        max_tokens=600,
        json_mode=True,
    )
    scores = parse_json_model(resp.text, JudgeScores)
    return apply_thresholds(scores, thresholds), resp
