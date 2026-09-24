"""Adversarial critic: runs only after a FAIL verdict.

Produces a structured list of concrete problems that the regeneration step
must fix. If the critic call itself fails, we degrade to a critique derived
from the judge's output instead of aborting the request.
"""

from __future__ import annotations

import logging

from app.core.errors import LLMError
from app.core.llm import LLMClient, LLMResponse, parse_json_model
from app.generation.generator import format_context
from app.generation.prompts import CRITIC_SYSTEM, CRITIC_USER
from app.schemas.reliability import Critique, JudgeResult
from app.schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


def critique_from_judge(judge: JudgeResult) -> Critique:
    """Fallback critique built from the judge's findings (no extra LLM call)."""
    missing = ["The answer omits information from the sources needed to fully answer the question."] if "completeness" in judge.failed_checks else []
    irrelevant = ["Parts of the answer do not address the question."] if "relevance" in judge.failed_checks else []
    return Critique(
        unsupported_claims=list(judge.unsupported_claims),
        missing_evidence=missing,
        irrelevant_content=irrelevant,
        instructions=judge.reason,
    )


def critique_answer(
    llm: LLMClient,
    question: str,
    chunks: list[RetrievedChunk],
    answer: str,
    judge: JudgeResult,
    *,
    model: str,
) -> tuple[Critique, LLMResponse | None, str | None]:
    """Return `(critique, llm_response, error)`; never raises for LLM failures."""
    try:
        resp = llm.complete(
            [
                {"role": "system", "content": CRITIC_SYSTEM},
                {
                    "role": "user",
                    "content": CRITIC_USER.format(
                        context=format_context(chunks), question=question, answer=answer, judge_reason=judge.reason
                    ),
                },
            ],
            purpose="critic",
            model=model,
            temperature=0.2,
            max_tokens=900,
            json_mode=True,
        )
        return parse_json_model(resp.text, Critique), resp, None
    except LLMError as exc:
        logger.warning("critic failed; using judge-derived critique", extra={"error": exc.message})
        return critique_from_judge(judge), None, exc.message
