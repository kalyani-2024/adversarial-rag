"""Structured outputs of the reliability judge and adversarial critic."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Verdict = Literal["PASS", "FAIL"]


class JudgeScores(BaseModel):
    """What the judge LLM is asked to return (validated on parse)."""

    faithfulness: float = Field(ge=0, le=1, description="Share of claims supported by the context")
    relevance: float = Field(ge=0, le=1, description="How directly the answer addresses the question")
    completeness: float = Field(ge=0, le=1, description="How fully the context-supported answer is given")
    unsupported_claims: list[str] = Field(default_factory=list)
    reason: str = ""


class JudgeResult(JudgeScores):
    """Judge scores plus a verdict computed *in code* from configured thresholds."""

    verdict: Verdict
    failed_checks: list[str] = Field(default_factory=list)

    @property
    def grounded(self) -> bool:
        """No faithfulness-type failure (faithfulness is a hard constraint; relevance/completeness are soft)."""
        return not {"faithfulness", "unsupported_claims"} & set(self.failed_checks)

    @property
    def aggregate(self) -> float:
        """Single number used to pick the best attempt (faithfulness-weighted)."""
        return 0.5 * self.faithfulness + 0.25 * self.relevance + 0.25 * self.completeness


class Critique(BaseModel):
    unsupported_claims: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    weak_reasoning: list[str] = Field(default_factory=list)
    irrelevant_content: list[str] = Field(default_factory=list)
    instructions: str = Field("", description="Concrete guidance for the regeneration step")


class Attempt(BaseModel):
    """One generate/judge round. Attempt 0 is the initial answer."""

    attempt: int
    answer: str
    judge: JudgeResult | None = None
    critique: Critique | None = Field(None, description="Critique produced *after* judging this attempt")


class Thresholds(BaseModel):
    faithfulness: float = Field(ge=0, le=1)
    relevance: float = Field(ge=0, le=1)
    completeness: float = Field(ge=0, le=1)
    # LLM judges often list unsupported claims yet still score faithfulness ~0.95
    # (observed with qwen3.8-27b). Treat any listed claim as a failure.
    fail_on_unsupported_claims: bool = True
