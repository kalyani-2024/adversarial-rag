"""Per-request trace models (OpenTelemetry-shaped: named spans with attributes)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SpanOut(BaseModel):
    name: str
    status: Literal["ok", "error", "skipped"] = "ok"
    start_offset_ms: float = Field(0.0, description="Start time relative to the request start")
    duration_ms: float = 0.0
    attributes: dict[str, Any] = Field(default_factory=dict)


class TokenUsage(BaseModel):
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class TraceOut(BaseModel):
    request_id: str
    spans: list[SpanOut] = Field(default_factory=list)
    total_ms: float = 0.0
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    generation_ms: float = 0.0
    evaluation_ms: float = 0.0
    retrieved_chunks: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_retries: int = 0
    throttle_ms: float = Field(0.0, description="Time spent waiting on provider rate limits / transient errors")
    estimated_cost_usd: float | None = Field(
        None, description="Only set when PRICE_*_PER_1M are configured; an estimate, not a bill"
    )
