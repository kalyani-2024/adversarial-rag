"""Lightweight per-request tracing.

A `Trace` is a list of named, timed spans plus LLM token accounting. The shape
deliberately mirrors OpenTelemetry (trace id, span name, start offset,
duration, status, attributes), so exporting to an OTel collector later is an
adapter, not a redesign. We avoid the OTel SDK today because there is no
collector to send to and it adds dependencies without adding capability.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

from app.core.llm import LLMResponse
from app.schemas.trace import SpanOut, TokenUsage, TraceOut

logger = logging.getLogger("rag.trace")

# Spans whose durations roll up into the latency breakdown.
_GENERATION_SPANS = {"generation", "regeneration"}
_EVALUATION_SPANS = {"judge", "critic"}


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


class Span:
    def __init__(self, name: str, start_offset_ms: float) -> None:
        self.name = name
        self.start_offset_ms = start_offset_ms
        self.duration_ms = 0.0
        self.status = "ok"
        self.attributes: dict[str, Any] = {}

    def set(self, **attributes: Any) -> None:
        self.attributes.update(attributes)

    def to_out(self) -> SpanOut:
        return SpanOut(
            name=self.name,
            status=self.status,
            start_offset_ms=round(self.start_offset_ms, 1),
            duration_ms=round(self.duration_ms, 1),
            attributes=self.attributes,
        )


class Trace:
    def __init__(self, request_id: str | None = None) -> None:
        self.request_id = request_id or new_request_id()
        self._t0 = time.perf_counter()
        self.spans: list[Span] = []
        self.usage = TokenUsage()
        self.llm_calls_by_purpose: dict[str, int] = {}
        self.llm_retries = 0
        self.throttle_ms = 0.0

    def _offset_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        span = Span(name, self._offset_ms())
        span.set(**attributes)
        start = time.perf_counter()
        try:
            yield span
        except Exception as exc:
            span.status = "error"
            span.set(error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            span.duration_ms = (time.perf_counter() - start) * 1000
            self.spans.append(span)
            logger.info(
                "span",
                extra={"request_id": self.request_id, "span": name, "status": span.status,
                       "duration_ms": round(span.duration_ms, 1), **_loggable(span.attributes)},
            )

    def add_span(self, name: str, duration_ms: float, *, status: str = "ok", **attributes: Any) -> None:
        """Record a span timed elsewhere (e.g. sub-stages measured inside a component)."""
        span = Span(name, max(self._offset_ms() - duration_ms, 0.0))
        span.duration_ms, span.status = duration_ms, status
        span.set(**attributes)
        self.spans.append(span)

    def record_llm(self, resp: LLMResponse | None, purpose: str) -> None:
        if resp is None:
            return
        self.usage.llm_calls += 1
        self.usage.prompt_tokens += resp.prompt_tokens
        self.usage.completion_tokens += resp.completion_tokens
        self.llm_calls_by_purpose[purpose] = self.llm_calls_by_purpose.get(purpose, 0) + 1
        self.llm_retries += resp.attempts - 1
        self.throttle_ms += resp.throttle_ms

    def total_ms(self) -> float:
        return self._offset_ms()

    def to_out(
        self,
        *,
        retrieved_chunks: int = 0,
        price_prompt_per_1m: float | None = None,
        price_completion_per_1m: float | None = None,
    ) -> TraceOut:
        def total(names: set[str]) -> float:
            return round(sum(s.duration_ms for s in self.spans if s.name in names and s.status != "skipped"), 1)

        cost = None
        if price_prompt_per_1m is not None and price_completion_per_1m is not None:
            cost = round(
                self.usage.prompt_tokens / 1e6 * price_prompt_per_1m
                + self.usage.completion_tokens / 1e6 * price_completion_per_1m,
                6,
            )
        return TraceOut(
            request_id=self.request_id,
            spans=[s.to_out() for s in self.spans],
            total_ms=round(self.total_ms(), 1),
            retrieval_ms=total({"hybrid_retrieval"}),
            rerank_ms=total({"reranking"}),
            generation_ms=total(_GENERATION_SPANS),
            evaluation_ms=total(_EVALUATION_SPANS),
            retrieved_chunks=retrieved_chunks,
            llm_calls=self.usage.llm_calls,
            prompt_tokens=self.usage.prompt_tokens,
            completion_tokens=self.usage.completion_tokens,
            llm_retries=self.llm_retries,
            throttle_ms=round(self.throttle_ms, 1),
            estimated_cost_usd=cost,
        )


def _loggable(attrs: dict[str, Any]) -> dict[str, Any]:
    """Only scalar attributes go to logs (no chunk text / answers => no PII in logs)."""
    return {f"attr_{k}": v for k, v in attrs.items() if isinstance(v, (int, float, bool, str)) and len(str(v)) < 200}
