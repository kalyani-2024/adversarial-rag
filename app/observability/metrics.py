"""In-process service metrics exposed at GET /metrics.

Counters plus a bounded window of recent latencies for percentiles. This is
per-process and resets on restart; in production you would export the same
numbers to Prometheus (counters/histograms) instead of keeping them here.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque

from pydantic import BaseModel

from app.schemas.query import QueryResponse


class LatencyStats(BaseModel):
    count: int
    p50_ms: float | None
    p95_ms: float | None
    mean_ms: float | None


class MetricsSnapshot(BaseModel):
    uptime_s: float
    queries_total: int
    queries_by_status: dict[str, int]
    queries_by_mode: dict[str, int]
    errors_by_type: dict[str, int]
    verdicts_final: dict[str, int]
    retries_total: int
    queries_with_retry: int
    retry_rate: float | None
    improved_by_loop: int
    llm_calls_total: int
    prompt_tokens_total: int
    completion_tokens_total: int
    estimated_cost_usd_total: float | None
    latency_total: LatencyStats
    latency_by_stage: dict[str, LatencyStats]
    documents_ingested: int
    documents_deleted: int


def _stats(values: list[float]) -> LatencyStats:
    if not values:
        return LatencyStats(count=0, p50_ms=None, p95_ms=None, mean_ms=None)
    ordered = sorted(values)

    def pct(p: float) -> float:
        return round(ordered[min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))], 1)

    return LatencyStats(count=len(values), p50_ms=pct(0.5), p95_ms=pct(0.95), mean_ms=round(sum(values) / len(values), 1))


class MetricsRegistry:
    def __init__(self, window: int = 1000) -> None:
        self._lock = threading.Lock()
        self._start = time.time()
        self._status: Counter[str] = Counter()
        self._mode: Counter[str] = Counter()
        self._errors: Counter[str] = Counter()
        self._verdicts: Counter[str] = Counter()
        self._retries = 0
        self._with_retry = 0
        self._adversarial_answered = 0
        self._improved = 0
        self._llm_calls = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._cost: float | None = None
        self._docs_in = 0
        self._docs_out = 0
        self._total: deque[float] = deque(maxlen=window)
        self._stages: dict[str, deque[float]] = {}
        self._window = window

    def record_query(self, r: QueryResponse) -> None:
        with self._lock:
            self._status[r.status] += 1
            self._mode[r.mode] += 1
            if r.reliability.enabled:
                self._adversarial_answered += 1
                self._retries += r.reliability.retries
                self._with_retry += int(r.reliability.retries > 0)
                self._improved += int(r.reliability.improved)
                if r.reliability.final:
                    self._verdicts[r.reliability.final.verdict] += 1
                elif r.reliability.judge_error:
                    self._verdicts["UNVERIFIED"] += 1
            t = r.trace
            self._llm_calls += t.llm_calls
            self._prompt_tokens += t.prompt_tokens
            self._completion_tokens += t.completion_tokens
            if t.estimated_cost_usd is not None:
                self._cost = (self._cost or 0.0) + t.estimated_cost_usd
            self._total.append(t.total_ms)
            for span in t.spans:
                if span.status != "skipped":
                    self._stages.setdefault(span.name, deque(maxlen=self._window)).append(span.duration_ms)

    def record_error(self, error_type: str) -> None:
        with self._lock:
            self._errors[error_type] += 1

    def record_document(self, *, ingested: int = 0, deleted: int = 0) -> None:
        with self._lock:
            self._docs_in += ingested
            self._docs_out += deleted

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            total_q = sum(self._status.values())
            return MetricsSnapshot(
                uptime_s=round(time.time() - self._start, 1),
                queries_total=total_q,
                queries_by_status=dict(self._status),
                queries_by_mode=dict(self._mode),
                errors_by_type=dict(self._errors),
                verdicts_final=dict(self._verdicts),
                retries_total=self._retries,
                queries_with_retry=self._with_retry,
                retry_rate=round(self._with_retry / self._adversarial_answered, 3) if self._adversarial_answered else None,
                improved_by_loop=self._improved,
                llm_calls_total=self._llm_calls,
                prompt_tokens_total=self._prompt_tokens,
                completion_tokens_total=self._completion_tokens,
                estimated_cost_usd_total=round(self._cost, 6) if self._cost is not None else None,
                latency_total=_stats(list(self._total)),
                latency_by_stage={k: _stats(list(v)) for k, v in sorted(self._stages.items())},
                documents_ingested=self._docs_in,
                documents_deleted=self._docs_out,
            )
