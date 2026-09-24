"""Query orchestration: request -> RunConfig -> graph -> response (+ trace, metrics)."""

from __future__ import annotations

import logging

from app.core.config import Settings
from app.core.llm import LLMClient
from app.observability.metrics import MetricsRegistry
from app.observability.query_log import log_query
from app.services.priority import QueryActivity
from app.observability.tracing import Trace
from app.pipeline.graph import RunConfig, build_graph, recursion_limit
from app.retrieval.hybrid import HybridRetriever
from app.schemas.query import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    QueryRequest,
    QueryResponse,
    ReliabilityReport,
)
from app.schemas.reliability import Thresholds

logger = logging.getLogger(__name__)


class QueryService:
    def __init__(self, settings: Settings, llm: LLMClient, retriever: HybridRetriever, metrics: MetricsRegistry,
                 activity: QueryActivity | None = None) -> None:
        self.settings = settings
        self.llm = llm
        self.retriever = retriever
        self.metrics = metrics
        self.activity = activity or QueryActivity()
        self.graph = build_graph(llm, retriever)

    def resolve(self, request: QueryRequest) -> RunConfig:
        s, o = self.settings, request.options
        pick = lambda override, default: default if override is None else override  # noqa: E731
        return RunConfig(
            mode=o.mode,
            top_k=pick(o.top_k, s.final_top_k),
            max_retries=pick(o.max_retries, s.max_retries),
            thresholds=Thresholds(
                faithfulness=pick(o.faithfulness_threshold, s.faithfulness_threshold),
                relevance=pick(o.relevance_threshold, s.relevance_threshold),
                completeness=pick(o.completeness_threshold, s.completeness_threshold),
                fail_on_unsupported_claims=pick(o.fail_on_unsupported_claims, s.fail_on_unsupported_claims),
            ),
            use_reranker=pick(o.use_reranker, s.reranker_enabled),
            use_query_rewrite=pick(o.use_query_rewrite, s.query_rewrite_enabled),
            generator_temperature=s.generation_temperature,
            judge_model=s.effective_judge_model,
            critic_model=s.effective_critic_model,
        )

    def run(self, request: QueryRequest, request_id: str | None = None, emit=None) -> QueryResponse:
        """Answer a query. `emit(event, data)` receives streaming events (status, sources, token, reset)."""
        with self.activity.running():  # background indexing yields while this runs
            return self._run(request, request_id, emit)

    def _run(self, request: QueryRequest, request_id: str | None, emit) -> QueryResponse:
        run = self.resolve(request)
        trace = Trace(request_id)

        if self.retriever.store.count_chunks() == 0:
            response = self._response(
                request, trace, run, status="no_documents", answer=INSUFFICIENT_EVIDENCE_ANSWER,
                state={"retrieval_query": request.query, "query_rewritten": False},
            )
            self.metrics.record_query(response)
            log_query(response, include_content=self.settings.log_content)
            return response

        try:
            state = self.graph.invoke(
                {"query": request.query, "history": request.history, "run": run},
                config={"configurable": {"trace": trace, "emit": emit},
                        "recursion_limit": recursion_limit(run.max_retries)},
            )
        except Exception as exc:
            self.metrics.record_error(type(exc).__name__)
            logger.error("query failed", extra={"request_id": trace.request_id, "error": str(exc)})
            raise

        response = self._response(request, trace, run, status=state["status"], answer=state["answer"], state=state)
        self.metrics.record_query(response)
        log_query(response, include_content=self.settings.log_content)
        return response

    def _response(self, request: QueryRequest, trace: Trace, run: RunConfig, *, status, answer, state) -> QueryResponse:
        attempts = state.get("attempts", [])
        judged = [a for a in attempts if a.judge is not None]
        best = state.get("best_attempt")
        initial = judged[0].judge if judged else None
        final = attempts[best].judge if best is not None and attempts and attempts[best].judge else None
        retrieval = state.get("retrieval")
        chunks = retrieval.chunks if retrieval else []
        reliability = ReliabilityReport(
            enabled=run.mode == "adversarial" and bool(attempts),
            initial=initial,
            final=final,
            retries=state.get("retries", 0),
            max_retries=run.max_retries,
            improved=bool(initial and final and best and final.aggregate > initial.aggregate),
            attempts=attempts,
            judge_error=state.get("judge_error"),
        )
        return QueryResponse(
            request_id=trace.request_id,
            mode=run.mode,
            query=request.query,
            retrieval_query=state.get("retrieval_query", request.query),
            query_rewritten=state.get("query_rewritten", False),
            status=status,
            answer=answer,
            citations=state.get("citations", []),
            retrieved_chunks=chunks,
            reliability=reliability,
            trace=trace.to_out(
                retrieved_chunks=len(chunks),
                price_prompt_per_1m=self.settings.price_prompt_per_1m,
                price_completion_per_1m=self.settings.price_completion_per_1m,
            ),
        )
