"""The reliability pipeline as an explicit LangGraph state machine.

    START -> rewrite -> retrieve --(no evidence)--------------------------> abstain -> END
                           |
                           v
                        generate --(baseline mode: no judge)---------------> finalize -> END
                           |
                           v
                  +----> judge --(PASS | retries exhausted | judge error)--> finalize -> END
                  |        |
                  |      (FAIL)
                  |        v
                  |      critic
                  |        v
                  +--- regenerate

Routing decisions are pure functions of the state (see `route_*`), which is
what makes the loop testable and guarantees termination: every FAIL edge
increments `retries`, and `route_after_judge` stops at `max_retries`. LangGraph's
`recursion_limit` is a second, independent backstop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.core.errors import LLMError
from app.core.llm import LLMClient
from app.generation.critic import critique_answer
from app.generation.generator import GeneratedAnswer, generate_answer, is_abstention
from app.generation.judge import judge_answer
from app.generation.rewrite import decide_rewrite, rewrite_query
from app.observability.tracing import Trace
from app.retrieval.hybrid import HybridRetriever, evidence_check
from app.schemas.query import INSUFFICIENT_EVIDENCE_ANSWER, ChatTurn, Citation
from app.schemas.reliability import Attempt, Critique, JudgeResult, Thresholds
from app.schemas.retrieval import RetrievalResult


@dataclass(frozen=True)
class RunConfig:
    """Effective settings for one request (server defaults + request overrides)."""

    mode: Literal["adversarial", "baseline"]
    top_k: int
    max_retries: int
    thresholds: Thresholds
    use_reranker: bool
    use_query_rewrite: bool
    generator_temperature: float
    judge_model: str
    critic_model: str = ""


class PipelineState(TypedDict, total=False):
    # inputs
    query: str
    history: list[ChatTurn]
    run: RunConfig
    # retrieval
    retrieval_query: str
    query_rewritten: bool
    retrieval: RetrievalResult
    evidence_ok: bool
    evidence_reason: str
    # generation / reliability loop
    attempts: list[Attempt]
    attempt_citations: list[list[Citation]]
    abstained: bool
    retries: int
    pending_critique: Critique | None
    judge_error: str | None
    # outputs
    status: Literal["answered", "insufficient_evidence"]
    answer: str
    citations: list[Citation]
    best_attempt: int


def _trace(config: RunnableConfig) -> Trace:
    return config["configurable"]["trace"]


# --- routing (pure functions) ------------------------------------------------------------------
def route_after_retrieve(state: PipelineState) -> str:
    return "generate" if state.get("evidence_ok") else "abstain"


def route_after_generate(state: PipelineState) -> str:
    if state["run"].mode == "baseline":
        return "finalize"
    return "judge"


def route_after_judge(state: PipelineState) -> str:
    if state.get("judge_error"):
        return "finalize"  # cannot verify: return the answer flagged as unverified
    last = state["attempts"][-1]
    if last.judge is not None and last.judge.verdict == "PASS":
        return "finalize"
    if state.get("retries", 0) >= state["run"].max_retries:
        return "finalize"  # loop guard
    return "critic"


def select_best_attempt(attempts: list[Attempt]) -> int:
    """Rank by PASS, then grounded (no faithfulness failure), then aggregate score; ties -> earliest.

    Returning the *best* rather than the *latest* attempt means a retry can
    never make the final answer worse than the initial one, which is the
    failure mode of an always-on rewrite step. Grounding outranks the
    aggregate score so that a fluent-but-hallucinated answer never beats an
    honest "the sources don't cover X" answer that scores lower on relevance.
    """
    judged = [a for a in attempts if a.judge is not None]
    if not judged:
        return len(attempts) - 1
    best = max(
        judged,
        key=lambda a: (a.judge.verdict == "PASS", a.judge.grounded, round(a.judge.aggregate, 4), -a.attempt),
    )
    return best.attempt


# --- graph ------------------------------------------------------------------------------------
def build_graph(llm: LLMClient, retriever: HybridRetriever):
    def rewrite(state: PipelineState, config: RunnableConfig) -> dict[str, Any]:
        trace, run, query = _trace(config), state["run"], state["query"]
        decision = decide_rewrite(query, state.get("history", []))
        if not (run.use_query_rewrite and decision.should_rewrite):
            reason = decision.reason if run.use_query_rewrite else "disabled"
            trace.add_span("query_rewrite", 0.0, status="skipped", reason=reason)
            return {"retrieval_query": query, "query_rewritten": False}
        with trace.span("query_rewrite", reason=decision.reason) as span:
            new_query, resp = rewrite_query(llm, query, state.get("history", []), model=run.judge_model)
            trace.record_llm(resp, "rewrite")
            span.set(rewritten=new_query != query)
        return {"retrieval_query": new_query, "query_rewritten": new_query != query}

    def retrieve(state: PipelineState, config: RunnableConfig) -> dict[str, Any]:
        trace, run = _trace(config), state["run"]
        result = retriever.retrieve(state["retrieval_query"], top_k=run.top_k, use_reranker=run.use_reranker)
        ok, reason = evidence_check(result, retriever.settings)
        trace.add_span(
            "hybrid_retrieval",
            result.retrieval_ms,
            status="error" if result.dense_error and not result.chunks else "ok",
            dense_candidates=result.dense_candidates,
            bm25_candidates=result.bm25_candidates,
            fused_candidates=result.fused_candidates,
            dense_error=result.dense_error or "",
        )
        if result.reranker_used or result.reranker_error:
            trace.add_span(
                "reranking", result.rerank_ms, status="error" if result.reranker_error else "ok",
                fallback=bool(result.reranker_error), returned=len(result.chunks),
            )
        else:
            trace.add_span("reranking", 0.0, status="skipped", reason="disabled")
        trace.add_span("evidence_gate", 0.0, status="ok", passed=ok, reason=reason)
        return {"retrieval": result, "evidence_ok": ok, "evidence_reason": reason}

    def abstain(state: PipelineState) -> dict[str, Any]:
        return {
            "status": "insufficient_evidence", "abstained": True, "answer": INSUFFICIENT_EVIDENCE_ANSWER,
            "attempts": [], "attempt_citations": [], "citations": [],
        }

    def _record_generation(state: PipelineState, gen: GeneratedAnswer) -> dict[str, Any]:
        attempts = list(state.get("attempts", []))
        cites = list(state.get("attempt_citations", []))
        attempts.append(Attempt(attempt=len(attempts), answer=gen.answer))
        cites.append(gen.citations)
        return {"attempts": attempts, "attempt_citations": cites, "abstained": gen.abstained}

    def generate(state: PipelineState, config: RunnableConfig) -> dict[str, Any]:
        trace, run = _trace(config), state["run"]
        with trace.span("generation", chunks=len(state["retrieval"].chunks)) as span:
            gen = generate_answer(llm, state["retrieval_query"], state["retrieval"].chunks, temperature=run.generator_temperature)
            trace.record_llm(gen.response, "generate")
            span.set(citations=len(gen.citations), invalid_citations=len(gen.invalid_citations), abstained=gen.abstained)
        return {**_record_generation(state, gen), "retries": 0}

    def judge(state: PipelineState, config: RunnableConfig) -> dict[str, Any]:
        trace, run = _trace(config), state["run"]
        attempts = list(state["attempts"])
        current = attempts[-1]
        with trace.span("judge", attempt=current.attempt) as span:
            try:
                result, resp = judge_answer(
                    llm, state["retrieval_query"], state["retrieval"].chunks, current.answer, run.thresholds, model=run.judge_model
                )
            except LLMError as exc:
                span.status = "error"
                span.set(error=exc.message)
                return {"judge_error": exc.message}
            trace.record_llm(resp, "judge")
            span.set(verdict=result.verdict, faithfulness=result.faithfulness,
                     relevance=result.relevance, completeness=result.completeness,
                     failed_checks=",".join(result.failed_checks) or "none",
                     unsupported_claims=len(result.unsupported_claims))
        attempts[-1] = current.model_copy(update={"judge": result})
        return {"attempts": attempts}

    def critic(state: PipelineState, config: RunnableConfig) -> dict[str, Any]:
        trace, run = _trace(config), state["run"]
        attempts = list(state["attempts"])
        current = attempts[-1]
        with trace.span("critic", attempt=current.attempt) as span:
            critique, resp, error = critique_answer(
                llm, state["retrieval_query"], state["retrieval"].chunks, current.answer, current.judge, model=run.critic_model or run.judge_model
            )
            trace.record_llm(resp, "critic")
            if error:
                span.status = "error"
                span.set(fallback="judge-derived critique", error=error)
            span.set(issues=sum(len(v) for v in critique.model_dump().values() if isinstance(v, list)))
        attempts[-1] = current.model_copy(update={"critique": critique})
        return {"attempts": attempts, "pending_critique": critique}

    def regenerate(state: PipelineState, config: RunnableConfig) -> dict[str, Any]:
        trace, run = _trace(config), state["run"]
        retries = state.get("retries", 0) + 1
        with trace.span("regeneration", retry=retries) as span:
            gen = generate_answer(
                llm, state["retrieval_query"], state["retrieval"].chunks, temperature=run.generator_temperature,
                critique=state["pending_critique"], previous_answer=state["attempts"][-1].answer,
            )
            trace.record_llm(gen.response, "regenerate")
            span.set(citations=len(gen.citations), abstained=gen.abstained)
        return {**_record_generation(state, gen), "retries": retries, "pending_critique": None}

    def finalize(state: PipelineState) -> dict[str, Any]:
        attempts = state["attempts"]
        best = select_best_attempt(attempts)
        return {
            "best_attempt": best,
            "answer": attempts[best].answer,
            "citations": state["attempt_citations"][best],
            "status": "insufficient_evidence" if is_abstention(attempts[best].answer) else "answered",
        }

    g = StateGraph(PipelineState)
    for name, fn in [("rewrite", rewrite), ("retrieve", retrieve), ("abstain", abstain), ("generate", generate),
                     ("judge", judge), ("critic", critic), ("regenerate", regenerate), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "rewrite")
    g.add_edge("rewrite", "retrieve")
    g.add_conditional_edges("retrieve", route_after_retrieve, {"generate": "generate", "abstain": "abstain"})
    g.add_conditional_edges("generate", route_after_generate, {"judge": "judge", "finalize": "finalize"})
    g.add_conditional_edges("judge", route_after_judge, {"critic": "critic", "finalize": "finalize"})
    g.add_edge("critic", "regenerate")
    g.add_edge("regenerate", "judge")
    g.add_edge("abstain", END)
    g.add_edge("finalize", END)
    return g.compile()


def recursion_limit(max_retries: int) -> int:
    """Upper bound on graph steps: 4 fixed nodes + 3 per retry (+ judge/finalize) + slack."""
    return 8 + 3 * max_retries + 4
