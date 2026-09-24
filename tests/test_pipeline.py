import json

import pytest

from app.core.errors import LLMRateLimitError
from app.generation.judge import apply_thresholds
from app.observability.metrics import MetricsRegistry
from app.pipeline.graph import RunConfig, route_after_judge, select_best_attempt
from app.retrieval.bm25 import BM25Index
from app.retrieval.hybrid import HybridRetriever
from app.schemas.query import INSUFFICIENT_EVIDENCE_ANSWER, QueryOptions, QueryRequest
from app.schemas.reliability import Attempt, JudgeScores, Thresholds
from app.services.query_service import QueryService
from tests.fakes import FakeLLM

PAPER = (
    "The Transformer classifier achieves ROC-AUC 0.959 with recall 0.894 on the held-out test split.\n\n"
    "The LSTM autoencoder baseline reaches ROC-AUC 0.601 on the same split.\n\n"
    "Sessions were sampled from roughly 15,000 students in the EdNet-KT2 dataset."
)
T = Thresholds(faithfulness=0.8, relevance=0.7, completeness=0.6)


def judge_json(f: float, r: float = 0.9, c: float = 0.9, claims: list[str] | None = None) -> str:
    return json.dumps({"faithfulness": f, "relevance": r, "completeness": c,
                       "unsupported_claims": claims or [], "reason": f"faithfulness {f}"})


CRITIQUE = json.dumps({"unsupported_claims": ["trained on ImageNet"], "contradictions": [], "missing_evidence": [],
                       "weak_reasoning": [], "irrelevant_content": [], "instructions": "Remove the ImageNet claim."})


def make_service(settings, document_service, llm, *, ingest=True) -> QueryService:
    if ingest:
        document_service.ingest("paper.txt", PAPER.encode())
    retriever = HybridRetriever(settings, document_service.store, document_service.embedder,
                                document_service.dense_index, BM25Index())
    retriever.refresh_sparse_index()
    return QueryService(settings, llm, retriever, MetricsRegistry())


def ask(service, query="What ROC-AUC does the Transformer achieve?", **options):
    return service.run(QueryRequest(query=query, options=QueryOptions(**options)))


# --- pure logic --------------------------------------------------------------------------------
def test_threshold_logic():
    passed = apply_thresholds(JudgeScores(faithfulness=0.8, relevance=0.7, completeness=0.6), T)
    assert passed.verdict == "PASS" and passed.failed_checks == []  # boundary is inclusive
    failed = apply_thresholds(JudgeScores(faithfulness=0.79, relevance=0.2, completeness=0.9), T)
    assert failed.verdict == "FAIL" and failed.failed_checks == ["faithfulness", "relevance"]


def test_unsupported_claims_fail_even_with_high_score():
    scores = JudgeScores(faithfulness=0.95, relevance=0.9, completeness=0.9, unsupported_claims=["made up"])
    assert apply_thresholds(scores, T).failed_checks == ["unsupported_claims"]
    lenient = T.model_copy(update={"fail_on_unsupported_claims": False})
    assert apply_thresholds(scores, lenient).verdict == "PASS"


def _state(verdict_scores: list[float], retries: int, max_retries: int, judge_error=None):
    attempts = [Attempt(attempt=i, answer=f"a{i}", judge=apply_thresholds(JudgeScores(faithfulness=f, relevance=0.9, completeness=0.9), T))
                for i, f in enumerate(verdict_scores)]
    run = RunConfig(mode="adversarial", top_k=5, max_retries=max_retries, thresholds=T,
                    use_reranker=False, use_query_rewrite=False, generator_temperature=0, judge_model="m")
    return {"attempts": attempts, "retries": retries, "run": run, "judge_error": judge_error}


def test_route_after_judge():
    assert route_after_judge(_state([0.95], 0, 2)) == "finalize"   # PASS
    assert route_after_judge(_state([0.2], 0, 2)) == "critic"      # FAIL, budget left
    assert route_after_judge(_state([0.2, 0.3, 0.4], 2, 2)) == "finalize"  # budget exhausted
    assert route_after_judge(_state([0.2], 0, 0)) == "finalize"    # retries disabled
    assert route_after_judge(_state([0.2], 0, 2, judge_error="boom")) == "finalize"


def test_grounded_attempt_beats_higher_scoring_hallucination():
    hallucinated = Attempt(attempt=0, answer="fluent but invented", judge=apply_thresholds(
        JudgeScores(faithfulness=0.95, relevance=1.0, completeness=1.0, unsupported_claims=["invented"]), T))
    honest = Attempt(attempt=1, answer="sources do not cover X", judge=apply_thresholds(
        JudgeScores(faithfulness=1.0, relevance=0.5, completeness=0.5), T))
    assert hallucinated.judge.verdict == honest.judge.verdict == "FAIL"
    assert hallucinated.judge.aggregate > honest.judge.aggregate
    assert select_best_attempt([hallucinated, honest]) == 1


def test_select_best_attempt_prefers_pass_then_score_then_earliest():
    assert select_best_attempt(_state([0.5, 0.95, 0.9], 2, 2)["attempts"]) == 1
    assert select_best_attempt(_state([0.7, 0.3], 1, 2)["attempts"]) == 0  # retry got worse -> keep original
    assert select_best_attempt(_state([0.9, 0.9], 1, 2)["attempts"]) == 0  # tie -> earliest


# --- full graph ------------------------------------------------------------------------------------
def test_pass_first_time_makes_no_extra_calls(settings, document_service):
    llm = FakeLLM({"generate": ["The Transformer reaches ROC-AUC 0.959 [1]."], "judge": [judge_json(0.95)]})
    r = ask(make_service(settings, document_service, llm))
    assert r.status == "answered" and r.reliability.final.verdict == "PASS"
    assert r.reliability.retries == 0 and not r.reliability.improved
    assert [p for p, _ in llm.calls] == ["generate", "judge"]
    assert r.trace.llm_calls == 2
    assert r.citations and r.citations[0].document_name == "paper.txt"
    names = [s.name for s in r.trace.spans]
    assert names[:4] == ["query_rewrite", "hybrid_retrieval", "reranking", "evidence_gate"]
    assert "critic" not in names


def test_fail_then_pass_triggers_one_retry(settings, document_service):
    llm = FakeLLM({
        "generate": ["It reaches 0.959 and was trained on ImageNet [1]."],
        "judge": [judge_json(0.4, claims=["trained on ImageNet"]), judge_json(0.95)],
        "critic": [CRITIQUE],
        "regenerate": ["The Transformer reaches ROC-AUC 0.959 [1]."],
    })
    r = ask(make_service(settings, document_service, llm))
    assert [p for p, _ in llm.calls] == ["generate", "judge", "critic", "regenerate", "judge"]
    rel = r.reliability
    assert rel.retries == 1 and rel.improved
    assert rel.initial.verdict == "FAIL" and rel.final.verdict == "PASS"
    assert rel.attempts[0].critique.unsupported_claims == ["trained on ImageNet"]
    assert "ImageNet" not in r.answer
    assert [s.name for s in r.trace.spans if s.name in {"judge", "critic", "regeneration"}] == \
        ["judge", "critic", "regeneration", "judge"]


def test_max_retries_bounds_the_loop(settings, document_service):
    llm = FakeLLM({"generate": ["bad [1]"], "judge": [judge_json(0.1)], "critic": [CRITIQUE], "regenerate": ["still bad [1]"]})
    r = ask(make_service(settings, document_service, llm), max_retries=2)
    assert r.reliability.retries == 2
    assert llm.count("judge") == 3 and llm.count("critic") == 2 and llm.count("regenerate") == 2
    assert r.reliability.final.verdict == "FAIL"
    assert r.status == "answered"  # best effort answer is still returned, flagged FAIL


def test_worse_retry_does_not_replace_better_answer(settings, document_service):
    llm = FakeLLM({"generate": ["original [1]"], "judge": [judge_json(0.7), judge_json(0.2)],
                   "critic": [CRITIQUE], "regenerate": ["worse rewrite [1]"]})
    r = ask(make_service(settings, document_service, llm), max_retries=1)
    assert r.answer == "original [1]"
    assert not r.reliability.improved


def test_zero_retries_never_calls_critic(settings, document_service):
    llm = FakeLLM({"generate": ["x [1]"], "judge": [judge_json(0.1)]})
    ask(make_service(settings, document_service, llm), max_retries=0)
    assert llm.count("critic") == 0 and llm.count("regenerate") == 0


def test_baseline_mode_skips_judge(settings, document_service):
    llm = FakeLLM({"generate": ["The Transformer reaches ROC-AUC 0.959 [1]."]})
    r = ask(make_service(settings, document_service, llm), mode="baseline")
    assert [p for p, _ in llm.calls] == ["generate"]
    assert not r.reliability.enabled and r.reliability.final is None


def test_insufficient_evidence_makes_zero_llm_calls(settings, document_service):
    llm = FakeLLM()
    r = ask(make_service(settings, document_service, llm), query="quantum entanglement in black holes")
    assert r.status == "insufficient_evidence"
    assert r.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert llm.calls == [] and r.citations == []


def test_empty_index_returns_no_documents(settings, document_service):
    llm = FakeLLM()
    r = ask(make_service(settings, document_service, llm, ingest=False))
    assert r.status == "no_documents" and llm.calls == []


def test_generator_abstention_is_reported(settings, document_service):
    llm = FakeLLM({"generate": [INSUFFICIENT_EVIDENCE_ANSWER], "judge": [judge_json(1.0, 1.0, 1.0)]})
    r = ask(make_service(settings, document_service, llm))
    assert r.status == "insufficient_evidence" and r.citations == []


def test_judge_failure_returns_unverified_answer(settings, document_service):
    llm = FakeLLM({"generate": ["answer [1]"], "judge": ["this is not json"]})
    r = ask(make_service(settings, document_service, llm))
    assert r.status == "answered" and r.answer == "answer [1]"
    assert r.reliability.judge_error and r.reliability.final is None
    assert llm.count("critic") == 0


def test_critic_failure_falls_back_to_judge_critique(settings, document_service):
    llm = FakeLLM({"generate": ["bad [1]"], "judge": [judge_json(0.3, claims=["made-up claim"]), judge_json(0.95)],
                   "critic": ["garbage"], "regenerate": ["good [1]"]})
    r = ask(make_service(settings, document_service, llm), max_retries=1)
    assert r.reliability.attempts[0].critique.unsupported_claims == ["made-up claim"]
    assert r.answer == "good [1]"
    critic_span = next(s for s in r.trace.spans if s.name == "critic")
    assert critic_span.status == "error"


def test_generation_rate_limit_propagates(settings, document_service):
    def rate_limited(messages, purpose):
        raise LLMRateLimitError("rate limited")

    service = make_service(settings, document_service, FakeLLM({"generate": rate_limited}))
    with pytest.raises(LLMRateLimitError):
        ask(service)
    assert service.metrics.snapshot().errors_by_type == {"LLMRateLimitError": 1}


def test_metrics_accumulate(settings, document_service):
    llm = FakeLLM({"generate": ["bad [1]"], "judge": [judge_json(0.3), judge_json(0.95)],
                   "critic": [CRITIQUE], "regenerate": ["good [1]"]})
    service = make_service(settings, document_service, llm)
    ask(service)
    snap = service.metrics.snapshot()
    assert snap.queries_total == 1 and snap.retry_rate == 1.0 and snap.improved_by_loop == 1
    assert snap.llm_calls_total == 5 and snap.latency_by_stage["judge"].count == 2


def test_cost_estimate_only_when_prices_configured(settings, document_service):
    llm = FakeLLM({"generate": ["a [1]"], "judge": [judge_json(0.95)]})
    service = make_service(settings, document_service, llm)
    assert ask(service).trace.estimated_cost_usd is None
    settings.price_prompt_per_1m, settings.price_completion_per_1m = 1.0, 2.0
    r = ask(service)
    # FakeLLM: 100 prompt + 20 completion tokens per call, 2 calls
    assert r.trace.estimated_cost_usd == pytest.approx(200 / 1e6 * 1.0 + 40 / 1e6 * 2.0)
