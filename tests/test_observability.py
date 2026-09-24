import json
import logging

import pytest

from app.core.llm import LLMResponse
from app.observability.logging import JsonFormatter
from app.observability.metrics import _stats
from app.observability.tracing import Trace


def test_trace_spans_usage_and_breakdown():
    t = Trace("rid")
    with t.span("generation", chunks=3) as s:
        s.set(citations=2)
    t.add_span("hybrid_retrieval", 12.0, dense_candidates=5)
    t.add_span("reranking", 0.0, status="skipped")
    t.record_llm(LLMResponse(text="x", model="m", prompt_tokens=100, completion_tokens=10), "generate")
    t.record_llm(None, "critic")  # failed call: nothing recorded
    out = t.to_out(retrieved_chunks=3)
    assert out.request_id == "rid" and out.llm_calls == 1
    assert out.prompt_tokens == 100 and out.completion_tokens == 10
    assert out.retrieval_ms == 12.0 and out.rerank_ms == 0.0
    gen = next(s for s in out.spans if s.name == "generation")
    assert gen.attributes == {"chunks": 3, "citations": 2}
    assert out.estimated_cost_usd is None


def test_span_records_errors():
    t = Trace()
    with pytest.raises(ValueError):
        with t.span("judge"):
            raise ValueError("bad")
    assert t.spans[0].status == "error" and "bad" in t.spans[0].attributes["error"]


def test_json_formatter_includes_extras():
    record = logging.makeLogRecord({"name": "x", "levelname": "INFO", "msg": "span", "request_id": "abc", "duration_ms": 1.5})
    payload = json.loads(JsonFormatter().format(record))
    assert payload["msg"] == "span" and payload["request_id"] == "abc" and payload["duration_ms"] == 1.5


def test_logs_do_not_contain_answer_text(caplog):
    t = Trace()
    with caplog.at_level(logging.INFO, logger="rag.trace"):
        with t.span("generation", answer_preview="SECRET ANSWER " * 50, n=1):
            pass
    record = caplog.records[-1]
    assert "attr_answer_preview" not in record.__dict__  # long free text filtered
    assert record.__dict__["attr_n"] == 1


def test_latency_percentiles():
    s = _stats([float(i) for i in range(1, 101)])
    assert s.count == 100 and s.p50_ms == 51.0 and s.p95_ms == 95.0 and s.mean_ms == 50.5
    assert _stats([]).p50_ms is None


def test_query_log_records_details_and_respects_content_flag(settings, document_service, caplog):
    from app.retrieval.bm25 import BM25Index
    from app.retrieval.hybrid import HybridRetriever
    from app.schemas.query import QueryRequest
    from app.services.query_service import QueryService
    from app.observability.metrics import MetricsRegistry
    from tests.fakes import FakeLLM

    judge = json.dumps({"faithfulness": 0.95, "relevance": 1, "completeness": 1,
                        "unsupported_claims": ["secret claim text"], "reason": "one claim unsupported"})
    llm = FakeLLM({"generate": ["AUC 0.959 [1]."], "judge": [judge, judge.replace('["secret claim text"]', "[]")],
                   "critic": ['{"unsupported_claims": ["x"], "instructions": "fix"}'], "regenerate": ["AUC 0.959 [1]."]})
    document_service.ingest("paper.txt", b"The Transformer achieves ROC-AUC 0.959 on the test split.")
    retriever = HybridRetriever(settings, document_service.store, document_service.embedder, document_service.dense_index, BM25Index())
    retriever.refresh_sparse_index()
    service = QueryService(settings, llm, retriever, MetricsRegistry())

    with caplog.at_level(logging.INFO, logger="rag.query"):
        service.run(QueryRequest(query="What ROC-AUC does the Transformer reach?"))
    msgs = [r.msg for r in caplog.records if r.name == "rag.query"]
    assert msgs == ["query received", "retrieval", "reliability attempt", "reliability attempt", "query completed"]
    first_attempt = next(r for r in caplog.records if r.msg == "reliability attempt")
    assert first_attempt.failed_checks == ["unsupported_claims"] and first_attempt.unsupported_claims == ["secret claim text"]

    caplog.clear()
    settings.log_content = False
    with caplog.at_level(logging.INFO, logger="rag.query"):
        service.run(QueryRequest(query="What ROC-AUC does the Transformer reach?"))
    for r in caplog.records:
        assert "query" not in r.__dict__ and "unsupported_claims" not in r.__dict__
