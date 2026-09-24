import json

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.services.container import build_container
from tests.fakes import FakeLLM, HashEmbedder

PAPER = b"The Transformer classifier achieves ROC-AUC 0.959 on the held-out split. The LSTM reaches 0.601."
PASS = json.dumps({"faithfulness": 1, "relevance": 1, "completeness": 1, "unsupported_claims": [], "reason": "ok"})
FAIL = json.dumps({"faithfulness": 0.9, "relevance": 1, "completeness": 1, "unsupported_claims": ["invented"], "reason": "x"})
CRIT = json.dumps({"unsupported_claims": ["invented"], "instructions": "remove it"})


def _events(client, query):
    with client.stream("POST", "/query/stream", json={"query": query}) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def _client(settings, llm):
    app = create_app(settings, container_factory=lambda s: build_container(s, llm=llm, embedder=HashEmbedder()))
    return TestClient(app)


def test_stream_pass(settings):
    llm = FakeLLM({"generate": ["The Transformer reaches ROC-AUC 0.959 [1]."], "judge": [PASS]})
    with _client(settings, llm) as c:
        c.post("/documents?wait=true", files={"file": ("p.txt", PAPER, "text/plain")})
        ev = _events(c, "What ROC-AUC does the Transformer achieve?")
    kinds = [e for e, _ in ev]
    assert kinds[0] == "status" and "sources" in kinds and kinds[-1] == "final"
    tokens = "".join(d["text"] for e, d in ev if e == "token")
    assert tokens == "The Transformer reaches ROC-AUC 0.959 [1]."
    assert [d["stage"] for e, d in ev if e == "status"] == ["retrieving", "generating", "verifying"]
    final = ev[-1][1]
    assert final["answer"] == tokens and final["citations"][0]["index"] == 1
    assert ev[kinds.index("sources")][1]["sources"][0]["index"] == 1


def test_stream_reset_on_retry(settings):
    llm = FakeLLM({"generate": ["Draft with invented claim [1]."], "judge": [FAIL, PASS], "critic": [CRIT],
                   "regenerate": ["Clean answer [1]."]})
    with _client(settings, llm) as c:
        c.post("/documents?wait=true", files={"file": ("p.txt", PAPER, "text/plain")})
        ev = _events(c, "What ROC-AUC does the Transformer achieve?")
    kinds = [e for e, _ in ev]
    reset = kinds.index("reset")
    after = "".join(d["text"] for e, d in ev[reset:] if e == "token")
    assert after == "Clean answer [1]." and ev[-1][1]["answer"] == "Clean answer [1]."
    assert "revising" in [d["stage"] for e, d in ev if e == "status"]


def test_stream_abstain_has_no_tokens(settings):
    with _client(settings, FakeLLM()) as c:
        c.post("/documents?wait=true", files={"file": ("p.txt", PAPER, "text/plain")})
        ev = _events(c, "quantum chromodynamics lattice gauge theory")
    assert "token" not in [e for e, _ in ev] and ev[-1][1]["status"] == "insufficient_evidence"


def test_stream_error_event(settings):
    from app.core.errors import LLMRateLimitError

    def boom(messages, purpose):
        raise LLMRateLimitError("rate limited")

    with _client(settings, FakeLLM({"generate": boom})) as c:
        c.post("/documents?wait=true", files={"file": ("p.txt", PAPER, "text/plain")})
        ev = _events(c, "What ROC-AUC does the Transformer achieve?")
    assert ev[-1] == ("error", {"code": "llm_rate_limited", "message": "rate limited", "status": 429})


def test_stream_validation_is_plain_422(settings):
    with _client(settings, FakeLLM()) as c:
        r = c.post("/query/stream", json={"query": "   "})
    assert r.status_code == 422
