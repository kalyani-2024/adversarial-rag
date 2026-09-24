import json

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.services.container import build_container
from tests.conftest import make_pdf
from tests.fakes import FakeLLM, HashEmbedder

PAPER = b"The Transformer classifier achieves ROC-AUC 0.959 on the held-out split. The LSTM reaches 0.601."
JUDGE_PASS = json.dumps({"faithfulness": 0.95, "relevance": 0.9, "completeness": 0.9, "unsupported_claims": [], "reason": "ok"})


@pytest.fixture
def llm():
    return FakeLLM({"generate": ["The Transformer reaches ROC-AUC 0.959 [1]."], "judge": [JUDGE_PASS]})


@pytest.fixture
def client(settings, llm):
    app = create_app(settings, container_factory=lambda s: build_container(s, llm=llm, embedder=HashEmbedder()))
    with TestClient(app) as c:
        yield c


def _upload(client, name="paper.txt", content=PAPER, mime="text/plain"):
    return client.post("/documents", files={"file": (name, content, mime)})


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["chunks"] == 0 and body["index_consistent"]
    assert body["llm_configured"] is False


def test_document_lifecycle(client):
    r = _upload(client)
    assert r.status_code == 201
    doc = r.json()["document"]
    assert doc["filename"] == "paper.txt" and doc["num_chunks"] >= 1

    listing = client.get("/documents").json()
    assert listing["total_documents"] == 1 and listing["total_chunks"] == doc["num_chunks"]

    dup = _upload(client, name="copy.txt")
    assert dup.status_code == 409
    assert dup.json()["error"]["code"] == "duplicate_document"
    assert dup.json()["error"]["details"]["document_id"] == doc["id"]

    d = client.delete(f"/documents/{doc['id']}")
    assert d.status_code == 200 and d.json()["chunks_removed"] == doc["num_chunks"]
    assert client.get("/documents").json()["total_chunks"] == 0
    assert client.delete(f"/documents/{doc['id']}").status_code == 404


def test_pdf_upload_has_pages(client):
    r = _upload(client, "p.pdf", make_pdf(["Alpha content", "Beta content"]), "application/pdf")
    assert r.status_code == 201 and r.json()["document"]["num_pages"] == 2


@pytest.mark.parametrize(
    "name,content,status,code",
    [
        ("image.png", b"\x89PNG....", 415, "unsupported_file_type"),
        ("broken.pdf", b"%PDF-1.4 garbage", 422, "document_parse_error"),
        ("empty.txt", b"   ", 422, "document_parse_error"),
    ],
)
def test_upload_errors(client, name, content, status, code):
    r = _upload(client, name, content)
    assert r.status_code == status
    assert r.json()["error"]["code"] == code
    assert r.json()["error"]["request_id"]


def test_query_end_to_end(client):
    _upload(client)
    r = client.post("/query", json={"query": "What ROC-AUC does the Transformer achieve?"}, headers={"x-request-id": "req-123"})
    assert r.status_code == 200
    body = r.json()
    assert r.headers["x-request-id"] == "req-123" and body["request_id"] == "req-123"
    assert body["status"] == "answered" and body["citations"][0]["label"].startswith("paper.txt")
    assert body["retrieved_chunks"][0]["rrf_score"] > 0
    assert body["reliability"]["final"]["verdict"] == "PASS"
    assert body["trace"]["llm_calls"] == 2

    metrics = client.get("/metrics").json()
    assert metrics["queries_total"] == 1 and metrics["llm_calls_total"] == 2


def test_query_without_documents(client):
    body = client.post("/query", json={"query": "anything?"}).json()
    assert body["status"] == "no_documents"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"query": ""},
        {"query": "   "},
        {"query": "x" * 2001},
        {"query": "ok", "options": {"top_k": 0}},
        {"query": "ok", "options": {"max_retries": 99}},
        {"query": "ok", "options": {"faithfulness_threshold": 1.5}},
        {"query": "ok", "options": {"mode": "yolo"}},
    ],
)
def test_query_validation(client, payload):
    r = client.post("/query", json=payload)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


def test_llm_rate_limit_maps_to_429(settings):
    from app.core.errors import LLMRateLimitError

    def boom(messages, purpose):
        raise LLMRateLimitError("rate limited")

    llm = FakeLLM({"generate": boom})
    app = create_app(settings, container_factory=lambda s: build_container(s, llm=llm, embedder=HashEmbedder()))
    with TestClient(app) as c:
        _upload(c)
        r = c.post("/query", json={"query": "What ROC-AUC does the Transformer achieve?"})
    assert r.status_code == 429 and r.headers["retry-after"] == "10"
    assert r.json()["error"]["code"] == "llm_rate_limited"


def test_legacy_ingest_endpoint_still_works(client):
    r = client.post("/ingest", files={"file": ("old.txt", b"legacy client content", "text/plain")})
    assert r.status_code == 200


def test_openapi_documents_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/documents", "/documents/{document_id}", "/query", "/metrics"} <= set(paths)
