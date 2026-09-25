"""The embedded (in-process) UI backend must behave like the HTTP client."""

import json
import sys
from pathlib import Path

import pytest

from app.services.container import build_container
from tests.fakes import FakeLLM, HashEmbedder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))
import embedded  # noqa: E402
from client import APIError  # noqa: E402

PASS = json.dumps({"faithfulness": 1, "relevance": 1, "completeness": 1, "unsupported_claims": [], "reason": "ok"})
PAPER = b"The Transformer classifier achieves ROC-AUC 0.959 on the held-out split."


@pytest.fixture
def client(settings, monkeypatch):
    container = build_container(settings, llm=FakeLLM({"generate": ["It reaches ROC-AUC 0.959 [1]."], "judge": [PASS]}),
                                embedder=HashEmbedder())
    container.startup()
    monkeypatch.setattr(embedded, "_container", lambda: container)
    yield embedded.EmbeddedClient()
    container.shutdown()


def test_upload_list_delete(client):
    doc = client.upload("paper.txt", PAPER, "text/plain")["document"]
    assert doc["status"] == "processing"
    client.c.documents._executor.shutdown(wait=True)
    listed = client.list_documents()
    assert listed["total_documents"] == 1 and listed["documents"][0]["status"] == "ready"
    assert client.delete(doc["id"])["chunks_removed"] >= 1


def test_errors_map_to_api_error(client):
    with pytest.raises(APIError) as exc:
        client.upload("image.png", b"\x89PNG", "image/png")
    assert exc.value.status == 415 and exc.value.code == "unsupported_file_type"
    with pytest.raises(APIError) as exc:
        list(client.query_stream("   ", [], {}))
    assert exc.value.status == 422


def test_query_stream_matches_http_events(client):
    client.c.documents.ingest("paper.txt", PAPER)
    events = list(client.query_stream("What ROC-AUC does the Transformer achieve?", [], {}))
    kinds = [e for e, _ in events]
    assert kinds[0] == "status" and "sources" in kinds and kinds[-1] == "final"
    assert "".join(d["text"] for e, d in events if e == "token") == "It reaches ROC-AUC 0.959 [1]."
    final = events[-1][1]
    assert final["status"] == "answered" and final["citations"][0]["index"] == 1
