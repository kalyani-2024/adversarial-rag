"""In-process backend for single-process hosts (e.g. Streamlit Community Cloud).

Same interface as `client.RagClient`, but instead of calling the FastAPI server
over HTTP it calls the services directly: the same container, pipeline,
background ingestion, streaming events and logging. Selected with
RAG_BACKEND=embedded; local development and Docker keep using the HTTP API.
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client import APIError  # noqa: E402

logger = logging.getLogger("rag.ui.embedded")


def _export_secrets_to_env() -> None:
    """Streamlit Cloud keeps secrets in st.secrets; the backend reads environment variables."""
    try:
        for key, value in st.secrets.items():
            if isinstance(value, (str, int, float, bool)):
                os.environ.setdefault(key, str(value))
    except Exception:  # no secrets file locally
        pass


@st.cache_resource(show_spinner="Loading models and indexing the sample document (first start only)…")
def _container():
    """One backend per server process, shared by all sessions (cached across reruns)."""
    _export_secrets_to_env()
    default_seed = ROOT / "eval" / "corpus"
    if "SEED_DIR" not in os.environ and default_seed.is_dir():
        os.environ["SEED_DIR"] = str(default_seed)  # the demo comes up with the sample paper indexed

    from app.core.config import Settings
    from app.observability.logging import configure_logging
    from app.services.container import build_container

    settings = Settings()
    configure_logging(settings.log_level, json_logs=settings.log_json)
    container = build_container(settings)
    container.startup()
    logger.info("embedded backend ready", extra={"chunks": container.store.count_chunks()})
    return container


def _api_error(exc: Exception) -> APIError:
    from app.core.errors import AppError

    if isinstance(exc, AppError):
        return APIError(exc.status_code, exc.code, exc.message)
    logger.exception("embedded backend error")
    return APIError(500, "internal_error", "Internal error.")


class EmbeddedClient:
    def __init__(self) -> None:
        self.c = _container()

    def health(self) -> dict:
        return {"status": "ok", "chunks": self.c.store.count_chunks()}

    def list_documents(self) -> dict:
        docs = [d.model_dump(mode="json") for d in self.c.documents.list_documents()]
        return {"documents": docs, "total_documents": len(docs), "total_chunks": sum(d["num_chunks"] for d in docs)}

    def upload(self, name: str, content: bytes, mime: str) -> dict:
        try:
            info = self.c.documents.submit(name, content)
        except Exception as exc:
            raise _api_error(exc) from exc
        return {"document": info.model_dump(mode="json"), "message": f"Indexing '{info.filename}' in the background."}

    def delete(self, document_id: str) -> dict:
        try:
            removed = self.c.documents.delete(document_id)
        except Exception as exc:
            raise _api_error(exc) from exc
        return {"id": document_id, "chunks_removed": removed}

    def query_stream(self, query: str, history: list[dict], options: dict) -> Iterator[tuple[str, dict]]:
        """Run the pipeline in a worker thread and yield the same events as POST /query/stream."""
        from pydantic import ValidationError

        from app.observability.tracing import new_request_id
        from app.schemas.query import QueryRequest

        try:
            request = QueryRequest(query=query, history=history, options=options)
        except ValidationError as exc:
            raise APIError(422, "validation_error", exc.errors()[0].get("msg", "Invalid request.")) from exc

        events: queue.Queue = queue.Queue()

        def worker() -> None:
            try:
                response = self.c.queries.run(request, request_id=new_request_id(),
                                              emit=lambda e, d: events.put((e, d)))
                events.put(("final", response.model_dump(mode="json")))
            except Exception as exc:
                err = _api_error(exc)
                events.put(("error", {"code": err.code, "message": err.message, "status": err.status}))
            finally:
                events.put(None)

        threading.Thread(target=worker, daemon=True).start()
        while (item := events.get()) is not None:
            yield item
