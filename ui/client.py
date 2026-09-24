"""Thin HTTP client for the RAG API (the UI never imports backend internals)."""

from __future__ import annotations

import json
from typing import Any

import requests


class APIError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


class RagClient:
    def __init__(self, base_url: str, timeout: float = 180.0) -> None:
        # On Windows "localhost" often tries IPv6 (::1) first and waits ~2 s before falling
        # back to IPv4, on *every* request. uvicorn listens on 127.0.0.1, so go there directly.
        self.base_url = base_url.rstrip("/").replace("://localhost", "://127.0.0.1")
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs) -> Any:
        try:
            resp = requests.request(method, f"{self.base_url}{path}", timeout=kwargs.pop("timeout", self.timeout), **kwargs)
        except requests.RequestException as exc:
            raise APIError(
                0, "unreachable",
                f"Cannot reach the API at {self.base_url} ({type(exc).__name__}). Start it with "
                "`uvicorn api:app --port 8000` (model warm-up takes ~10-30 s), then refresh this page.",
            ) from exc
        if resp.status_code >= 400:
            try:
                err = resp.json().get("error", {})
                raise APIError(resp.status_code, err.get("code", "error"), err.get("message", resp.text))
            except ValueError:
                raise APIError(resp.status_code, "error", resp.text[:300]) from None
        return resp.json()

    def health(self) -> dict:
        return self._request("GET", "/health", timeout=5)

    def list_documents(self) -> dict:
        return self._request("GET", "/documents", timeout=10)

    def upload(self, name: str, content: bytes, mime: str) -> dict:
        return self._request("POST", "/documents", files={"file": (name, content, mime)}, timeout=300)

    def delete(self, document_id: str) -> dict:
        return self._request("DELETE", f"/documents/{document_id}", timeout=30)

    def query(self, query: str, history: list[dict], options: dict) -> dict:
        return self._request("POST", "/query", json={"query": query, "history": history, "options": options})

    def query_stream(self, query: str, history: list[dict], options: dict):
        """Yield `(event, data)` pairs from POST /query/stream (Server-Sent Events)."""
        try:
            resp = requests.post(f"{self.base_url}/query/stream", json={"query": query, "history": history, "options": options},
                                 stream=True, timeout=(5, self.timeout))
        except requests.RequestException as exc:
            raise APIError(0, "unreachable", f"Cannot reach the API at {self.base_url} ({type(exc).__name__}). Start it with "
                           "`uvicorn api:app --port 8000`, then refresh this page.") from exc
        if resp.status_code >= 400:  # validation errors arrive as normal JSON before any streaming
            try:
                err = resp.json().get("error", {})
            except ValueError:
                err = {}
            raise APIError(resp.status_code, err.get("code", "error"), err.get("message", resp.text[:300]))
        resp.encoding = "utf-8"
        event, data_lines = None, []
        with resp:
            for line in resp.iter_lines(decode_unicode=True):
                if line is None:
                    continue
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
                elif line == "" and event:
                    yield event, json.loads("\n".join(data_lines))
                    event, data_lines = None, []

    def metrics(self) -> dict:
        return self._request("GET", "/metrics", timeout=10)
