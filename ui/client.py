"""Thin HTTP client for the RAG API (the UI never imports backend internals)."""

from __future__ import annotations

from typing import Any

import requests


class APIError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


class RagClient:
    def __init__(self, base_url: str, timeout: float = 180.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs) -> Any:
        try:
            resp = requests.request(method, f"{self.base_url}{path}", timeout=kwargs.pop("timeout", self.timeout), **kwargs)
        except requests.RequestException as exc:
            raise APIError(0, "unreachable", f"Cannot reach the API at {self.base_url} ({type(exc).__name__}).") from exc
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

    def metrics(self) -> dict:
        return self._request("GET", "/metrics", timeout=10)
