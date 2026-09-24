"""FastAPI dependency providers (resolved from the app-scoped container)."""

from __future__ import annotations

from fastapi import Request

from app.services.container import Container


def get_container(request: Request) -> Container:
    return request.app.state.container
