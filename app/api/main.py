"""FastAPI application factory.

Routes are thin: validate input (Pydantic), call a service, return a schema.
Cross-cutting concerns live here: request ids, access logging, and mapping
domain errors to consistent JSON error bodies.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.routes import documents, query, system
from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.observability.logging import configure_logging
from app.observability.tracing import new_request_id
from app.services.container import Container, build_container

logger = logging.getLogger("rag.api")

DESCRIPTION = """
Hybrid-retrieval RAG with a **conditional adversarial reliability loop**.

`POST /query` pipeline: query rewrite (only if needed) → dense (FAISS) + BM25 retrieval →
Reciprocal Rank Fusion → cross-encoder rerank → evidence gate → grounded generation with
citations → reliability judge → *(only on FAIL)* adversarial critic → regeneration → judge …
up to `MAX_RETRIES`. Every response carries citations, retrieval scores, judge scores and a
per-stage execution trace.
"""


def _error_body(request: Request, code: str, message: str, details: dict | None = None) -> dict:
    return {"error": {"code": code, "message": message, "details": details or {},
                      "request_id": getattr(request.state, "request_id", None)}}


def create_app(settings: Settings | None = None, container_factory: Callable[[Settings], Container] | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, json_logs=settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        container = (container_factory or build_container)(settings)
        container.startup()
        app.state.container = container
        yield
        container.shutdown()

    app = FastAPI(
        title="RAG Reliability Lab API",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = request.headers.get("x-request-id") or new_request_id()
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = request.state.request_id
        logger.info(
            "request",
            extra={"request_id": request.state.request_id, "method": request.method, "path": request.url.path,
                   "status_code": response.status_code, "duration_ms": round((time.perf_counter() - start) * 1000, 1)},
        )
        return response

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        level = logging.WARNING if exc.status_code < 500 else logging.ERROR
        logger.log(level, "request failed", extra={"request_id": getattr(request.state, "request_id", None),
                                                   "code": exc.code, "error": exc.message})
        headers = {"Retry-After": "10"} if exc.status_code == 429 else None
        return JSONResponse(_error_body(request, exc.code, exc.message, exc.details), status_code=exc.status_code, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        details = {"errors": [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]}
        return JSONResponse(_error_body(request, "validation_error", "Invalid request.", details), status_code=422)

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        logger.exception("unhandled error", extra={"request_id": getattr(request.state, "request_id", None)})
        return JSONResponse(_error_body(request, "internal_error", "Internal server error."), status_code=500)

    app.include_router(system.router)
    app.include_router(documents.router)
    app.include_router(query.router)
    app.include_router(documents.legacy_router)
    return app
