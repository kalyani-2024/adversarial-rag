from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app import __version__
from app.api.dependencies import get_container
from app.observability.metrics import MetricsSnapshot
from app.services.container import Container

router = APIRouter(tags=["system"])


class HealthResponse(BaseModel):
    status: str
    version: str
    documents: int
    chunks: int
    vectors: int
    index_consistent: bool
    llm_configured: bool
    llm_model: str
    judge_model: str
    critic_model: str
    reranker_enabled: bool


@router.get("/health", response_model=HealthResponse, summary="Liveness + index status")
def health(container: Container = Depends(get_container)):
    s = container.settings
    chunks = container.store.count_chunks()
    vectors = container.documents.dense_index.size
    key = s.groq_api_key
    return HealthResponse(
        status="ok",
        version=__version__,
        documents=len(container.store.list_documents()),
        chunks=chunks,
        vectors=vectors,
        index_consistent=chunks == vectors,
        llm_configured=bool(key and key.get_secret_value().strip()),
        llm_model=s.llm_model,
        judge_model=s.effective_judge_model,
        critic_model=s.effective_critic_model,
        reranker_enabled=s.reranker_enabled,
    )


@router.get("/metrics", response_model=MetricsSnapshot, summary="Aggregate service metrics (in-process)")
def metrics(container: Container = Depends(get_container)):
    return container.metrics.snapshot()
