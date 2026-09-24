"""Retrieval models. Every stage's score is kept so retrieval is inspectable."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    chunk_index: int
    page: int | None = None
    text: str

    dense_score: float | None = Field(None, description="Cosine similarity (normalized embeddings)")
    dense_rank: int | None = None
    bm25_score: float | None = None
    bm25_rank: int | None = None
    rrf_score: float | None = Field(None, description="Reciprocal Rank Fusion score")
    fusion_rank: int | None = None
    rerank_score: float | None = Field(None, description="Cross-encoder relevance logit")
    final_rank: int | None = None


class RetrievalResult(BaseModel):
    """Output of the hybrid retrieval + rerank stage."""

    chunks: list[RetrievedChunk]
    dense_candidates: int = 0
    bm25_candidates: int = 0
    fused_candidates: int = 0
    reranker_used: bool = False
    reranker_error: str | None = None
    dense_error: str | None = None
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
