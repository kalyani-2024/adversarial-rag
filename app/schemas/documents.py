"""Document and chunk models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ChunkRecord(BaseModel):
    """A persisted chunk. `id` is the integer key shared by SQLite and FAISS."""

    id: int
    chunk_id: str = Field(description="Stable human-readable id: '<document_id>:<chunk_index>'")
    document_id: str
    document_name: str
    chunk_index: int
    page: int | None = Field(None, description="1-based page number, when the format has pages")
    text: str
    context: str = Field("", description="Index-time description prepended for retrieval; not part of the passage")

    @property
    def index_text(self) -> str:
        return f"{self.context}\n{self.text}" if self.context else self.text


class DocumentInfo(BaseModel):
    id: str
    status: Literal["processing", "ready", "failed"] = "ready"
    error: str | None = None
    progress: float | None = Field(None, description="0..1 while processing (embedding progress)")
    stage: str | None = Field(None, description="parsing | embedding | indexing while processing")
    filename: str
    file_type: str
    content_hash: str = Field(description="SHA-256 of the raw bytes; used for de-duplication")
    size_bytes: int
    num_pages: int | None = None
    num_chunks: int
    created_at: datetime


class DocumentUploadResponse(BaseModel):
    document: DocumentInfo
    message: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total_documents: int
    total_chunks: int


class DocumentDeleteResponse(BaseModel):
    id: str
    chunks_removed: int
    message: str
