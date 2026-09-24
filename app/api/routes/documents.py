from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile, status

from app.api.dependencies import get_container
from app.schemas.documents import DocumentDeleteResponse, DocumentListResponse, DocumentUploadResponse
from app.services.container import Container

router = APIRouter(prefix="/documents", tags=["documents"])


def _upload(file: UploadFile, container: Container) -> DocumentUploadResponse:
    content = file.file.read()
    info = container.documents.ingest(file.filename or "upload", content)
    container.metrics.record_document(ingested=1)
    return DocumentUploadResponse(document=info, message=f"Indexed '{info.filename}' into {info.num_chunks} chunks.")


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a document (PDF, TXT, Markdown, DOCX)",
    responses={409: {"description": "Duplicate content"}, 413: {"description": "Too large"},
               415: {"description": "Unsupported type"}, 422: {"description": "Unparseable"}},
)
def upload_document(file: UploadFile = File(...), container: Container = Depends(get_container)):
    # Sync handler: FastAPI runs it in a worker thread, so parsing/embedding never blocks the event loop.
    return _upload(file, container)


@router.get("", response_model=DocumentListResponse, summary="List indexed documents")
def list_documents(container: Container = Depends(get_container)):
    docs = container.documents.list_documents()
    return DocumentListResponse(documents=docs, total_documents=len(docs), total_chunks=sum(d.num_chunks for d in docs))


@router.delete("/{document_id}", response_model=DocumentDeleteResponse, summary="Delete a document and its chunks")
def delete_document(document_id: str, container: Container = Depends(get_container)):
    removed = container.documents.delete(document_id)
    container.metrics.record_document(deleted=1)
    return DocumentDeleteResponse(id=document_id, chunks_removed=removed, message=f"Deleted document and {removed} chunks.")


legacy_router = APIRouter(tags=["legacy"])


@legacy_router.post("/ingest", response_model=DocumentUploadResponse, deprecated=True, summary="Deprecated: use POST /documents")
def legacy_ingest(file: UploadFile = File(...), container: Container = Depends(get_container)):
    return _upload(file, container)
