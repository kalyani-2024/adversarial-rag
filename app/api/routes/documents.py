from __future__ import annotations

from fastapi import APIRouter, Depends, File, Query, UploadFile, status
from fastapi.responses import JSONResponse

from app.api.dependencies import get_container
from app.core.errors import FileTooLargeError
from app.schemas.documents import DocumentDeleteResponse, DocumentInfo, DocumentListResponse, DocumentUploadResponse
from app.services.container import Container

router = APIRouter(prefix="/documents", tags=["documents"])


def _read_upload(file: UploadFile, container: Container) -> tuple[str, bytes]:
    name = file.filename or "upload"
    limit = int(container.settings.max_upload_mb * 1024 * 1024)
    if file.size is not None and file.size > limit:  # reject before reading 100+ MB into memory
        raise FileTooLargeError(f"'{name}' exceeds the {container.settings.max_upload_mb:g} MB upload limit.")
    return name, file.file.read()


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document (PDF, TXT, Markdown, DOCX); indexed in the background",
    responses={201: {"description": "Indexed synchronously (wait=true)"}, 409: {"description": "Duplicate content"},
               413: {"description": "Too large"}, 415: {"description": "Unsupported type"},
               422: {"description": "Unparseable (wait=true only; otherwise reported as status=failed)"}},
)
def upload_document(
    file: UploadFile = File(...),
    wait: bool = Query(False, description="Block until indexing finishes (small files, scripts, tests)"),
    container: Container = Depends(get_container),
):
    """Returns immediately with `status=processing`; poll `GET /documents/{id}` for progress.

    Type, size and duplicates are validated before the response, so those errors are immediate.
    Parsing/embedding runs in a single background worker so large files never block the API.
    """
    # Sync handler: FastAPI runs it in a worker thread, so reading the upload never blocks the event loop.
    name, content = _read_upload(file, container)
    if wait:
        info = container.documents.ingest(name, content)
        container.metrics.record_document(ingested=1)
        return JSONResponse(
            DocumentUploadResponse(document=info, message=f"Indexed '{info.filename}' into {info.num_chunks} chunks.")
            .model_dump(mode="json"),
            status_code=status.HTTP_201_CREATED,
        )
    info = container.documents.submit(name, content)
    container.metrics.record_document(ingested=1)
    return DocumentUploadResponse(document=info, message=f"Indexing '{info.filename}' in the background.")


@router.get("", response_model=DocumentListResponse, summary="List documents with status and progress")
def list_documents(container: Container = Depends(get_container)):
    docs = container.documents.list_documents()
    return DocumentListResponse(documents=docs, total_documents=len(docs), total_chunks=sum(d.num_chunks for d in docs))


@router.get("/{document_id}", response_model=DocumentInfo, summary="Document status, progress and error")
def get_document(document_id: str, container: Container = Depends(get_container)):
    return container.documents.get(document_id)


@router.delete("/{document_id}", response_model=DocumentDeleteResponse,
               summary="Delete a document and its chunks (cancels indexing if still processing)")
def delete_document(document_id: str, container: Container = Depends(get_container)):
    removed = container.documents.delete(document_id)
    container.metrics.record_document(deleted=1)
    return DocumentDeleteResponse(id=document_id, chunks_removed=removed, message=f"Deleted document and {removed} chunks.")


legacy_router = APIRouter(tags=["legacy"])


@legacy_router.post("/ingest", response_model=DocumentUploadResponse, deprecated=True,
                    summary="Deprecated: use POST /documents (synchronous, v1 compatibility)")
def legacy_ingest(file: UploadFile = File(...), container: Container = Depends(get_container)):
    name, content = _read_upload(file, container)
    info = container.documents.ingest(name, content)
    container.metrics.record_document(ingested=1)
    return DocumentUploadResponse(document=info, message=f"Indexed '{info.filename}' into {info.num_chunks} chunks.")
