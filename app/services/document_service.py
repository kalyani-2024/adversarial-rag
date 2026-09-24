"""Document ingestion and lifecycle.

    bytes -> validate -> hash/de-dup -> parse -> clean -> chunk -> embed
          -> SQLite (source of truth) -> FAISS (vectors) -> notify listeners (BM25)

Write ordering: vectors are computed *before* the SQLite transaction, so an
embedding failure leaves no partial state. If the FAISS write fails after the
SQLite commit, we compensate by deleting the document again.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import uuid
from collections.abc import Callable
from pathlib import Path

from app.core.config import Settings
from app.core.errors import (
    DocumentNotFoundError,
    DocumentParseError,
    DuplicateDocumentError,
    FileTooLargeError,
)
from app.ingestion.chunking import chunk_pages
from app.ingestion.cleaning import clean_text
from app.ingestion.parsers import PageText, detect_file_type, parse_document
from app.retrieval.dense import DenseIndex
from app.retrieval.embeddings import Embedder
from app.schemas.documents import DocumentInfo
from app.storage.document_store import DocumentStore

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(self, settings: Settings, store: DocumentStore, embedder: Embedder, dense_index: DenseIndex) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.dense_index = dense_index
        self._write_lock = threading.Lock()
        self._listeners: list[Callable[[], None]] = []

    def add_listener(self, callback: Callable[[], None]) -> None:
        """Register a callback fired after the corpus changes (e.g. BM25 rebuild)."""
        self._listeners.append(callback)

    def _notify(self) -> None:
        for callback in self._listeners:
            callback()

    # --- ingestion -----------------------------------------------------------
    def ingest(self, filename: str, content: bytes) -> DocumentInfo:
        filename = Path(filename).name  # never trust client-supplied paths
        file_type = detect_file_type(filename)
        max_bytes = int(self.settings.max_upload_mb * 1024 * 1024)
        if len(content) > max_bytes:
            raise FileTooLargeError(f"'{filename}' exceeds the {self.settings.max_upload_mb:g} MB upload limit.")

        content_hash = hashlib.sha256(content).hexdigest()
        existing = self.store.get_by_hash(content_hash)
        if existing is not None:
            raise DuplicateDocumentError(
                f"'{filename}' is already indexed as '{existing.filename}'.",
                details={"document_id": existing.id},
            )

        parsed = parse_document(filename, content)
        pages = [PageText(clean_text(p.text, file_type=file_type), p.page) for p in parsed.pages]
        chunks = chunk_pages(pages, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            hint = " (scanned PDFs need OCR, which is not supported)" if file_type == "pdf" else ""
            raise DocumentParseError(f"No extractable text found in '{filename}'{hint}.")

        vectors = self.embedder.embed([c.text for c in chunks])

        document_id = uuid.uuid4().hex[:12]
        with self._write_lock:
            # Re-check under the lock: two concurrent uploads of the same file.
            if self.store.get_by_hash(content_hash) is not None:
                raise DuplicateDocumentError(f"'{filename}' is already indexed.")
            records = self.store.add_document(
                document_id=document_id,
                filename=filename,
                file_type=file_type,
                content_hash=content_hash,
                size_bytes=len(content),
                num_pages=parsed.num_pages,
                chunks=chunks,
            )
            try:
                self.dense_index.add([r.id for r in records], vectors)
                self.dense_index.save()
            except Exception:
                self.store.delete_document(document_id)
                self.dense_index.remove([r.id for r in records])
                raise
        self._notify()

        info = self.store.get_document(document_id)
        assert info is not None
        logger.info(
            "document ingested",
            extra={"document_id": document_id, "doc_name": filename, "chunks": len(records), "file_type": file_type},
        )
        return info

    # --- lifecycle -----------------------------------------------------------
    def list_documents(self) -> list[DocumentInfo]:
        return self.store.list_documents()

    def delete(self, document_id: str) -> int:
        with self._write_lock:
            if self.store.get_document(document_id) is None:
                raise DocumentNotFoundError(f"Document '{document_id}' not found.")
            chunk_ids = self.store.delete_document(document_id)
            self.dense_index.remove(chunk_ids)
            self.dense_index.save()
        self._notify()
        logger.info("document deleted", extra={"document_id": document_id, "chunks": len(chunk_ids)})
        return len(chunk_ids)

    def sync_index(self) -> int:
        """Make FAISS consistent with SQLite (startup self-heal).

        Handles a missing/corrupt index file, a crash between the two writes, or
        a changed embedding model (dimension mismatch). Returns #vectors rebuilt.
        """
        chunks = self.store.all_chunks()
        store_ids = {c.id for c in chunks}
        if store_ids == self.dense_index.ids():
            probe = self.embedder.embed(["dimension probe"]) if chunks else None
            if probe is None or probe.shape[1] == self.dense_index.dim:
                return 0
        logger.warning("dense index out of sync with document store; rebuilding", extra={"chunks": len(chunks)})
        with self._write_lock:
            self.dense_index.reset()
            if chunks:
                self.dense_index.add([c.id for c in chunks], self.embedder.embed([c.text for c in chunks]))
            self.dense_index.save()
        return len(chunks)

    def ingest_directory(self, directory: Path) -> list[DocumentInfo]:
        """Ingest every supported file in `directory`; duplicates are skipped."""
        ingested: list[DocumentInfo] = []
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            try:
                ingested.append(self.ingest(path.name, path.read_bytes()))
            except DuplicateDocumentError:
                continue
            except Exception as exc:
                logger.warning("seed ingest failed", extra={"doc_name": path.name, "error": str(exc)})
        return ingested
