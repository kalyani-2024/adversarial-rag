"""Document ingestion and lifecycle.

    bytes -> validate -> hash/de-dup -> register (status=processing)
          -> parse -> clean -> chunk -> embed in batches (progress)
          -> SQLite chunks (source of truth) -> FAISS vectors -> status=ready -> notify (BM25)

Two entry points share one pipeline:
  * `submit()`: returns immediately; a single background worker does the
    heavy lifting. Used by the API so a 100 MB PDF never blocks a request, and
    one worker means indexing never competes with itself for CPU.
  * `ingest()`: synchronous; used for the seed corpus, the eval harness and tests.

Failure handling: vectors are computed before any chunk is written, so an
embedding failure leaves no partial chunks. If the FAISS write fails after
the SQLite insert, the chunks are deleted again (compensating action).
Background failures are kept as `status=failed` with the error so the user
can see what happened; synchronous failures remove the document entirely.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from app.core.config import Settings
from app.core.errors import (
    AppError,
    DocumentNotFoundError,
    DocumentParseError,
    DuplicateDocumentError,
    FileTooLargeError,
)
from app.ingestion.chunking import chunk_pages
from app.ingestion.cleaning import clean_text
from app.ingestion.context import INDEX_VERSION, front_matter_context, is_front_matter
from app.ingestion.parsers import PageText, detect_file_type, parse_document
from app.retrieval.dense import DenseIndex
from app.retrieval.embeddings import Embedder
from app.schemas.documents import DocumentInfo
from app.services.priority import QueryActivity
from app.storage.document_store import DocumentStore

logger = logging.getLogger(__name__)

EMBED_BATCH = 256  # chunks per embedding call: bounds memory and gives progress updates


class _Cancelled(Exception):
    """The document was deleted while it was being processed."""


class DocumentService:
    def __init__(self, settings: Settings, store: DocumentStore, embedder: Embedder, dense_index: DenseIndex,
                 activity: QueryActivity | None = None) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.dense_index = dense_index
        self._write_lock = threading.RLock()  # re-entrant: rollback runs inside the write section
        self._listeners: list[Callable[[], None]] = []
        self._executor: ThreadPoolExecutor | None = None
        self._progress: dict[str, tuple[str, float | None]] = {}  # doc id -> (stage, fraction)
        self._cancelled: set[str] = set()
        self.activity = activity or QueryActivity()
        if self.store.get_meta("index_version") is None and self.store.count_chunks() == 0:
            self.store.set_meta("index_version", str(INDEX_VERSION))  # new store: nothing to upgrade

    def add_listener(self, callback: Callable[[], None]) -> None:
        """Register a callback fired after the corpus changes (e.g. BM25 update)."""
        self._listeners.append(callback)

    def _notify(self) -> None:
        for callback in self._listeners:
            callback()

    # --- entry points ------------------------------------------------------------
    def ingest(self, filename: str, content: bytes) -> DocumentInfo:
        """Synchronously index a document. On failure nothing is left behind and the error is raised."""
        info, file_type = self._register(filename, content)
        try:
            self._process(info.id, info.filename, file_type, content)
        except Exception:
            self._discard(info.id)
            raise
        return self._with_progress(self.store.get_document(info.id))

    def submit(self, filename: str, content: bytes) -> DocumentInfo:
        """Validate and register now; parse/embed/index in the background worker."""
        info, file_type = self._register(filename, content)
        self._progress[info.id] = ("queued", None)
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")
        self._executor.submit(self._process_in_background, info.id, info.filename, file_type, content)
        return self._with_progress(info)

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)

    # --- pipeline ------------------------------------------------------------------
    def _register(self, filename: str, content: bytes) -> tuple[DocumentInfo, str]:
        filename = Path(filename).name  # never trust client-supplied paths
        file_type = detect_file_type(filename)
        max_bytes = int(self.settings.max_upload_mb * 1024 * 1024)
        if len(content) > max_bytes:
            raise FileTooLargeError(f"'{filename}' exceeds the {self.settings.max_upload_mb:g} MB upload limit.")
        if not content:
            raise DocumentParseError(f"'{filename}' is empty.")

        content_hash = hashlib.sha256(content).hexdigest()
        existing = self.store.get_by_hash(content_hash)
        if existing is not None and existing.status == "failed":
            self._discard(existing.id)  # a retry of a failed upload is allowed
        elif existing is not None:
            state = "is still being indexed" if existing.status == "processing" else "is already indexed"
            raise DuplicateDocumentError(f"'{filename}' {state} as '{existing.filename}'.",
                                         details={"document_id": existing.id})
        try:
            info = self.store.create_document(document_id=uuid.uuid4().hex[:12], filename=filename,
                                              file_type=file_type, content_hash=content_hash, size_bytes=len(content))
        except sqlite3.IntegrityError as exc:  # lost a race with a concurrent upload of the same bytes
            raise DuplicateDocumentError(f"'{filename}' is already being indexed.") from exc
        return info, file_type

    def _process_in_background(self, document_id: str, filename: str, file_type: str, content: bytes) -> None:
        try:
            self._process(document_id, filename, file_type, content)
        except _Cancelled:
            logger.info("ingestion cancelled", extra={"document_id": document_id})
        except Exception as exc:
            message = exc.message if isinstance(exc, AppError) else f"Indexing failed: {type(exc).__name__}: {exc}"
            self._rollback_chunks(document_id)
            self.store.set_status(document_id, "failed", message)
            logger.warning("ingestion failed", extra={"document_id": document_id, "doc_name": filename, "error": message})
        finally:
            self._progress.pop(document_id, None)
            self._cancelled.discard(document_id)

    def _check_cancelled(self, document_id: str) -> None:
        if document_id in self._cancelled:
            raise _Cancelled()

    def _process(self, document_id: str, filename: str, file_type: str, content: bytes) -> None:
        self._progress[document_id] = ("parsing", None)
        parsed = parse_document(filename, content)
        pages = [PageText(clean_text(p.text, file_type=file_type), p.page) for p in parsed.pages]
        chunks = chunk_pages(pages, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            hint = " (scanned PDFs need OCR, which is not supported)" if file_type == "pdf" else ""
            raise DocumentParseError(f"No extractable text found in '{filename}'{hint}.")

        for chunk in chunks:  # describe front-matter chunks so "who are the authors?" can find them
            if is_front_matter(chunk.chunk_index, chunk.page):
                chunk.context = front_matter_context(filename)
        texts = [c.index_text for c in chunks]
        batches = []
        for start in range(0, len(texts), EMBED_BATCH):
            self._check_cancelled(document_id)
            self.activity.wait_until_idle(max_wait_s=10.0)  # let live queries use the CPU first
            self._progress[document_id] = ("embedding", start / len(texts))
            batches.append(self.embedder.embed(texts[start : start + EMBED_BATCH]))
        vectors = np.vstack(batches)

        self._progress[document_id] = ("indexing", 1.0)
        with self._write_lock:
            self._check_cancelled(document_id)
            records = self.store.add_chunks(document_id, chunks, parsed.num_pages)
            try:
                self.dense_index.add([r.id for r in records], vectors)
                self.dense_index.save()
            except Exception:
                self._rollback_chunks(document_id)
                raise
            self.store.set_status(document_id, "ready")
        self._notify()
        logger.info("document ingested", extra={"document_id": document_id, "doc_name": filename,
                                                "chunks": len(records), "file_type": file_type})

    def _rollback_chunks(self, document_id: str) -> None:
        ids = [c.id for c in self.store.get_document_chunks(document_id)]
        if ids:
            with self._write_lock:
                self.dense_index.remove(ids)
                self.dense_index.save()
                self.store.delete_chunks(document_id)

    def _discard(self, document_id: str) -> None:
        with self._write_lock:
            chunk_ids = self.store.delete_document(document_id)
            if chunk_ids:
                self.dense_index.remove(chunk_ids)
                self.dense_index.save()

    # --- lifecycle -----------------------------------------------------------------
    def _with_progress(self, info: DocumentInfo | None) -> DocumentInfo:
        assert info is not None
        if info.status != "processing":  # progress is only meaningful while processing
            return info
        stage, frac = self._progress.get(info.id, ("queued", None))
        return info.model_copy(update={"stage": stage, "progress": frac})

    def get(self, document_id: str) -> DocumentInfo:
        info = self.store.get_document(document_id)
        if info is None:
            raise DocumentNotFoundError(f"Document '{document_id}' not found.")
        return self._with_progress(info)

    def list_documents(self) -> list[DocumentInfo]:
        return [self._with_progress(d) for d in self.store.list_documents()]

    def delete(self, document_id: str) -> int:
        info = self.store.get_document(document_id)
        if info is None:
            raise DocumentNotFoundError(f"Document '{document_id}' not found.")
        if info.status == "processing":
            self._cancelled.add(document_id)  # the worker stops at its next checkpoint
        with self._write_lock:
            chunk_ids = self.store.delete_document(document_id)
            self.dense_index.remove(chunk_ids)
            self.dense_index.save()
        self._notify()
        logger.info("document deleted", extra={"document_id": document_id, "chunks": len(chunk_ids)})
        return len(chunk_ids)

    def recover_interrupted(self) -> int:
        """Startup: documents left 'processing' by a restart are marked failed (their bytes are gone)."""
        return self.store.fail_interrupted()

    def sync_index(self) -> int:
        """Make FAISS consistent with SQLite (startup self-heal).

        Handles a missing/corrupt index file, a crash between the two writes, or
        a changed embedding model (dimension mismatch). Returns #vectors rebuilt.
        """
        upgraded = self._upgrade_index_version()
        chunks = self.store.all_chunks()
        store_ids = {c.id for c in chunks}
        if not upgraded and store_ids == self.dense_index.ids():
            probe = self.embedder.embed(["dimension probe"]) if chunks else None
            if probe is None or probe.shape[1] == self.dense_index.dim:
                return 0
        logger.warning("dense index out of sync with document store; rebuilding", extra={"chunks": len(chunks)})
        with self._write_lock:
            self.dense_index.reset()
            for start in range(0, len(chunks), EMBED_BATCH):
                batch = chunks[start : start + EMBED_BATCH]
                self.dense_index.add([c.id for c in batch], self.embedder.embed([c.index_text for c in batch]))
            self.dense_index.save()
        return len(chunks)

    def _upgrade_index_version(self) -> bool:
        """Bring a store built by an older version up to date. Returns True if vectors must be rebuilt.

        v2 added index-time context for front-matter chunks. The context is deterministic, so it can
        be back-filled for existing chunks; their vectors are then re-embedded by `sync_index`.
        """
        current = self.store.get_meta("index_version")
        if current == str(INDEX_VERSION):
            return False
        chunks = self.store.all_chunks()
        updates = [(front_matter_context(c.document_name), c.id) for c in chunks
                   if is_front_matter(c.chunk_index, c.page) and not c.context]
        if updates:
            self.store.set_chunk_contexts(updates)
        self.store.set_meta("index_version", str(INDEX_VERSION))
        if chunks:
            logger.warning("index upgraded", extra={"from_version": current or "1", "to_version": INDEX_VERSION,
                                                    "chunks_contextualized": len(updates)})
        return bool(chunks)

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
