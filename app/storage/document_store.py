"""SQLite-backed source of truth for documents and chunks.

Why SQLite: it is transactional, zero-ops, ships with Python, and keeps chunk
text + metadata queryable. FAISS stores only vectors keyed by the same integer
chunk id, so the vector index can always be rebuilt from this store.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.ingestion.chunking import TextChunk
from app.schemas.documents import ChunkRecord, DocumentInfo

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    file_type     TEXT NOT NULL,
    content_hash  TEXT NOT NULL UNIQUE,
    size_bytes    INTEGER NOT NULL,
    num_pages     INTEGER,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id   TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL,
    page          INTEGER,
    text          TEXT NOT NULL,
    UNIQUE(document_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
"""

_CHUNK_SELECT = """
SELECT c.id, c.document_id, d.filename, c.chunk_index, c.page, c.text
FROM chunks c JOIN documents d ON d.id = c.document_id
"""


def _row_to_chunk(row: sqlite3.Row) -> ChunkRecord:
    return ChunkRecord(
        id=row[0],
        chunk_id=f"{row[1]}:{row[3]}",
        document_id=row[1],
        document_name=row[2],
        chunk_index=row[3],
        page=row[4],
        text=row[5],
    )


class DocumentStore:
    def __init__(self, db_path: Path | str) -> None:
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._lock = threading.RLock()

    def _migrate(self) -> None:
        """Add columns introduced after v2.0 to existing databases (idempotent)."""
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(documents)")}
        with self._conn:
            if "status" not in cols:  # processing | ready | failed
                self._conn.execute("ALTER TABLE documents ADD COLUMN status TEXT NOT NULL DEFAULT 'ready'")
            if "error" not in cols:
                self._conn.execute("ALTER TABLE documents ADD COLUMN error TEXT")

    # --- writes ----------------------------------------------------------------
    def create_document(
        self, *, document_id: str, filename: str, file_type: str, content_hash: str, size_bytes: int,
        status: str = "processing",
    ) -> DocumentInfo:
        """Register a document before its (possibly slow) processing starts.

        The UNIQUE content_hash makes this the de-duplication point: a second
        upload of the same bytes fails here even while the first is processing.
        """
        created = datetime.now(timezone.utc).isoformat()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO documents (id, filename, file_type, content_hash, size_bytes, num_pages, created_at, status)"
                " VALUES (?, ?, ?, ?, ?, NULL, ?, ?)",
                (document_id, filename, file_type, content_hash, size_bytes, created, status),
            )
        info = self.get_document(document_id)
        assert info is not None
        return info

    def add_chunks(self, document_id: str, chunks: list[TextChunk], num_pages: int | None) -> list[ChunkRecord]:
        """Insert all chunks of a document in one transaction; returns them with ids."""
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT INTO chunks (document_id, chunk_index, page, text) VALUES (?, ?, ?, ?)",
                [(document_id, c.chunk_index, c.page, c.text) for c in chunks],
            )
            self._conn.execute("UPDATE documents SET num_pages = ? WHERE id = ?", (num_pages, document_id))
        return self.get_document_chunks(document_id)

    def set_status(self, document_id: str, status: str, error: str | None = None) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE documents SET status = ?, error = ? WHERE id = ?", (status, error, document_id))

    def fail_interrupted(self) -> int:
        """Mark documents left 'processing' by a crash/restart as failed. Returns how many."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE documents SET status = 'failed', error = 'Interrupted by a server restart; please re-upload.'"
                " WHERE status = 'processing'"
            )
            self._conn.execute(
                "DELETE FROM chunks WHERE document_id IN (SELECT id FROM documents WHERE status = 'failed')"
            )
        return cur.rowcount

    def delete_chunks(self, document_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))

    def delete_document(self, document_id: str) -> list[int]:
        """Delete a document (chunks cascade). Returns the removed chunk ids."""
        with self._lock, self._conn:
            ids = [r[0] for r in self._conn.execute("SELECT id FROM chunks WHERE document_id = ?", (document_id,))]
            self._conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        return ids

    # --- reads -----------------------------------------------------------------
    def get_document(self, document_id: str) -> DocumentInfo | None:
        rows = self._list_documents("WHERE d.id = ?", (document_id,))
        return rows[0] if rows else None

    def get_by_hash(self, content_hash: str) -> DocumentInfo | None:
        rows = self._list_documents("WHERE d.content_hash = ?", (content_hash,))
        return rows[0] if rows else None

    def list_documents(self) -> list[DocumentInfo]:
        return self._list_documents()

    def _list_documents(self, where: str = "", params: tuple = ()) -> list[DocumentInfo]:
        sql = f"""
            SELECT d.id, d.filename, d.file_type, d.content_hash, d.size_bytes, d.num_pages,
                   d.created_at, COUNT(c.id), d.status, d.error
            FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
            {where}
            GROUP BY d.id ORDER BY d.created_at DESC
        """
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [
            DocumentInfo(
                id=r[0], filename=r[1], file_type=r[2], content_hash=r[3], size_bytes=r[4],
                num_pages=r[5], created_at=datetime.fromisoformat(r[6]), num_chunks=r[7],
                status=r[8], error=r[9],
            )
            for r in rows
        ]

    def get_document_chunks(self, document_id: str) -> list[ChunkRecord]:
        with self._lock:
            rows = self._conn.execute(_CHUNK_SELECT + " WHERE c.document_id = ? ORDER BY c.chunk_index", (document_id,)).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def get_chunks(self, ids: list[int]) -> dict[int, ChunkRecord]:
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        with self._lock:
            rows = self._conn.execute(_CHUNK_SELECT + f" WHERE c.id IN ({placeholders})", ids).fetchall()
        return {r[0]: _row_to_chunk(r) for r in rows}

    def all_chunks(self) -> list[ChunkRecord]:
        with self._lock:
            rows = self._conn.execute(_CHUNK_SELECT + " ORDER BY c.id").fetchall()
        return [_row_to_chunk(r) for r in rows]

    def count_chunks(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def close(self) -> None:
        self._conn.close()
