"""Dense retrieval over a FAISS index.

`IndexIDMap2(IndexFlatIP)`:
  * exact inner-product search (== cosine, vectors are normalized). At this
    corpus size (thousands of chunks) exact search is ~1 ms; approximate
    indexes (HNSW/IVF) only pay off at ~10^5-10^6 vectors.
  * `IDMap2` lets us key vectors by the SQLite chunk id and `remove_ids()`
    them when a document is deleted.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import faiss
import numpy as np


class DenseIndex:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._index: faiss.IndexIDMap2 | None = None
        self._lock = threading.RLock()
        if path is not None and path.exists():
            self._index = faiss.read_index(str(path))

    @property
    def dim(self) -> int | None:
        return self._index.d if self._index is not None else None

    @property
    def size(self) -> int:
        return int(self._index.ntotal) if self._index is not None else 0

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        if len(ids) == 0:
            return
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        with self._lock:
            if self._index is None or self._index.d != vectors.shape[1]:
                self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(vectors.shape[1]))
            self._index.add_with_ids(vectors, np.asarray(ids, dtype="int64"))

    def remove(self, ids: list[int]) -> int:
        if not ids or self._index is None:
            return 0
        with self._lock:
            return int(self._index.remove_ids(np.asarray(ids, dtype="int64")))

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return `(chunk_id, cosine_similarity)` pairs, best first."""
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return []
            k = min(k, self._index.ntotal)
            q = np.ascontiguousarray(query_vector.reshape(1, -1), dtype="float32")
            scores, ids = self._index.search(q, k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def ids(self) -> set[int]:
        with self._lock:
            if self._index is None:
                return set()
            return set(faiss.vector_to_array(self._index.id_map).tolist())

    def reset(self) -> None:
        with self._lock:
            self._index = None

    def save(self) -> None:
        """Atomic write (tmp file + rename) so a crash never leaves a torn index."""
        if self._path is None:
            return
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            if self._index is None:
                self._path.unlink(missing_ok=True)
                return
            tmp = self._path.with_suffix(".tmp")
            faiss.write_index(self._index, str(tmp))
            os.replace(tmp, self._path)
