"""
Vector store management using numpy.

Handles embedding generation with sentence-transformers (all-MiniLM-L6-v2),
document chunking, indexing, similarity search, and index persistence.
(Using numpy instead of FAISS for maximum compatibility on all OSs)
"""

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "faiss_index")
EMBEDDINGS_PATH = os.path.join(INDEX_DIR, "embeddings.npy")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 50  # overlap between consecutive chunks
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # dimension for all-MiniLM-L6-v2


class VectorStore:
    """Manages a numpy index backed by sentence-transformer embeddings."""

    def __init__(self) -> None:
        self.model = SentenceTransformer(MODEL_NAME)
        self.chunks: list[str] = []
        self.embeddings: np.ndarray | None = None
        self._load_from_disk()
        
    @property
    def index(self):
        class MockIndex:
            def __init__(self, parent):
                self.parent = parent
            @property
            def ntotal(self):
                return len(self.parent.chunks)
        return MockIndex(self)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, text: str) -> int:
        """Chunk *text*, embed, and add to the index.

        Returns the number of new chunks added.
        """
        new_chunks = self._chunk_text(text)
        if not new_chunks:
            return 0

        new_embs = self.model.encode(new_chunks, normalize_embeddings=True)
        new_embs = np.array(new_embs, dtype="float32")

        if self.embeddings is None:
            self.embeddings = new_embs
        else:
            self.embeddings = np.vstack([self.embeddings, new_embs])

        self.chunks.extend(new_chunks)
        self._save_to_disk()
        return len(new_chunks)

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """Return the *top_k* most similar chunks for *query*."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        query_embedding = self.model.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding[0], dtype="float32")

        distances = np.dot(self.embeddings, query_embedding)
        k = min(top_k, len(self.chunks))
        top_indices = np.argsort(distances)[::-1][:k]

        results: list[str] = []
        for idx in top_indices:
            if 0 <= idx < len(self.chunks):
                results.append(self.chunks[idx])
        return results

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save_to_disk(self) -> None:
        os.makedirs(INDEX_DIR, exist_ok=True)
        if self.embeddings is not None:
            np.save(EMBEDDINGS_PATH, self.embeddings)
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)

    def _load_from_disk(self) -> None:
        if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(METADATA_PATH):
            self.embeddings = np.load(EMBEDDINGS_PATH)
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)

    # ------------------------------------------------------------------
    # Text chunking
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_text(text: str) -> list[str]:
        """Split *text* into overlapping chunks of ~CHUNK_SIZE characters."""
        text = text.strip()
        if not text:
            return []

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
vector_store = VectorStore()
