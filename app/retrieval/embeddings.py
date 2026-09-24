"""Embedding model abstraction."""

from __future__ import annotations

import logging
import threading
from typing import Protocol

import numpy as np

from app.core.errors import EmbeddingError

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized float32 vectors of shape (len(texts), dim)."""
        ...


class SentenceTransformerEmbedder:
    """Local SentenceTransformers model, loaded lazily on first use."""

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer

                    logger.info("loading embedding model", extra={"model": self.model_name})
                    self._model = SentenceTransformer(self.model_name, device="cpu")
                except Exception as exc:
                    raise EmbeddingError(f"Could not load embedding model '{self.model_name}': {exc}") from exc
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype="float32")
        model = self._load()
        try:
            vecs = model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,  # cosine similarity == inner product
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed: {exc}") from exc
        return np.asarray(vecs, dtype="float32")
