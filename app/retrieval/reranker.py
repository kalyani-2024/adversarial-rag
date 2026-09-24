"""Second-stage reranking.

Bi-encoders (the dense retriever) embed query and chunk independently, which
is fast but coarse. A cross-encoder reads (query, chunk) *together* and scores
relevance with full attention, which is much more precise but costs one
forward pass per candidate, so we only run it on the fused shortlist.

`Reranker` is a protocol so a hosted reranker (Cohere, Voyage, Jina...) can
replace the local model by implementing `score()`.
"""

from __future__ import annotations

import logging
import threading
from typing import Protocol

logger = logging.getLogger(__name__)


class RerankerError(RuntimeError):
    pass


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Higher = more relevant. Scale is model-specific."""
        ...


class CrossEncoderReranker:
    """`cross-encoder/ms-marco-MiniLM-L-6-v2`: ~22M params, ~10-40 ms per 20 pairs on CPU."""

    def __init__(self, model_name: str, max_length: int = 512) -> None:
        self.name = model_name
        self.max_length = max_length
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import CrossEncoder

                    logger.info("loading reranker", extra={"model": self.name})
                    self._model = CrossEncoder(self.name, max_length=self.max_length, device="cpu")
                except Exception as exc:
                    raise RerankerError(f"Could not load reranker '{self.name}': {exc}") from exc
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        model = self._load()
        try:
            # Raw logits (no sigmoid) so the evidence threshold is in logit space.
            scores = model.predict([(query, t) for t in texts], show_progress_bar=False, activation_fn=None)
        except TypeError:  # older sentence-transformers: no activation_fn kwarg
            scores = model.predict([(query, t) for t in texts], show_progress_bar=False)
        except Exception as exc:
            raise RerankerError(f"Reranking failed: {exc}") from exc
        return [float(s) for s in scores]
