"""Composition root: builds and wires every component exactly once.

Keeping construction here (instead of module-level singletons) is what lets
tests swap in fakes, and makes startup order explicit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.core.config import Settings
from app.core.llm import GroqLLM, LLMClient
from app.observability.metrics import MetricsRegistry
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import DenseIndex
from app.retrieval.embeddings import Embedder, SentenceTransformerEmbedder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker, Reranker
from app.services.document_service import DocumentService
from app.services.priority import QueryActivity
from app.services.query_service import QueryService
from app.storage.document_store import DocumentStore

logger = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    store: DocumentStore
    documents: DocumentService
    retriever: HybridRetriever
    queries: QueryService
    metrics: MetricsRegistry

    def startup(self) -> None:
        """Warm models, self-heal the vector index, build BM25, ingest seed docs."""
        if self.settings.warmup_models:
            self._warmup()
        interrupted = self.documents.recover_interrupted()
        rebuilt = self.documents.sync_index()
        self.retriever.refresh_sparse_index()
        seeded = []
        if self.settings.seed_dir and self.settings.seed_dir.is_dir():
            seeded = self.documents.ingest_directory(self.settings.seed_dir)
        logger.info(
            "startup complete",
            extra={"chunks": self.store.count_chunks(), "vectors_rebuilt": rebuilt, "seeded_documents": len(seeded),
                   "interrupted_uploads_marked_failed": interrupted},
        )

    def _warmup(self) -> None:
        try:
            self.retriever.embedder.embed(["warmup"])
            if self.retriever.reranker is not None:
                self.retriever.reranker.score("warmup", ["warmup"])
        except Exception as exc:  # degraded mode is handled per request
            logger.warning("model warmup failed", extra={"error": str(exc)})

    def shutdown(self) -> None:
        self.documents.shutdown()
        self.store.close()


def build_container(
    settings: Settings,
    *,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
) -> Container:
    store = DocumentStore(settings.db_path)
    embedder = embedder or SentenceTransformerEmbedder(settings.embedding_model)
    if reranker is None and settings.reranker_enabled:
        reranker = CrossEncoderReranker(settings.reranker_model, max_length=settings.reranker_max_length)
    dense = DenseIndex(settings.index_dir / "dense.faiss")
    metrics = MetricsRegistry()
    activity = QueryActivity()  # shared: queries take priority over background indexing

    documents = DocumentService(settings, store, embedder, dense, activity)
    retriever = HybridRetriever(settings, store, embedder, dense, BM25Index(), reranker)
    documents.add_listener(retriever.refresh_sparse_index)
    queries = QueryService(settings, llm or GroqLLM(settings), retriever, metrics, activity)
    return Container(settings, store, documents, retriever, queries, metrics)
