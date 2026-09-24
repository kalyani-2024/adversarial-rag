"""Hybrid retrieval: dense + BM25 -> RRF -> cross-encoder rerank -> top-k.

Every stage degrades gracefully:
  * embedding failure  -> BM25-only retrieval (dense_error recorded)
  * reranker failure   -> fusion order (reranker_error recorded)
"""

from __future__ import annotations

import logging
import time

from app.core.config import Settings
from app.core.errors import EmbeddingError
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import DenseIndex
from app.retrieval.embeddings import Embedder
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import Reranker
from app.schemas.retrieval import RetrievalResult, RetrievedChunk
from app.storage.document_store import DocumentStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        store: DocumentStore,
        embedder: Embedder,
        dense_index: DenseIndex,
        bm25_index: BM25Index,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.dense_index = dense_index
        self.bm25_index = bm25_index
        self.reranker = reranker

    def refresh_sparse_index(self) -> None:
        """Bring BM25 in line with the store, touching only documents that were added or removed."""
        stored = {d.id for d in self.store.list_documents() if d.num_chunks}
        indexed = self.bm25_index.document_ids()
        for doc_id in indexed - stored:
            self.bm25_index.remove_document(doc_id)
        for doc_id in stored - indexed:
            self.bm25_index.add_document(doc_id, self.store.get_document_chunks(doc_id))

    def retrieve(self, query: str, *, top_k: int | None = None, use_reranker: bool | None = None) -> RetrievalResult:
        s = self.settings
        top_k = top_k or s.final_top_k
        use_reranker = s.reranker_enabled if use_reranker is None else use_reranker
        result = RetrievalResult(chunks=[])

        # --- stage 1: candidate generation --------------------------------------
        t0 = time.perf_counter()
        dense_hits: list[tuple[int, float]] = []
        try:
            query_vec = self.embedder.embed([query])[0]
            dense_hits = self.dense_index.search(query_vec, s.dense_top_k)
        except EmbeddingError as exc:
            result.dense_error = exc.message
            logger.warning("dense retrieval failed; falling back to BM25 only", extra={"error": exc.message})
        bm25_hits = self.bm25_index.search(query, s.bm25_top_k)

        fused = reciprocal_rank_fusion(
            {"dense": [i for i, _ in dense_hits], "bm25": [i for i, _ in bm25_hits]}, k=s.rrf_k
        )
        result.dense_candidates, result.bm25_candidates, result.fused_candidates = len(dense_hits), len(bm25_hits), len(fused)
        result.retrieval_ms = (time.perf_counter() - t0) * 1000
        if not fused:
            return result

        shortlist = fused[: max(s.rerank_candidates, top_k)]
        records = self.store.get_chunks([f.id for f in shortlist])
        dense_scores, bm25_scores = dict(dense_hits), dict(bm25_hits)
        candidates: list[RetrievedChunk] = []
        for fusion_rank, item in enumerate(shortlist, start=1):
            rec = records.get(item.id)
            if rec is None:  # deleted between search and fetch
                continue
            candidates.append(
                RetrievedChunk(
                    chunk_id=rec.chunk_id,
                    document_id=rec.document_id,
                    document_name=rec.document_name,
                    chunk_index=rec.chunk_index,
                    page=rec.page,
                    text=rec.text,
                    dense_score=dense_scores.get(item.id),
                    dense_rank=item.ranks.get("dense"),
                    bm25_score=bm25_scores.get(item.id),
                    bm25_rank=item.ranks.get("bm25"),
                    rrf_score=item.score,
                    fusion_rank=fusion_rank,
                )
            )

        # --- stage 2: rerank -------------------------------------------------
        if use_reranker and self.reranker is not None and candidates:
            t1 = time.perf_counter()
            try:
                scores = self.reranker.score(query, [c.text for c in candidates])
                for c, sc in zip(candidates, scores):
                    c.rerank_score = sc
                candidates.sort(key=lambda c: c.rerank_score, reverse=True)
                result.reranker_used = True
            except Exception as exc:
                result.reranker_error = str(exc)
                logger.warning("reranker failed; using fusion order", extra={"error": str(exc)})
            result.rerank_ms = (time.perf_counter() - t1) * 1000

        final = candidates[:top_k]
        for rank, c in enumerate(final, start=1):
            c.final_rank = rank
        result.chunks = final
        return result


def evidence_check(result: RetrievalResult, settings: Settings) -> tuple[bool, str]:
    """Decide whether retrieval found enough evidence to attempt an answer.

    This deterministic gate runs *before* any LLM call: if nothing relevant
    was retrieved, the cheapest and safest answer is an explicit abstention.
    """
    if not result.chunks:
        return False, "no candidates retrieved"
    if result.reranker_used:
        best = max(c.rerank_score for c in result.chunks if c.rerank_score is not None)
        if best < settings.min_rerank_score:
            return False, f"best rerank score {best:.2f} < {settings.min_rerank_score}"
        return True, f"best rerank score {best:.2f}"
    dense = [c.dense_score for c in result.chunks if c.dense_score is not None]
    if dense:
        best = max(dense)
        if best < settings.min_dense_score:
            return False, f"best dense score {best:.2f} < {settings.min_dense_score}"
        return True, f"best dense score {best:.2f}"
    # Dense unavailable: fall back to "BM25 found lexical overlap".
    return (True, "bm25-only evidence") if any(c.bm25_score for c in result.chunks) else (False, "no lexical overlap")
