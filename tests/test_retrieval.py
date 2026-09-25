import pytest

from app.core.errors import EmbeddingError
from app.retrieval.bm25 import BM25Index, tokenize
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.hybrid import HybridRetriever, evidence_check
from app.schemas.documents import ChunkRecord
from app.schemas.retrieval import RetrievalResult, RetrievedChunk

DOCS = {
    "ml.txt": "The Transformer achieves ROC-AUC 0.959 on the held-out split.\n\n"
    "Isolation Forest reaches ROC-AUC 0.768 and is the weakest baseline.",
    "cooking.txt": "Sourdough bread needs a starter, flour, water and salt.\n\n"
    "Bake at 250 degrees with steam for a crisp crust.",
}


def _chunk(i: int, text: str) -> ChunkRecord:
    return ChunkRecord(id=i, chunk_id=f"d:{i}", document_id="d", document_name="d.txt", chunk_index=i, text=text)


# --- BM25 ----------------------------------------------------------------------------
def test_tokenize_keeps_compounds_and_parts():
    toks = tokenize("The ROC-AUC is 0.959!")
    assert "roc-auc" in toks and "roc" in toks and "auc" in toks and "0.959" in toks
    assert "the" not in toks and "is" not in toks


def test_bm25_ranks_exact_term_first():
    idx = BM25Index()
    idx.build([_chunk(1, "cats and dogs"), _chunk(2, "the metric was 0.959 exactly"), _chunk(3, "dogs are loyal")])
    hits = idx.search("which value is 0.959", k=3)
    assert hits[0][0] == 2
    assert all(score > 0 for _, score in hits)


def test_bm25_no_overlap_returns_nothing():
    idx = BM25Index()
    idx.build([_chunk(1, "cats and dogs"), _chunk(2, "birds fly")])
    assert idx.search("quantum entanglement", k=5) == []


def test_bm25_empty_index():
    idx = BM25Index()
    idx.build([])
    assert idx.search("anything", k=5) == []


# --- RRF ----------------------------------------------------------------------------------
def test_rrf_formula():
    fused = reciprocal_rank_fusion({"dense": [1, 2], "bm25": [2, 3]}, k=60)
    by_id = {f.id: f for f in fused}
    assert by_id[2].score == pytest.approx(1 / 62 + 1 / 61)
    assert by_id[1].score == pytest.approx(1 / 61)
    assert by_id[3].score == pytest.approx(1 / 62)
    assert fused[0].id == 2  # agreement between retrievers wins
    assert by_id[2].ranks == {"dense": 2, "bm25": 1}


def test_rrf_deterministic_tie_break():
    fused = reciprocal_rank_fusion({"dense": [5], "bm25": [4]}, k=60)
    assert [f.id for f in fused] == [4, 5]


def test_rrf_ignores_duplicates_and_empty():
    fused = reciprocal_rank_fusion({"dense": [1, 1], "bm25": []}, k=60)
    assert len(fused) == 1 and fused[0].score == pytest.approx(1 / 61)
    assert reciprocal_rank_fusion({"dense": [], "bm25": []}) == []


def test_rrf_invalid_k():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"a": [1]}, k=0)


# --- Hybrid retriever -------------------------------------------------------------------------
class KeywordReranker:
    name = "keyword"

    def __init__(self, keyword: str) -> None:
        self.keyword = keyword

    def score(self, query, texts):
        return [5.0 if self.keyword in t else -8.0 for t in texts]


class BrokenReranker:
    name = "broken"

    def score(self, query, texts):
        raise RuntimeError("reranker OOM")


@pytest.fixture
def retriever(settings, document_service):
    settings.reranker_enabled = True  # no reranker attached unless a test sets one
    for name, text in DOCS.items():
        document_service.ingest(name, text.encode())
    r = HybridRetriever(
        settings, document_service.store, document_service.embedder, document_service.dense_index, BM25Index()
    )
    document_service.add_listener(r.refresh_sparse_index)
    r.refresh_sparse_index()
    return r


def test_hybrid_returns_scored_metadata(retriever):
    res = retriever.retrieve("ROC-AUC of the Transformer", top_k=2)
    top = res.chunks[0]
    assert top.document_name == "ml.txt"
    assert top.final_rank == 1 and top.fusion_rank == 1
    assert top.rrf_score and top.bm25_score and top.dense_score is not None
    assert res.dense_candidates > 0 and res.bm25_candidates > 0
    assert len(res.chunks) <= 2


def test_reranker_reorders(retriever):
    retriever.reranker = KeywordReranker("Sourdough")
    res = retriever.retrieve("ROC-AUC of the Transformer", top_k=3)
    assert res.reranker_used
    assert "Sourdough" in res.chunks[0].text
    assert res.chunks[0].rerank_score == 5.0


def test_reranker_failure_falls_back_to_fusion(retriever):
    retriever.reranker = BrokenReranker()
    res = retriever.retrieve("ROC-AUC of the Transformer", top_k=2)
    assert not res.reranker_used and "OOM" in res.reranker_error
    assert res.chunks[0].document_name == "ml.txt"
    assert res.chunks[0].rerank_score is None


def test_reranker_can_be_disabled_per_request(retriever):
    retriever.reranker = KeywordReranker("Sourdough")
    res = retriever.retrieve("ROC-AUC", use_reranker=False)
    assert not res.reranker_used


def test_embedding_failure_falls_back_to_bm25(retriever):
    def boom(texts):
        raise EmbeddingError("embedding service down")

    retriever.embedder = type("E", (), {"embed": staticmethod(boom)})()
    res = retriever.retrieve("sourdough starter", top_k=2)
    assert res.dense_error and res.dense_candidates == 0
    assert res.chunks and "Sourdough" in res.chunks[0].text


def test_delete_updates_sparse_index(retriever, document_service):
    ml = next(d for d in document_service.list_documents() if d.filename == "ml.txt")
    document_service.delete(ml.id)
    res = retriever.retrieve("ROC-AUC 0.959", top_k=5)
    assert all(c.document_name != "ml.txt" for c in res.chunks)


def test_empty_corpus(settings, document_service):
    r = HybridRetriever(settings, document_service.store, document_service.embedder, document_service.dense_index, BM25Index())
    res = r.retrieve("anything")
    assert res.chunks == []
    assert evidence_check(res, settings) == (False, "no candidates retrieved")


# --- Evidence gate ----------------------------------------------------------------------------
def _rc(**scores) -> RetrievedChunk:
    return RetrievedChunk(chunk_id="d:0", document_id="d", document_name="d", chunk_index=0, text="t", **scores)


def test_evidence_gate_rerank_threshold(settings):
    settings.min_rerank_score = -5.0
    ok = RetrievalResult(chunks=[_rc(rerank_score=-3.0)], reranker_used=True)
    bad = RetrievalResult(chunks=[_rc(rerank_score=-11.0)], reranker_used=True)
    assert evidence_check(ok, settings)[0] is True
    assert evidence_check(bad, settings)[0] is False


def test_evidence_gate_dense_fallback(settings):
    settings.min_dense_score = 0.25
    assert evidence_check(RetrievalResult(chunks=[_rc(dense_score=0.4)]), settings)[0] is True
    assert evidence_check(RetrievalResult(chunks=[_rc(dense_score=0.1)]), settings)[0] is False


def test_evidence_gate_bm25_only(settings):
    assert evidence_check(RetrievalResult(chunks=[_rc(bm25_score=2.1)]), settings)[0] is True
    assert evidence_check(RetrievalResult(chunks=[_rc()]), settings)[0] is False


def test_bm25_works_on_tiny_corpus():
    """Regression: Okapi IDF is <= 0 when a term is in >= half the docs (N=2 here)."""
    idx = BM25Index()
    idx.build([_chunk(1, "transformer roc-auc 0.959"), _chunk(2, "sourdough bread")])
    hits = idx.search("transformer", k=2)
    assert hits and hits[0][0] == 1


def test_bm25_segments_match_single_index_and_scale():
    """Per-document segments must score exactly like one global index, and stay fast at scale."""
    import random
    import time

    rng = random.Random(0)
    vocab = [f"w{i}" for i in range(5000)]

    def mk(i, doc):
        return ChunkRecord(id=i, chunk_id=f"{doc}:{i}", document_id=doc, document_name=doc, chunk_index=i,
                           text=" ".join(rng.choices(vocab, k=80)))

    chunks = [mk(i, f"doc{i % 7}") for i in range(50_000)]
    whole = BM25Index()
    whole.build(chunks)
    incremental = BM25Index()
    for d in {c.document_id for c in chunks}:
        incremental.add_document(d, [c for c in chunks if c.document_id == d])

    q = "w1 w42 w999 w4000"
    hits = incremental.search(q, k=20)
    assert hits == whole.search(q, k=20)
    assert len(hits) == 20 and hits == sorted(hits, key=lambda h: -h[1])
    # Median of several runs, generous bound: catches a regression to scanning every chunk
    # (seconds) without flaking on a busy machine (typically ~1 ms).
    timings = []
    for _ in range(5):
        start = time.perf_counter()
        incremental.search(q, k=20)
        timings.append(time.perf_counter() - start)
    median = sorted(timings)[2]
    assert median < 0.5, f"BM25 query median {median:.3f}s on 50k chunks"


def test_bm25_remove_document_restores_statistics():
    a = [ChunkRecord(id=1, chunk_id="a:0", document_id="a", document_name="a", chunk_index=0, text="alpha beta")]
    b = [ChunkRecord(id=2, chunk_id="b:0", document_id="b", document_name="b", chunk_index=0, text="alpha gamma")]
    only_a = BM25Index()
    only_a.build(a)
    idx = BM25Index()
    idx.add_document("a", a)
    idx.add_document("b", b)
    idx.remove_document("b")
    assert idx.search("alpha beta", 5) == only_a.search("alpha beta", 5)
    assert idx.size == 1 and idx.document_ids() == {"a"}
