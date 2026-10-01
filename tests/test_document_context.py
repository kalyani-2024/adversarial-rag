"""Document-level questions ("who are the authors?") must find the title page.

The title page of a paper lists names and affiliations but never the words
"author", "title" or "university". Front-matter chunks therefore get an
index-time context (app/ingestion/context.py) that retrieval can match, while
the passage text shown to the user stays verbatim.
"""

from app.generation.generator import format_context
from app.ingestion import parsers
from app.ingestion.context import INDEX_VERSION, front_matter_context
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import DenseIndex
from app.retrieval.hybrid import HybridRetriever
from app.services.document_service import DocumentService
from app.storage.document_store import DocumentStore
from tests.conftest import make_pdf

TITLE_PAGE = (
    "Personalized Behavioral Drift Detection for Online Exam Integrity\n"
    "Srivathsa H\nBITS Pilani Dubai Campus\nDubai, UAE\nMili Mohan\nBITS Pilani Dubai Campus\n\n"
)
BODY = " ".join(f"Section {i} discusses clickstream timing features and anomaly families in detail." for i in range(60))
PAPER = (TITLE_PAGE + BODY).encode()


def _retriever(settings, service: DocumentService) -> HybridRetriever:
    r = HybridRetriever(settings, service.store, service.embedder, service.dense_index, BM25Index())
    r.refresh_sparse_index()
    return r


def test_front_matter_chunks_get_context_and_text_stays_verbatim(document_service: DocumentService):
    info = document_service.ingest("paper.txt", PAPER)
    chunks = document_service.store.get_document_chunks(info.id)
    assert len(chunks) > 3
    expected = front_matter_context("paper.txt")
    assert [bool(c.context) for c in chunks[:4]] == [True, True, False, False]
    assert chunks[0].context == expected
    assert "authors" not in chunks[0].text.lower()  # the passage itself is untouched…
    assert chunks[0].index_text.startswith(expected) and chunks[0].text in chunks[0].index_text  # …the index text is not
    assert chunks[3].index_text == chunks[3].text


def test_authors_question_retrieves_the_title_page(settings, document_service: DocumentService):
    document_service.ingest("paper.txt", PAPER)
    retriever = _retriever(settings, document_service)
    for question in ("who are the authors", "Which university are the authors from?", "What is the title of the paper?"):
        result = retriever.retrieve(question, top_k=3)
        top = result.chunks[0]
        assert top.chunk_index in (0, 1) and top.bm25_rank is not None, question
        assert any(c.chunk_index == 0 and "Srivathsa" in c.text for c in result.chunks), question


def test_body_questions_are_not_hijacked_by_front_matter(settings, document_service: DocumentService):
    document_service.ingest("paper.txt", PAPER)
    result = _retriever(settings, document_service).retrieve("clickstream timing features anomaly families", top_k=3)
    assert result.chunks[0].chunk_index >= 2


def test_prompt_shows_context_as_a_note(settings, document_service: DocumentService):
    document_service.ingest("paper.txt", PAPER)
    chunks = _retriever(settings, document_service).retrieve("who are the authors", top_k=2).chunks
    prompt = format_context(chunks)
    assert "(Note about this passage: First page of the document paper.txt." in prompt
    assert "Srivathsa H" in prompt


def test_old_index_is_upgraded_on_startup(settings, embedder):
    store = DocumentStore(settings.db_path)
    service = DocumentService(settings, store, embedder, DenseIndex(settings.index_dir / "dense.faiss"))
    info = service.ingest("paper.txt", PAPER)
    # Simulate a store written before index version 2: no contexts, no version stamp.
    store.set_chunk_contexts([(None, c.id) for c in store.get_document_chunks(info.id)])
    store._conn.execute("DELETE FROM meta")
    store._conn.commit()
    old_vector_count = service.dense_index.size

    reopened = DocumentService(settings, DocumentStore(settings.db_path), embedder, DenseIndex(settings.index_dir / "dense.faiss"))
    assert reopened.sync_index() == old_vector_count  # every vector re-embedded with the new index text
    chunks = reopened.store.get_document_chunks(info.id)
    assert chunks[0].context == front_matter_context("paper.txt") and not chunks[3].context
    assert reopened.store.get_meta("index_version") == str(INDEX_VERSION)
    assert reopened.sync_index() == 0  # idempotent


def test_new_store_is_stamped_and_not_rebuilt(settings, embedder):
    service = DocumentService(settings, DocumentStore(settings.db_path), embedder, DenseIndex(settings.index_dir / "dense.faiss"))
    assert service.store.get_meta("index_version") == str(INDEX_VERSION)
    service.ingest("paper.txt", PAPER)
    assert service.sync_index() == 0


def test_pdf_uses_pdfminer_and_falls_back_to_pypdf(monkeypatch):
    pdf = make_pdf(["Alpha page one", "Beta page two"])
    parsed = parsers.parse_document("p.pdf", pdf)
    assert parsed.num_pages == 2 and "Alpha" in parsed.pages[0].text and "Beta" in parsed.pages[1].text

    monkeypatch.setattr(parsers, "_pdfminer_pages", lambda content: None)  # pdfminer unavailable / failed
    fallback = parsers.parse_document("p.pdf", pdf)
    assert fallback.num_pages == 2 and "Beta" in fallback.pages[1].text

    monkeypatch.setattr(parsers, "_pdfminer_pages", lambda content: ["only one page"])  # page-count mismatch
    assert parsers.parse_document("p.pdf", pdf).num_pages == 2
