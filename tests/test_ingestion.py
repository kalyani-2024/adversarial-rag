import pytest

from app.core.errors import (
    DocumentNotFoundError,
    DocumentParseError,
    DuplicateDocumentError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from app.ingestion.chunking import chunk_pages, chunk_text
from app.ingestion.cleaning import clean_text
from app.ingestion.parsers import PageText, parse_document
from app.retrieval.dense import DenseIndex
from app.services.document_service import DocumentService
from app.storage.document_store import DocumentStore
from tests.conftest import make_pdf

LOREM = " ".join(f"Sentence number {i} talks about topic {i % 7}." for i in range(80))


# --- chunking ------------------------------------------------------------------
def test_chunks_respect_size_and_cover_text():
    chunks = chunk_text(LOREM, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 5
    assert all(len(c) <= 200 for c in chunks)
    # every sentence survives chunking
    for i in range(80):
        assert any(f"Sentence number {i} " in c for c in chunks)


def test_chunks_do_not_split_words():
    words = set(LOREM.split())
    for c in chunk_text(LOREM, chunk_size=150, chunk_overlap=30):
        for w in c.split():
            assert w in words


def test_chunks_overlap():
    chunks = chunk_text(LOREM, chunk_size=200, chunk_overlap=60)
    shared = [set(a.split()[-3:]) & set(b.split()) for a, b in zip(chunks, chunks[1:])]
    assert all(shared)


def test_chunk_overlap_must_be_smaller():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=10, chunk_overlap=10)


def test_short_and_empty_text():
    assert chunk_text("hello world", 100, 10) == ["hello world"]
    assert chunk_text("   ", 100, 10) == []


def test_chunk_pages_keeps_page_numbers_and_global_index():
    pages = [PageText(LOREM[:500], page=1), PageText(LOREM[500:1000], page=2)]
    chunks = chunk_pages(pages, 200, 40)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert {c.page for c in chunks} == {1, 2}
    assert chunks[0].page == 1 and chunks[-1].page == 2


# --- cleaning / parsing -----------------------------------------------------------
def test_clean_text_normalizes():
    raw = "Retrie-\nval   works\r\n\n\n\nwell\x07"
    assert clean_text(raw, file_type="pdf") == "Retrieval works\n\nwell"


def test_clean_markdown_links_and_images():
    assert clean_text("See [the docs](http://x.io) ![img](a.png)", file_type="markdown") == "See the docs"


def test_parse_pdf_pages():
    parsed = parse_document("paper.pdf", make_pdf(["Alpha page one", "Beta page two"]))
    assert parsed.num_pages == 2
    assert [p.page for p in parsed.pages] == [1, 2]
    assert "Beta" in parsed.pages[1].text


def test_parse_unsupported_type():
    with pytest.raises(UnsupportedFileTypeError):
        parse_document("image.png", b"\x89PNG")


def test_parse_malformed_pdf():
    with pytest.raises(DocumentParseError):
        parse_document("broken.pdf", b"%PDF-1.4 this is not really a pdf")


def test_parse_binary_as_text_rejected():
    with pytest.raises(DocumentParseError):
        parse_document("notes.txt", b"abc\x00\x01\x02")


def test_parse_empty_file():
    with pytest.raises(DocumentParseError):
        parse_document("empty.md", b"")


# --- document service ---------------------------------------------------------------
def test_ingest_preserves_metadata(document_service: DocumentService):
    info = document_service.ingest("paper.pdf", make_pdf(["Alpha " * 60, "Beta " * 60]))
    assert info.file_type == "pdf" and info.num_pages == 2 and info.num_chunks >= 2
    chunks = document_service.store.get_document_chunks(info.id)
    assert all(c.document_name == "paper.pdf" for c in chunks)
    assert chunks[0].chunk_id == f"{info.id}:0"
    assert {c.page for c in chunks} == {1, 2}
    assert document_service.dense_index.size == len(chunks)


def test_ingest_markdown_and_txt(document_service: DocumentService):
    md = document_service.ingest("notes.md", b"# Title\n\nSome **markdown** content.")
    txt = document_service.ingest("notes.txt", b"Plain text content.")
    assert md.file_type == "markdown" and txt.file_type == "txt"
    assert md.num_pages is None
    assert len(document_service.list_documents()) == 2


def test_duplicate_detected_by_content_hash(document_service: DocumentService):
    first = document_service.ingest("a.txt", b"same content")
    with pytest.raises(DuplicateDocumentError) as exc:
        document_service.ingest("renamed.txt", b"same content")
    assert exc.value.details["document_id"] == first.id


def test_path_traversal_filename_is_sanitized(document_service: DocumentService):
    info = document_service.ingest("../../etc/evil.txt", b"content")
    assert info.filename == "evil.txt"


def test_file_too_large(document_service: DocumentService):
    document_service.settings.max_upload_mb = 0.0001
    with pytest.raises(FileTooLargeError):
        document_service.ingest("big.txt", b"x" * 1000)


def test_empty_extraction_rejected(document_service: DocumentService):
    with pytest.raises(DocumentParseError):
        document_service.ingest("blank.txt", b"   \n\n  ")
    assert document_service.list_documents() == []


def test_delete_removes_chunks_from_store_and_index(document_service: DocumentService):
    keep = document_service.ingest("keep.txt", LOREM.encode())
    drop = document_service.ingest("drop.txt", ("Other words " * 100).encode())
    before = document_service.dense_index.size
    removed = document_service.delete(drop.id)
    assert removed == drop.num_chunks
    assert document_service.dense_index.size == before - removed
    assert document_service.store.count_chunks() == keep.num_chunks
    assert document_service.dense_index.ids() == {c.id for c in document_service.store.all_chunks()}
    with pytest.raises(DocumentNotFoundError):
        document_service.delete(drop.id)


def test_embedding_failure_leaves_no_partial_state(document_service: DocumentService):
    def boom(texts):
        raise RuntimeError("model down")

    document_service.embedder.embed = boom
    with pytest.raises(RuntimeError):
        document_service.ingest("a.txt", b"some content")
    assert document_service.list_documents() == []


def test_index_persists_and_self_heals(settings, embedder, document_service: DocumentService):
    document_service.ingest("a.txt", LOREM.encode())
    n = document_service.dense_index.size

    # restart: index loads from disk and is already consistent
    reopened = DocumentService(settings, DocumentStore(settings.db_path), embedder, DenseIndex(settings.index_dir / "dense.faiss"))
    assert reopened.dense_index.size == n
    assert reopened.sync_index() == 0

    # index file lost (e.g. ephemeral disk): rebuilt from SQLite
    (settings.index_dir / "dense.faiss").unlink()
    healed = DocumentService(settings, DocumentStore(settings.db_path), embedder, DenseIndex(settings.index_dir / "dense.faiss"))
    assert healed.dense_index.size == 0
    assert healed.sync_index() == n
    assert healed.dense_index.size == n


def test_listener_notified(document_service: DocumentService):
    calls = []
    document_service.add_listener(lambda: calls.append(1))
    info = document_service.ingest("a.txt", b"hello")
    document_service.delete(info.id)
    assert len(calls) == 2
