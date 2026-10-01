"""Recursive, boundary-aware chunking.

Splits on the coarsest separator that fits (paragraph -> line -> sentence ->
word) so chunks end on natural boundaries instead of mid-word, then packs the
pieces into ~`chunk_size` character windows with `chunk_overlap` characters of
trailing context carried into the next chunk.

Chunks never cross page boundaries: that keeps page citations exact.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.context import index_text
from app.ingestion.parsers import PageText

SEPARATORS = ("\n\n", "\n", ". ", " ")


@dataclass
class TextChunk:
    text: str
    chunk_index: int
    page: int | None
    context: str = ""  # index-time description (see app/ingestion/context.py); never shown as the passage

    @property
    def index_text(self) -> str:
        return index_text(self.context, self.text)


def _split_recursive(text: str, chunk_size: int, separators: tuple[str, ...] = SEPARATORS) -> list[str]:
    """Split `text` into pieces no longer than `chunk_size` using the coarsest separator possible."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    if not separators:  # no boundary left: hard split
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, rest = separators[0], separators[1:]
    parts = text.split(sep)
    if len(parts) == 1:
        return _split_recursive(text, chunk_size, rest)

    pieces: list[str] = []
    for i, part in enumerate(parts):
        piece = part + (sep if i < len(parts) - 1 else "")
        if len(piece) <= chunk_size:
            if piece.strip():
                pieces.append(piece)
        else:
            pieces.extend(_split_recursive(piece, chunk_size, rest))
    return pieces


def _overlap_tail(text: str, overlap: int) -> str:
    """Last ~`overlap` characters of `text`, starting at a word boundary."""
    if overlap <= 0 or len(text) <= overlap:
        return text if overlap > 0 else ""
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1 :] if 0 <= space < len(tail) - 1 else tail


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Pack recursive pieces into overlapping chunks of at most ~`chunk_size` chars."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    pieces = _split_recursive(text, chunk_size - chunk_overlap)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > chunk_size:
            chunks.append(current.strip())
            current = _overlap_tail(current, chunk_overlap)
        current += piece
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if c]


def chunk_pages(pages: list[PageText], chunk_size: int, chunk_overlap: int) -> list[TextChunk]:
    """Chunk each page independently; `chunk_index` is global within the document."""
    out: list[TextChunk] = []
    for page in pages:
        for text in chunk_text(page.text, chunk_size, chunk_overlap):
            out.append(TextChunk(text=text, chunk_index=len(out), page=page.page))
    return out
