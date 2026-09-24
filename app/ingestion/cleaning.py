"""Text normalization applied before chunking."""

from __future__ import annotations

import re
import unicodedata

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")  # "retrie-\nval" -> "retrieval" (PDF line wraps)
_SPACES = re.compile(r"[ \t ]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_HTML_TAG = re.compile(r"<[^>\n]{1,200}>")


def clean_text(text: str, *, file_type: str = "txt") -> str:
    """Normalize unicode and whitespace; lightly de-noise markdown.

    Deliberately conservative: we keep headings, lists and numbers intact
    because they carry retrieval signal (BM25 in particular benefits).
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text)
    if file_type == "pdf":
        text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)
    if file_type == "markdown":
        text = _MD_IMAGE.sub("", text)
        text = _MD_LINK.sub(r"\1", text)
        text = _HTML_TAG.sub("", text)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()
