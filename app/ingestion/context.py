"""Index-time context for chunks ("contextual retrieval", without an LLM).

A chunk is retrieved by similarity between the question and the chunk's own
words. That fails for questions about the *document itself*: a title page lists
names, affiliations and e-mail addresses but never contains the words
"author", "title" or "university", so neither BM25, the embedding model nor
the cross-encoder connects "who are the authors?" to it.

The fix is to describe what the first-page chunks *are*. The description is:

  * stored separately from the chunk text (`context`), so the passage shown and
    cited to the user stays verbatim;
  * prepended only for indexing (embedding, BM25, reranking) and shown to the
    LLM as a note, so it can interpret the passage ("these names are the authors");
  * deterministic (no LLM call, no cost), so indexing stays fast and reproducible.

Measured on a real two-column paper (see docs/ARCHITECTURE.md): the reranker
score of the title page for "who are the authors" went from -11.3 to -3.5,
"who wrote this paper" -11.3 -> -0.7, "what is the title" -11.4 -> +0.2, while
an unrelated question stayed at -11.5 (no false matches).
"""

from __future__ import annotations

# Bump when the way chunks are indexed changes; stores built with an older
# version are re-contextualized and re-embedded at startup.
INDEX_VERSION = 2

# Chunks this close to the start of a document are treated as front matter.
FRONT_MATTER_CHUNKS = 2


def front_matter_context(filename: str) -> str:
    return (
        f"First page of the document {filename}. It gives the title of the document (paper, report or article), "
        "the authors who wrote it, their affiliations (university, company or organization), contact e-mails, "
        "the date or venue, and the abstract or summary of what the document is about."
    )


def is_front_matter(chunk_index: int, page: int | None) -> bool:
    return chunk_index < FRONT_MATTER_CHUNKS and page in (None, 1)


def index_text(context: str, text: str) -> str:
    """What gets embedded, BM25-indexed and reranked."""
    return f"{context}\n{text}" if context else text
