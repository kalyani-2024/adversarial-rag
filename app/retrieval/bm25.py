"""Sparse lexical retrieval (BM25) over per-document inverted-index segments.

Complements dense retrieval on exact-match signals embeddings blur: numbers
("0.959"), identifiers, acronyms ("ROC-AUC"), rare names.

Design (Lucene-style segments):
  * each document is one immutable *segment*: posting arrays (chunk position,
    term frequency) per term, plus chunk lengths;
  * corpus statistics (document frequency per term, chunk count, total
    length) are global, so scores are identical to one big index;
  * adding a document tokenizes only that document; deleting drops its
    segment and subtracts its statistics, so a 100 MB upload never forces a
    full re-index of everything else;
  * a query touches only the postings of its terms (vectorized numpy), so it
    stays in the low milliseconds at ~10^5 chunks.
Updates are copy-on-write: readers take a snapshot and never see a half-applied change.
Beyond ~10^6 chunks, or with several replicas, move to OpenSearch/Elasticsearch,
Tantivy or Postgres full-text.
"""

from __future__ import annotations

import heapq
import math
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from app.schemas.documents import ChunkRecord

# Keeps decimals and hyphenated terms together ("0.959", "roc-auc") ...
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have how in is it its of on or that the this to was were "
    "what when where which who why will with does do did can".split()
)

K1, B = 1.5, 0.75


def tokenize(text: str) -> list[str]:
    """Lowercase tokens; compound tokens also emit their parts ("roc-auc" -> roc-auc, roc, auc)."""
    tokens: list[str] = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok in _STOPWORDS:
            continue
        tokens.append(tok)
        if "-" in tok or "." in tok:
            tokens.extend(p for p in re.split(r"[.\-]", tok) if p and p not in _STOPWORDS and not p.isdigit())
    return tokens


@dataclass(frozen=True)
class _Segment:
    chunk_ids: np.ndarray  # int64, chunk id per local position
    lengths: np.ndarray  # float32, token count per chunk
    postings: dict[str, tuple[np.ndarray, np.ndarray]]  # term -> (local positions int32, tf float32)
    df: Counter  # term -> number of chunks in this segment containing it

    @staticmethod
    def build(chunks: list[ChunkRecord]) -> "_Segment":
        lists: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
        lengths = []
        for pos, chunk in enumerate(chunks):
            tokens = tokenize(chunk.text)
            lengths.append(len(tokens))
            for term, tf in Counter(tokens).items():
                docs, tfs = lists[term]
                docs.append(pos)
                tfs.append(tf)
        postings = {t: (np.asarray(d, dtype=np.int32), np.asarray(f, dtype=np.float32)) for t, (d, f) in lists.items()}
        return _Segment(
            chunk_ids=np.asarray([c.id for c in chunks], dtype=np.int64),
            lengths=np.asarray(lengths, dtype=np.float32),
            postings=postings,
            df=Counter({t: len(d) for t, (d, _) in postings.items()}),
        )


@dataclass(frozen=True)
class _Snapshot:
    segments: dict[str, _Segment]  # document_id -> segment
    df: Counter
    n_chunks: int
    total_len: float


_EMPTY = _Snapshot(segments={}, df=Counter(), n_chunks=0, total_len=0.0)


class BM25Index:
    """BM25 with Lucene's non-negative IDF.

        score(q, d) = sum_t IDF(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
        IDF(t)      = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))

    Classic Okapi IDF, ln((N - n_t + 0.5) / (n_t + 0.5)), goes to zero or
    negative once a term appears in half the chunks, which silently breaks
    retrieval on small corpora (e.g. a single uploaded document).
    """

    def __init__(self) -> None:
        self._snap = _EMPTY
        self._write_lock = threading.Lock()

    # --- introspection ------------------------------------------------------------
    @property
    def size(self) -> int:
        return self._snap.n_chunks

    def document_ids(self) -> set[str]:
        return set(self._snap.segments)

    # --- writes (copy-on-write) --------------------------------------------------
    def build(self, chunks: list[ChunkRecord]) -> None:
        """Rebuild from scratch (startup). Chunks are grouped into one segment per document."""
        by_doc: dict[str, list[ChunkRecord]] = defaultdict(list)
        for c in chunks:
            by_doc[c.document_id].append(c)
        segments = {doc_id: _Segment.build(cs) for doc_id, cs in by_doc.items()}
        df: Counter = Counter()
        for seg in segments.values():
            df.update(seg.df)
        with self._write_lock:
            self._snap = _Snapshot(segments, df, sum(len(s.chunk_ids) for s in segments.values()),
                                   float(sum(s.lengths.sum() for s in segments.values())))

    def add_document(self, document_id: str, chunks: list[ChunkRecord]) -> None:
        segment = _Segment.build(chunks)  # expensive part happens outside the lock
        with self._write_lock:
            old = self._snap
            base = old
            if document_id in old.segments:  # replace
                base = self._without(old, document_id)
            df = base.df.copy()
            df.update(segment.df)
            self._snap = _Snapshot({**base.segments, document_id: segment}, df,
                                   base.n_chunks + len(segment.chunk_ids), base.total_len + float(segment.lengths.sum()))

    def remove_document(self, document_id: str) -> None:
        with self._write_lock:
            if document_id in self._snap.segments:
                self._snap = self._without(self._snap, document_id)

    @staticmethod
    def _without(snap: _Snapshot, document_id: str) -> _Snapshot:
        seg = snap.segments[document_id]
        df = snap.df.copy()
        df.subtract(seg.df)
        df = +df  # drop zero counts
        segments = {k: v for k, v in snap.segments.items() if k != document_id}
        return _Snapshot(segments, df, snap.n_chunks - len(seg.chunk_ids), snap.total_len - float(seg.lengths.sum()))

    # --- reads -------------------------------------------------------------------
    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return `(chunk_id, bm25_score)` for chunks with a positive score, best first."""
        snap = self._snap  # atomic reference read: a consistent snapshot
        tokens = tokenize(query)
        if not snap.n_chunks or not tokens or k <= 0:
            return []
        n, avgdl = snap.n_chunks, snap.total_len / snap.n_chunks
        idf = {t: math.log(1 + (n - snap.df[t] + 0.5) / (snap.df[t] + 0.5)) for t in set(tokens) if snap.df.get(t)}
        if not idf:
            return []
        best: list[tuple[float, int]] = []
        for seg in snap.segments.values():
            acc = None
            for term in tokens:
                posting = seg.postings.get(term)
                if posting is None:
                    continue
                docs, tfs = posting
                norm = K1 * (1 - B + B * seg.lengths[docs] / avgdl) if avgdl else K1
                if acc is None:
                    acc = np.zeros(len(seg.chunk_ids), dtype=np.float32)
                acc[docs] += idf[term] * tfs * (K1 + 1) / (tfs + norm)
            if acc is None:
                continue
            candidates = np.flatnonzero(acc > 0)
            if candidates.size > k:
                candidates = candidates[np.argpartition(-acc[candidates], k - 1)[:k]]
            best.extend((float(acc[i]), int(seg.chunk_ids[i])) for i in candidates)
        top = heapq.nlargest(k, best, key=lambda x: (x[0], -x[1]))
        return [(chunk_id, score) for score, chunk_id in top]
