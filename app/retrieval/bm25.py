"""Sparse lexical retrieval (BM25).

Complements dense retrieval on exact-match signals embeddings blur: numbers
("0.959"), identifiers, acronyms ("ROC-AUC"), rare names.

The index is rebuilt in memory from SQLite whenever the corpus changes. That
is O(total tokens) and takes milliseconds for thousands of chunks; beyond
~10^5-10^6 chunks you would move to an inverted-index engine (Elasticsearch/
OpenSearch, Tantivy, or Postgres full-text).
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter

from app.schemas.documents import ChunkRecord

# Keeps decimals and hyphenated terms together ("0.959", "roc-auc") ...
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have how in is it its of on or that the this to was were "
    "what when where which who why will with does do did can".split()
)


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


class _BM25:
    """BM25 with Lucene's non-negative IDF.

        score(q, d) = sum_t IDF(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
        IDF(t)      = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))

    Classic Okapi IDF, ln((N - n_t + 0.5) / (n_t + 0.5)), goes to zero or
    negative once a term appears in half the chunks, which silently breaks
    retrieval on small corpora (e.g. a single uploaded document).
    """

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tfs = [Counter(doc) for doc in corpus]
        self.lengths = [len(doc) for doc in corpus]
        self.avgdl = (sum(self.lengths) / len(corpus)) if corpus else 0.0
        df: Counter[str] = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(corpus)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query_tokens: list[str]) -> list[float]:
        out = []
        for tf, dl in zip(self.tfs, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl) if self.avgdl else self.k1
            s = 0.0
            for t in query_tokens:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            out.append(s)
        return out


class BM25Index:
    def __init__(self) -> None:
        self._bm25: _BM25 | None = None
        self._ids: list[int] = []
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        return len(self._ids)

    def build(self, chunks: list[ChunkRecord]) -> None:
        """Rebuild from scratch, then swap in atomically (readers never see a half-built index)."""
        corpus = [tokenize(c.text) for c in chunks]
        bm25 = _BM25(corpus) if chunks else None
        with self._lock:
            self._bm25, self._ids = bm25, [c.id for c in chunks]

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return `(chunk_id, bm25_score)` for chunks with a positive score, best first."""
        with self._lock:
            bm25, ids = self._bm25, self._ids
        tokens = tokenize(query)
        if bm25 is None or not tokens:
            return []
        scores = bm25.scores(tokens)
        ranked = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)[:k]
        return [(ids[i], float(scores[i])) for i in ranked if scores[i] > 0]
