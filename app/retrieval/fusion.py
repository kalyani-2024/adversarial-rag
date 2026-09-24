"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009).

    RRF(d) = sum over rankers r of  1 / (k + rank_r(d))      (rank is 1-based)

Why RRF instead of a weighted sum of scores: cosine similarity (~[-1, 1]) and
BM25 (unbounded, corpus-dependent) live on incomparable scales, so score
fusion needs per-query normalization and tuned weights. RRF uses only ranks,
has one robust parameter (k=60 from the paper), and rewards documents that
both retrievers agree on.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FusedItem:
    id: int
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # ranker name -> 1-based rank


def reciprocal_rank_fusion(rankings: dict[str, list[int]], k: int = 60) -> list[FusedItem]:
    """Fuse ranked id lists. Ties are broken by best single rank, then id (deterministic)."""
    if k <= 0:
        raise ValueError("k must be positive")
    items: dict[int, FusedItem] = {}
    for name, ids in rankings.items():
        for rank, doc_id in enumerate(ids, start=1):
            item = items.setdefault(doc_id, FusedItem(id=doc_id, score=0.0))
            if name in item.ranks:  # ignore duplicates within one list
                continue
            item.ranks[name] = rank
            item.score += 1.0 / (k + rank)
    return sorted(items.values(), key=lambda it: (-it.score, min(it.ranks.values()), it.id))
