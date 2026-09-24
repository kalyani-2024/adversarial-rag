"""Deterministic evaluation metrics (no LLM involved).

Retrieval is scored against verbatim *evidence snippets* rather than chunk
ids, so labels stay valid when chunk size/overlap changes.
"""

from __future__ import annotations

import re
import statistics
import unicodedata

# Hyphen-like characters (incl. ASCII '-') become spaces so "gradient-reversal",
# "gradient‑reversal" (non-breaking hyphen) and "gradient reversal" all match.
# Applied to both the answer and the expected facts.
_DASHES = re.compile("[-‐-―−]")
# Thousands separators inside numbers: "15,000" / "15 000" -> "15000".
_THOUSANDS = re.compile(r"(?<=\d)[,   ](?=\d{3}\b)")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, NFKC, hyphens -> spaces, drop thousands separators, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _DASHES.sub(" ", text)
    text = _THOUSANDS.sub("", text)
    return _WS.sub(" ", text).strip()


def fact_recall(answer: str, facts: list[list[str]]) -> float | None:
    """Fraction of expected facts present in the answer (any alias counts). None if no facts."""
    if not facts:
        return None
    ans = normalize(answer)
    hits = sum(1 for aliases in facts if any(normalize(a) in ans for a in aliases))
    return hits / len(facts)


def evidence_positions(chunks: list[str], evidence: list[str]) -> list[int | None]:
    """For each evidence snippet, the 1-based rank of the first chunk containing it (None if absent)."""
    norm_chunks = [normalize(c) for c in chunks]
    out: list[int | None] = []
    for e in evidence:
        ne = normalize(e)
        out.append(next((i for i, c in enumerate(norm_chunks, start=1) if ne in c), None))
    return out


def retrieval_scores(chunks: list[str], evidence: list[str], k: int) -> dict[str, float]:
    """hit@k, evidence recall@k and MRR for one query."""
    pos = evidence_positions(chunks[:k], evidence)
    found = [p for p in pos if p is not None]
    return {
        "hit": 1.0 if found else 0.0,
        "recall": len(found) / len(evidence),
        "mrr": 1.0 / min(found) if found else 0.0,
    }


def mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.fmean(vals), 4) if vals else None


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))], 1)
