"""Golden dataset schema and loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from app.evaluation.metrics import normalize


class GoldenItem(BaseModel):
    id: str
    category: Literal["lookup", "synthesis", "bait", "compound", "unanswerable"]
    question: str
    # Each inner list is ONE fact with acceptable surface forms (any match counts).
    expected_facts: list[list[str]] = Field(default_factory=list)
    # Verbatim snippets from the corpus that a correct retrieval must surface.
    evidence: list[str] = Field(default_factory=list)
    answerable: bool = True
    note: str = ""


def load_golden(path: Path) -> list[GoldenItem]:
    items = [GoldenItem.model_validate(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate ids in golden dataset")
    return items


def validate_evidence(items: list[GoldenItem], corpus_text: str) -> list[str]:
    """Return evidence snippets that do NOT occur in the corpus (label drift guard)."""
    corpus = normalize(corpus_text)
    return [f"{i.id}: {e!r}" for i in items for e in i.evidence if normalize(e) not in corpus]
