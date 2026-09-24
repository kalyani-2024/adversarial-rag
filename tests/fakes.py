"""Deterministic test doubles: no network, no model downloads."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable

import numpy as np

from app.core.llm import LLMResponse, Message

Responder = Callable[[list[Message], str], str]


class FakeLLM:
    """Scripted LLM. Responses are chosen per `purpose` (e.g. 'judge').

    A purpose maps either to a list (consumed in order, last one repeats) or to
    a callable `(messages, purpose) -> str`.
    """

    def __init__(self, script: dict[str, list[str] | Responder] | None = None) -> None:
        self.script = script or {}
        self.calls: list[tuple[str, list[Message]]] = []

    def complete(self, messages, *, purpose, model=None, temperature=0.0, max_tokens=None, json_mode=False):
        self.calls.append((purpose, messages))
        entry = self.script.get(purpose)
        if entry is None:
            text = f"[fake {purpose}]"
        elif callable(entry):
            text = entry(messages, purpose)
        else:
            idx = min(sum(1 for p, _ in self.calls if p == purpose) - 1, len(entry) - 1)
            text = entry[idx]
        return LLMResponse(text=text, model="fake", prompt_tokens=100, completion_tokens=20, latency_ms=1.0)

    def stream(self, messages, *, purpose, on_token, model=None, temperature=0.0, max_tokens=None):
        """Streaming variant: emits the scripted text word by word."""
        resp = self.complete(messages, purpose=purpose, model=model, temperature=temperature, max_tokens=max_tokens)
        for word in re.findall(r"\S+\s*", resp.text):
            on_token(word)
        return resp

    def count(self, purpose: str) -> int:
        return sum(1 for p, _ in self.calls if p == purpose)


class HashEmbedder:
    """Bag-of-words hashing embedder: similar vocab -> similar vectors."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

