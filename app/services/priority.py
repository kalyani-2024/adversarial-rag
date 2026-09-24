"""Give interactive queries priority over background indexing.

Embedding a large upload is CPU-bound (~20 chunks/s on a laptop CPU) and uses
the same cores that query embedding and cross-encoder reranking need. The
ingestion worker therefore checks this gate between embedding batches and
pauses while any query is in flight, so chat latency stays flat while a big
file indexes. The pause is capped so indexing can never be starved forever.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator


class QueryActivity:
    def __init__(self) -> None:
        self._active = 0
        self._cond = threading.Condition()

    @property
    def active(self) -> int:
        return self._active

    @contextmanager
    def running(self) -> Iterator[None]:
        with self._cond:
            self._active += 1
        try:
            yield
        finally:
            with self._cond:
                self._active -= 1
                self._cond.notify_all()

    def wait_until_idle(self, max_wait_s: float) -> bool:
        """Block while queries are running (up to `max_wait_s`). Returns True if we had to wait."""
        with self._cond:
            if self._active == 0:
                return False
            self._cond.wait_for(lambda: self._active == 0, timeout=max_wait_s)
            return True
