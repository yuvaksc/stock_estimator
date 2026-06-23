"""Tiny in-process metrics registry (cache hit-rate, LLM calls, ingest counts).

Exposed via GET /metrics and logged periodically by the worker. Intentionally
process-local — good enough for a single-instance demo; swap for Prometheus to
scale out.
"""

from __future__ import annotations

import threading
from collections import Counter


class Metrics:
    def __init__(self) -> None:
        self._c: Counter[str] = Counter()
        self._lock = threading.Lock()

    def incr(self, key: str, n: int = 1) -> None:
        with self._lock:
            self._c[key] += n

    def snapshot(self) -> dict:
        with self._lock:
            c = dict(self._c)
        hits = c.get("cache_hits_exact", 0) + c.get("cache_hits_semantic", 0)
        misses = c.get("cache_misses", 0)
        total = hits + misses
        c["cache_hits"] = hits
        c["cache_hit_rate"] = round(hits / total, 4) if total else 0.0
        return c


metrics = Metrics()
