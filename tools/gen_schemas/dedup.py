"""Similarity dedup for generated schemas, on top of ``data.leakage``.

Reuses the project's word-3-gram Jaccard (the same primitive the firewall
uses for paraphrase detection) instead of a second, divergent notion of
"similar". A candidate over the threshold against anything already accepted
is DISCARDED and COUNTED — a rejection with a reason, not a silent drop.

An inverted trigram index keeps the sweep near-linear: only accepted schemas
sharing at least one trigram are ever scored.
"""
from __future__ import annotations

from data.leakage import jaccard, trigrams


class SchemaDeduper:
    def __init__(self, threshold: float = 0.6) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.threshold = threshold
        self._tris: list[set[str]] = []
        self._index: dict[str, set[int]] = {}
        self.n_checked = 0
        self.n_rejected = 0

    def __len__(self) -> int:
        return len(self._tris)

    def similarity(self, text: str) -> tuple[float, int]:
        """Highest Jaccard against anything accepted, and its index (-1 if none)."""
        tri = trigrams(text)
        candidates: set[int] = set()
        for g in tri:
            hit = self._index.get(g)
            if hit:
                candidates |= hit
        best, best_i = 0.0, -1
        for i in candidates:
            s = jaccard(tri, self._tris[i])
            if s > best:
                best, best_i = s, i
        return best, best_i

    def add(self, text: str) -> tuple[bool, float]:
        """(accepted, similarity). A rejected candidate is never indexed."""
        self.n_checked += 1
        sim, _ = self.similarity(text)
        if sim >= self.threshold:
            self.n_rejected += 1
            return False, sim
        tri = trigrams(text)
        idx = len(self._tris)
        self._tris.append(tri)
        for g in tri:
            self._index.setdefault(g, set()).add(idx)
        return True, sim

    def rejection_rate(self) -> float:
        return self.n_rejected / self.n_checked if self.n_checked else 0.0
