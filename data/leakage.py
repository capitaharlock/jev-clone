"""Contamination / leakage detector (#T-data-schema, #T-firewall seed).

Two layers: exact normalized-hash match against blocked texts, plus
word-3-gram Jaccard similarity to catch paraphrased benchmark/teacher
leaks. Runs in CI over every registered dataset sample.
"""
from __future__ import annotations

import hashlib
import re

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS.sub(" ", text.strip().lower())


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def trigrams(text: str) -> set[str]:
    words = normalize(text).split()
    return {" ".join(words[i : i + 3]) for i in range(len(words) - 2)} or set(words)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class LeakageDetector:
    def __init__(self, blocked_texts: list[str], sim_threshold: float = 0.5) -> None:
        self.sim_threshold = sim_threshold
        self._hashes = {text_hash(t) for t in blocked_texts}
        self._tri = [(t, trigrams(t)) for t in blocked_texts]

    def scan(self, text: str) -> tuple[bool, str]:
        """(hit, reason). Hit = exact or paraphrase of blocked content."""
        if text_hash(text) in self._hashes:
            return True, "exact match against blocked content"
        tri = trigrams(text)
        for raw, btri in self._tri:
            if jaccard(tri, btri) >= self.sim_threshold:
                return True, f"paraphrase-suspect of blocked: {raw[:80]!r}"
        return False, "clean"

    def scan_all(self, texts: list[str]) -> list[tuple[str, str]]:
        hits = []
        for t in texts:
            hit, reason = self.scan(t)
            if hit:
                hits.append((t, reason))
        return hits
