"""Dedupe: exact SHA-256 + semantic (normalized Jaccard); group-by-parent splits."""
from __future__ import annotations

import hashlib
import json
import re
from collections import deque

_WORD = re.compile(r"[a-z0-9]+")


def norm(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def exact_key(rec: dict) -> str:
    blob = json.dumps({"s": norm(rec["state"]), "q": [
        {"o": sorted(o["id"] for o in q["options"]), "a": q.get("answer")}
        for q in rec["questions"]]}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def semantic_key(rec: dict) -> str:
    # State tokens + question shape: same-state variants with different
    # option sets (distractor injection) are intentional, not dupes.
    toks = sorted(set(norm(rec["state"]).split()))
    shape = "|".join(
        ",".join(sorted(o["id"] for o in q["options"])) + "=" + str(q.get("answer"))
        for q in rec["questions"]
    )
    return hashlib.sha256((" ".join(toks[:64]) + "#" + shape).encode()).hexdigest()


class Deduper:
    def __init__(self, jaccard_threshold: float = 0.9, recent_window: int = 512):
        self.seen_exact: set[str] = set()
        self.seen_sem: set[str] = set()
        # Recent token sets from OTHER parents: variants of the same parent
        # are intentional (multi-Q packs, kept together by group_split), so
        # the Jaccard check only fires across different parents. Bounded
        # window keeps the pilot O(n) instead of O(n^2).
        self.recent: deque = deque(maxlen=recent_window)
        self.recent_parents: deque = deque(maxlen=recent_window)
        self.threshold = jaccard_threshold
        self.n_exact_dup = 0
        self.n_sem_dup = 0

    def check(self, rec: dict, parent_id: str = "") -> str | None:
        """Return None if accepted, else 'exact' / 'semantic'."""
        k = exact_key(rec)
        if k in self.seen_exact:
            self.n_exact_dup += 1
            return "exact"
        sk = semantic_key(rec)
        if sk in self.seen_sem:
            self.n_sem_dup += 1
            return "semantic"
        toks = set(norm(rec["state"]).split())
        for prev, pid in zip(self.recent, self.recent_parents):
            if pid == parent_id:
                continue
            union = toks | prev
            if not union:
                continue
            if len(toks & prev) / len(union) >= self.threshold:
                self.n_sem_dup += 1
                return "semantic"
        self.seen_exact.add(k)
        self.seen_sem.add(sk)
        self.recent.append(toks)
        self.recent_parents.append(parent_id)
        return None


def group_split(parent_id: str, n_groups: int = 10) -> str:
    """Deterministic split by parent group: paraphrases never leak across splits."""
    h = int(hashlib.sha256(parent_id.encode()).hexdigest(), 16)
    m = h % n_groups
    if m < 7:
        return "train"
    if m < 9:
        return "calibration"
    return "test"
