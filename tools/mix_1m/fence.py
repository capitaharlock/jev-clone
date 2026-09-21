"""Benchmark-clean fence for `decision-mix-clean-1m` (§§18, 77).

ZERO Banking77 / HelpSteer2 / PubMedQA rows and ZERO Jevals items in
train — asserted, not reported. Three layers, each of which raises:

1. **The registry layer.** `data.mix.clean_datasets()` is the only list
   the assembler plans from, and `assert_spec_clean()` re-checks the
   realised descriptor: a fenced source that reappears in a quota is a
   `FenceBreach`, not a warning.
2. **The id layer.** `#T-data-eval` freezes a clean-room Jevals id
   registry (`artifacts/gates/T-data-eval/jevals_ids.json`). Every
   selected question is checked against it under the same
   `<dataset>:train:<qid>` key `eval/data_eval.py::assert_sealed` uses,
   so the two barriers cannot disagree about what "in train" means.
3. **The content layer.** A normalised sha256 of the state text, using
   `data/leakage.py`'s normaliser, catches a fenced row that arrives
   through a different source id.

A missing blocklist is recorded as EMPTY and never as satisfied: the
registry layer still holds, and the gate says how many ids it checked.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.mix import MIX_1M_FENCED  # noqa: E402

#: `pubmedqa` is not in the registry at all; it is named so a future
#: prefetch job cannot add it without tripping this list first.
FENCED_DATASETS = frozenset(set(MIX_1M_FENCED) | {"pubmedqa"})

#: where #T-data-eval writes the frozen clean-room Jevals id registry
JEVALS_IDS = ROOT / "artifacts" / "gates" / "T-data-eval" / "jevals_ids.json"

_WS = re.compile(r"\s+")


class FenceBreach(AssertionError):
    """A fenced dataset, Jevals id or benchmark hash reached the mixture."""


def normalize(text: str) -> str:
    return _WS.sub(" ", (text or "").strip().lower())


def state_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def load_blocklist(path=None) -> set:
    """Normalised state hashes that must never enter train."""
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    return set(json.loads(p.read_text()))


def load_jevals_ids(path=None) -> set:
    """The frozen Jevals id registry, or an empty set if none is published."""
    p = Path(path or JEVALS_IDS)
    if not p.exists():
        return set()
    data = json.loads(p.read_text())
    ids = data.get("ids", data) if isinstance(data, dict) else data
    return set(ids)


def assert_spec_clean(datasets) -> list:
    """No fenced source may appear in the mixture. Raises `FenceBreach`."""
    hits = sorted(set(datasets) & FENCED_DATASETS)
    if hits:
        raise FenceBreach(
            f"benchmark fence breached: {hits} are in the mixture; §§18, 77 "
            f"require zero Banking77/HelpSteer2/PubMedQA rows in train")
    return sorted(datasets)


class Fence:
    """Row-level barrier, counting every rejection it makes."""

    def __init__(self, jevals_hashes=None, jevals_ids=None) -> None:
        self.jevals_hashes = set(jevals_hashes or ())
        self.jevals_ids = set(jevals_ids or ())
        self.blocked_dataset = 0
        self.blocked_hash = 0
        self.blocked_id = 0
        self.checked = 0

    def check(self, dataset: str, state: str, qid: str = "") -> tuple:
        """`(allowed, reason)`. False means the row must not enter train."""
        self.checked += 1
        if dataset in FENCED_DATASETS:
            self.blocked_dataset += 1
            return False, f"fenced dataset {dataset}"
        if qid and f"{dataset}:train:{qid}" in self.jevals_ids:
            self.blocked_id += 1
            return False, "jevals id registry hit"
        if qid in self.jevals_ids:
            self.blocked_id += 1
            return False, "jevals id registry hit"
        if self.jevals_hashes and state_hash(state) in self.jevals_hashes:
            self.blocked_hash += 1
            return False, "jevals state-hash blocklist hit"
        return True, "clean"

    @property
    def blocked(self) -> int:
        return self.blocked_dataset + self.blocked_hash + self.blocked_id

    def assert_clean(self) -> dict:
        """Nothing fenced may have been SEEN, let alone kept."""
        if self.blocked:
            raise FenceBreach(
                f"benchmark fence breached while assembling the mixture: "
                f"{self.blocked_dataset} fenced-dataset rows, "
                f"{self.blocked_id} Jevals ids, {self.blocked_hash} Jevals "
                f"state hashes reached the selection")
        return self.report()

    def report(self) -> dict:
        return {
            "fenced_datasets": sorted(FENCED_DATASETS),
            "jevals_ids_known": len(self.jevals_ids),
            "jevals_hashes_known": len(self.jevals_hashes),
            "jevals_ids_path": os.path.relpath(JEVALS_IDS, ROOT),
            "questions_checked": self.checked,
            "blocked": {"dataset": self.blocked_dataset,
                        "jevals_id": self.blocked_id,
                        "jevals_hash": self.blocked_hash},
            "clean": self.blocked == 0,
            "enforcement": ("tools.mix_1m.fence.FenceBreach — assembly "
                            "aborts; a fenced row is never counted, capped "
                            "or reported as a share"),
        }
