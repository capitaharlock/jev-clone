"""License tracker + shard writer + metrics."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


class LicenseTracker:
    def __init__(self):
        self.counts: Counter = Counter()

    def record(self, source: str, license: str = "mixed-research"):
        self.counts[f"{source} :: {license}"] += 1

    def report(self) -> dict:
        return dict(self.counts)


class ShardWriter:
    def __init__(self, out_dir: str | Path, shard_size: int = 10000):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.buf: list[dict] = []
        self.shard_idx = 0
        self.total = 0
        self.manifest: list[dict] = []

    def write(self, rec: dict):
        self.buf.append(rec)
        if len(self.buf) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        name = f"synth-v1-{self.shard_idx:03d}.jsonl"
        path = self.out_dir / name
        h = hashlib.sha256()
        with open(path, "w") as fh:
            for r in self.buf:
                line = json.dumps(r, sort_keys=True)
                h.update(line.encode())
                fh.write(line + "\n")
        self.manifest.append({"file": name, "rows": len(self.buf), "sha256": h.hexdigest()})
        self.total += len(self.buf)
        self.shard_idx += 1
        self.buf = []


class Metrics:
    def __init__(self):
        self.proposed = 0  # every candidate attempt, council + variants
        self.council = 0  # council-routed items only (agreement denominator)
        self.accepted = 0
        self.rejected_validator = 0
        self.rejected_dupe = 0
        self.agreed = 0
        self.adjudicated = 0
        self.inverted = 0

    def report(self) -> dict:
        return {
            "proposed": self.proposed,
            "accepted": self.accepted,
            "reject_rate": round(1 - self.accepted / max(1, self.proposed), 4),
            "agreement_rate": round(self.agreed / max(1, self.council), 4),
            "adjudicated": self.adjudicated,
            "roles_inverted_frac": round(self.inverted / max(1, self.council), 4),
        }
