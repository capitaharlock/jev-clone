"""Grounded state streamer: yields states with provenance, never invented worlds."""
from __future__ import annotations

import glob
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class State:
    text: str
    parent_example_id: str
    source: str


def stream_prefetch_states(pattern: str, limit: int | None = None):
    """Yield State rows from universal-schema JSONL shards (grounded source)."""
    n = 0
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                state = (r.get("state") or "").strip()
                if not state:
                    continue
                # Parent = source + state hash, NOT the row's question id:
                # some adapters reuse question ids across rows (e.g. every
                # boolq row carries "boolq-0"), which would collapse all
                # parents into one group and a single split.
                tag = Path(path).stem
                pid = tag + ":" + hashlib.sha256(state.encode()).hexdigest()[:16]
                yield State(text=state, parent_example_id=pid, source=path)
                n += 1
                if limit is not None and n >= limit:
                    return
