"""Multi-question packer: 3–10 Q per state, whole group shares one split (§§8, 58, 132)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tools.data_factory.deduper import group_split  # noqa: E402

MIN_Q = 3
MAX_Q = 10


def pack(entity_qs: list[dict], state: str, parent_id: str, seq_start: int) -> list[dict]:
    """Chunk an entity's questions into pack records. Skips entities with <3 Q."""
    if len(entity_qs) < MIN_Q:
        return []
    split = group_split(parent_id)
    import math

    n = len(entity_qs)
    n_chunks = max(1, math.ceil(n / MAX_Q))
    size = math.ceil(n / n_chunks)
    packs = []
    for i in range(0, n, size):
        chunk = entity_qs[i:i + size]
        if len(chunk) < MIN_Q:  # only possible when n < MIN_Q, guarded above
            continue
        packs.append({
            "state": state,
            "questions": chunk,
            "split": split,
            "parent_example_id": parent_id,
        })
    return packs
