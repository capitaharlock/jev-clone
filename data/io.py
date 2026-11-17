"""JSONL helpers shared by the builders, the trainer and the benchmarks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_episode(row: dict) -> bool:
    """A row the trainer can score: text fields and a gold among the ids."""
    candidates = row.get("candidates") or []
    ids = [candidate.get("id") for candidate in candidates]
    return bool(row.get("state") and row.get("question") and len(ids) >= 2
                and len(set(ids)) == len(ids)
                and all(candidate.get("text") for candidate in candidates)
                and row.get("answer") in ids)
