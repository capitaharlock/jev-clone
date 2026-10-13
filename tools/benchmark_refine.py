"""Incrementally rewrite Benchmark v2 prose without allowing gold changes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from data.episode_gen import qwen_paraphrase_batch

ROOT = Path(__file__).resolve().parents[1]
FACTORY = ROOT.parent / "decision-dataset-factory" / "work" / "benchmark-v2"
SOURCE = FACTORY / "dev.jsonl"
OUTPUT = FACTORY / "refined.jsonl"
REJECTS = FACTORY / "refine_rejects.jsonl"
STATE = FACTORY / "refine_state.json"
ACTIVE = FACTORY / "active.json"
VERSION = "benchmark-v2-qwen-prose-v1"
NUMBER = re.compile(r"(?<!\w)-?\d+(?:[.,]\d+)?")


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def read_rows(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except ValueError:
            # An interrupted final append is ignored and replaced next run.
            continue
    return rows


def required_tokens(row: dict) -> set[str]:
    """Facts that a rewrite may not drop: numbers and mentioned options."""
    state = row["state"]
    tokens = set(NUMBER.findall(state))
    for candidate in row["candidates"]:
        text = candidate["text"]
        if text.casefold() in state.casefold():
            tokens.add(text)
    return tokens


def acceptable(row: dict, rewritten: str | None) -> tuple[bool, str]:
    if not rewritten or len(rewritten.split()) < 8:
        return False, "empty-or-short"
    missing = sorted(token for token in required_tokens(row)
                     if token.casefold() not in rewritten.casefold())
    if missing:
        return False, "missing:" + "|".join(missing[:8])
    if rewritten.strip() == row["state"].strip():
        return False, "unchanged"
    return True, "ok"


def publish(row: dict, rewritten: str) -> dict:
    out = json.loads(json.dumps(row))
    out["state"] = rewritten.strip()
    out["provenance"]["source_id"] = "benchmark-v2-qwen-prose"
    out["provenance"]["revision"] = VERSION
    payload = json.dumps(
        {**out, "provenance": {**out["provenance"],
                               "content_sha256": "pending"}},
        sort_keys=True, ensure_ascii=False,
    ).encode()
    out["provenance"]["content_sha256"] = hashlib.sha256(payload).hexdigest()
    out["refinement"] = {
        "role": "prose-only",
        "gold_unchanged": True,
        "question_unchanged": True,
        "required_tokens": sorted(required_tokens(row)),
    }
    return out


def append_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def run(source: Path, output: Path, seconds: int, batch: int) -> dict:
    source_rows = read_rows(source)
    done_rows = read_rows(output)
    done = {row["id"] for row in done_rows}
    pending = [row for row in source_rows if row["id"] not in done]
    started = time.time()
    accepted = rejected = 0
    while pending and time.time() - started < seconds:
        chunk = pending[:batch]
        pending = pending[batch:]
        # Group by language because the writer's prompt is language-specific.
        for language in ("en", "es"):
            group = [row for row in chunk if row["language"] == language]
            if not group:
                continue
            rewrites = qwen_paraphrase_batch(
                [row["state"] for row in group], language, timeout=1800)
            good, bad = [], []
            for row, rewritten in zip(group, rewrites):
                ok, reason = acceptable(row, rewritten)
                if ok:
                    good.append(publish(row, rewritten))
                    accepted += 1
                else:
                    bad.append({"id": row["id"], "reason": reason,
                                "at": time.time()})
                    rejected += 1
            append_rows(output, good)
            append_rows(REJECTS, bad)
    published = len(read_rows(output))
    payload = {
        "status": "published" if published == len(source_rows) else "building",
        "source": str(source),
        "output": str(output),
        "source_rows": len(source_rows),
        "published": published,
        "remaining": len(source_rows) - published,
        "accepted_this_run": accepted,
        "rejected_this_run": rejected,
        "elapsed_s": round(time.time() - started, 1),
        "version": VERSION,
    }
    write_json(STATE, payload)
    if payload["status"] == "published":
        write_json(ACTIVE, {
            "path": str(output),
            "version": VERSION,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        })
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--batch", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.out, args.seconds, args.batch),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
