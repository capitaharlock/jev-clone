"""Consume train releases from the independent Decision Dataset Factory."""

from __future__ import annotations

import json
import os
from pathlib import Path

from data import benchmark_v2 as BV2
from data import episode_contract as EC

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = Path(os.environ.get(
    "FACTORY_V2_TRAIN_PATH",
    ROOT.parent / "decision-dataset-factory" / "work" /
    "training-v2" / "public" / "train.jsonl",
))


def convert(row: dict) -> dict:
    answers = row["gold"]["candidate_ids"]
    episode = {
        "id": f"factory-{row['id']}",
        "state": row["state"],
        "question": row["question"],
        "candidates": row["candidates"],
        "answer": answers[0] if len(answers) == 1 else None,
        "acceptable_answers": answers if len(answers) > 1 else None,
        "preference": None,
        "evidence": row["state"],
        "family": BV2.CAPABILITY_TO_FAMILY[row["capability"]],
        "lang": row["language"],
        "origin": f"factory:{row['provenance']['source_id']}",
        "generator_seed": 20261013,
        "generator_version": row["provenance"]["revision"],
        "variant_group": f"factory-{row['group_id']}",
        "schema_version": EC.SCHEMA_VERSION,
        "capability": row["capability"],
        "domain": row["domain"],
        "source_provenance": row["provenance"],
        "split": "train",
    }
    problems = EC.validate(episode)
    if problems:
        raise ValueError(f"{row.get('id')}: {'; '.join(problems)}")
    return episode


def load_training(path: str | Path | None = None) -> list[dict]:
    source = Path(path) if path else DEFAULT_TRAIN
    if not source.is_file():
        return []
    rows = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(convert(json.loads(line)))
    return rows
