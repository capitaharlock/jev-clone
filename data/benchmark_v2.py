"""Adapter for Decision Dataset Factory benchmark v2.

The benchmark stays owned by the independent factory. This module converts
its release contract into episode-v1 in memory so TinyJev can score it without
copying evaluation rows into any training pool.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from data import episode_contract as EC

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = Path(os.environ.get(
    "BENCHMARK_V2_PATH",
    ROOT.parent / "decision-dataset-factory" / "work" /
    "benchmark-v2" / "dev.jsonl",
))

CAPABILITY_TO_FAMILY = {
    "evidence_retrieval": EC.EXTRACTION,
    "attribute_comparison": EC.COMPARISON,
    "constrained_priority": EC.PRIORITY,
    "entailment_negation": EC.INFERENCE,
    "semantic_classification": EC.DESCRIPTION,
    "quantitative_reasoning": EC.COMPARISON,
    "multi_hop_reasoning": EC.INFERENCE,
    "causal_counterfactual": EC.INFERENCE,
    "temporal_spatial": EC.COMPARISON,
    "uncertainty_abstention": EC.INFERENCE,
    "code_correctness": EC.INFERENCE,
    "code_quality": EC.PRIORITY,
    "technical_answers": EC.PRIORITY,
    "knowledge": EC.INFERENCE,
    "science_reasoning": EC.INFERENCE,
    "inference_facts": EC.INFERENCE,
    "mathematics": EC.COMPARISON,
    "product_specs": EC.COMPARISON,
    "abstention": EC.INFERENCE,
}


def convert(row: dict) -> dict:
    capability = row["capability"]
    answers = row["gold"]["candidate_ids"]
    episode = {
        "id": row["id"],
        "state": row["state"],
        "question": row["question"],
        "candidates": row["candidates"],
        "answer": answers[0] if len(answers) == 1 else None,
        "acceptable_answers": answers if len(answers) > 1 else None,
        "preference": None,
        # The factory verifier justifies gold. episode-v1 requires a literal
        # state fragment, so the full state is the lossless evidence span.
        "evidence": row["state"],
        "family": CAPABILITY_TO_FAMILY[capability],
        "lang": row["language"],
        "origin": "benchmark-v2-eval-only",
        "generator_seed": 20261013,
        "generator_version": row["provenance"]["revision"],
        "variant_group": row["group_id"],
        "schema_version": EC.SCHEMA_VERSION,
        "capability": capability,
        "domain": row["domain"],
        "difficulty": row.get("difficulty"),
        "source_provenance": row["provenance"],
        "split": "eval-only",
    }
    problems = EC.validate(episode)
    if problems:
        raise ValueError(f"{row.get('id')}: {'; '.join(problems)}")
    return episode


def load_battery(path: str | Path | None = None,
                 limit_per_capability: int = 0) -> list[dict]:
    source = Path(path) if path else DEFAULT_PATH
    counts, episodes = {}, []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            capability = row["capability"]
            if limit_per_capability and counts.get(capability, 0) >= \
                    limit_per_capability:
                continue
            episodes.append(convert(row))
            counts[capability] = counts.get(capability, 0) + 1
    return episodes


def battery_sha(path: str | Path | None = None) -> str:
    source = Path(path) if path else DEFAULT_PATH
    return hashlib.sha256(source.read_bytes()).hexdigest()
