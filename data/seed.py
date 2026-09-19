"""Seed registry: P0 datasets + eval-only firewall list (#T-data-schema).

Revisions/hashes are placeholders until #T-data-p0 pins the real
immutable revisions — `register()` refuses empty values, so no run
can train against an unpinned card.
"""
from __future__ import annotations

from .registry import DatasetCard, Registry

# Eval-only benchmarks: must NEVER enter train (see source-register.md).
EVAL_ONLY_BENCHMARKS = [
    "MMLU-Pro",
    "GPQA",
    "SimpleQA",
    "MuSR",
    "RewardBench 2",
    "ARC Challenge",
    "OpenBookQA",
]

SEED_CARDS = [
    DatasetCard(
        id="huffpost",
        source_original="Kaggle rmisra/news-category-dataset",
        mirror="khalidalt/HuffPost",
        license="CC0",
        usage="train",
        revision="TBD-pin-in-T-data-p0",
        sha256="0" * 64,
        transform="state=headline+description; dynamic category subset (smoke only)",
    ),
    DatasetCard(
        id="banking77",
        source_original="github.com/PolyAI-LDN/task-specific-datasets",
        mirror="PolyAI/banking77",
        license="CC-BY-4.0",
        usage="train",
        revision="TBD-pin-in-T-data-p0",
        sha256="0" * 64,
        transform="utterance -> intent among dynamic labels",
    ),
    DatasetCard(
        id="boolq",
        source_original="github.com/google-research-datasets/boolean-questions",
        mirror="google/boolq",
        license="CC-BY-SA-3.0",
        usage="train",
        revision="TBD-pin-in-T-data-p0",
        sha256="0" * 64,
        transform="passage+question -> yes/no (boolean)",
    ),
    DatasetCard(
        id="civil-comments",
        source_original="tensorflow.org/datasets/catalog/civil_comments",
        mirror="google/civil_comments",
        license="CC0-1.0",
        usage="train",
        revision="TBD-pin-in-T-data-p0",
        sha256="0" * 64,
        transform="comment -> boolean toxicity questions (scores reserved for V2)",
    ),
    DatasetCard(
        id="helpsteer2",
        source_original="nvidia/HelpSteer2",
        mirror="nvidia/HelpSteer2",
        license="CC-BY-4.0",
        usage="train",
        revision="TBD-pin-in-T-data-p0",
        sha256="0" * 64,
        transform="prompt+response -> five 0-4 choice questions / pairwise",
    ),
    DatasetCard(
        id="clinc150",
        source_original="original vendor source (see card in T-gold)",
        mirror="DeepPavlov/clinc150",
        license="unclear",
        usage="eval-only",
        revision="TBD-pin-in-T-firewall",
        sha256="0" * 64,
        transform="holdout for semantic transfer + OOD; never train",
    ),
]


def seed_registry() -> Registry:
    reg = Registry()
    for card in SEED_CARDS:
        reg.register(card)
    return reg
