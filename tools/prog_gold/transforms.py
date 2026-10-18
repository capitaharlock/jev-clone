"""Consistency transforms (§105) + counterfactual pairs / adversarial context (§§56-57).

Every transform keeps `parent_example_id` and carries a WEIGHT, because an
unweighted transform is how a corpus ends up training mostly on its own
paraphrases. `TRANSFORM_WEIGHTS` is that budget: a base row counts 1.0, a
shuffled or paraphrased copy a fraction of it, and `transform_share`
reports the mass actually spent so the gate can hold it under a cap.

A transform never changes the answer. A COUNTERFACTUAL does, and that is
why it is not a transform but a pair: the question is re-asked about a
value the graph does NOT hold for the entity, so the flipped answer is
still graph-derived. Both halves carry the same parent and the same split,
so a model cannot meet one half in train and the other in eval.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tools.data_factory.teachers_local import LocalTeacher  # noqa: E402

_REDACTOR = LocalTeacher("qwen")

#: §105 — controlled weight per row variant. Base rows anchor at 1.0.
TRANSFORM_WEIGHTS = {
    "base": 1.0,
    "option-shuffle": 0.3,
    "paraphrase": 0.3,
    "counterfactual": 0.5,
    "irrelevant-context": 0.2,
}
#: transform mass may not exceed this share of total mass (§105)
MAX_TRANSFORM_SHARE = 0.35

#: §105 consistency transforms: same answer, different surface. A
#: COUNTERFACTUAL is not one of these — it changes the answer, so it is a
#: new graph-verified example (spec bullet (e)) and not surface variation
#: of an existing one. Only these three are charged to the transform budget.
CONSISTENCY_VARIANTS = ("option-shuffle", "paraphrase", "irrelevant-context")

# ~2k-token irrelevant filler, built once and reused (§57 robustness probe)
FILLER_SENTENCE = (
    "El inventario anual del archivo municipal registra legajos, sellos y actas "
    "de sesiones ordinarias con sus correspondientes anexos numerados. "
)


def weight_of(variant: str) -> float:
    return TRANSFORM_WEIGHTS.get(variant, TRANSFORM_WEIGHTS["base"])


def transform_share(variants: list[str]) -> float:
    """Share of total weight mass spent on §105 consistency transforms."""
    total = sum(weight_of(v) for v in variants)
    if total <= 0:
        return 0.0
    spent = sum(weight_of(v) for v in variants if v in CONSISTENCY_VARIANTS)
    return spent / total


def filler(n_tokens: int = 2000) -> str:
    words = len(FILLER_SENTENCE.split())
    reps = max(1, math.ceil(n_tokens / words))
    return " ".join([FILLER_SENTENCE.strip()] * reps)


def shuffle_options(options: list[dict], rng: random.Random) -> list[dict]:
    out = list(options)
    rng.shuffle(out)
    return out


def paraphrase_state(state: str, key: str) -> tuple[str, str]:
    """Teacher redacts wording only. Returns (new_state, redactor_name)."""
    return _REDACTOR.paraphrase(state, key), _REDACTOR.name


def irrelevant_context(state: str, n_tokens: int = 2000) -> str:
    """§57 — bury the state in `n_tokens` of unrelated prose. Answer unchanged."""
    return state + "\n\n[Contexto adicional: " + filler(n_tokens) + "]"


def counterfactual_pair(question: dict, gold_value: str, false_value: str,
                        false_label: str, qid: str) -> dict:
    """The false half of a counterfactual pair, built from a graph-true boolean.

    `false_value` must be a value the graph does NOT hold for this entity and
    property — the caller checks that against the store, and `graph.verify_record`
    re-checks it on the built record through the `negated` flag.
    """
    gold = question["gold"]
    flipped = dict(question)
    flipped["id"] = qid
    flipped["question"] = question["question"].replace(
        question["_subject_label"], false_label, 1)
    flipped["answer"] = "no"
    flipped["gold"] = {"kind": "graph", "entity": gold["entity"],
                       "prop": gold["prop"], "value": false_value, "negated": True}
    flipped["counterfactual_of"] = question["id"]
    flipped["_subject_label"] = false_label
    flipped["variant"] = "counterfactual"
    flipped["weight"] = weight_of("counterfactual")
    assert false_value != gold_value, "a counterfactual must change the value"
    return flipped
