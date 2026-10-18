"""The §86 layer plan for `decision-mix-clean-1m` (#T-mix-1m).

The registry — which corpus belongs to which family, origin, language
scope and §86 layer — lives in `data/mix.py` and ONLY there. What lives
here is the strategy's *plan*: the layer proportions §86 asks for, the
§48 question-type mix, and the arithmetic that turns a layer plan into
the per-source pull `data.mix.allocate()` water-fills.

A layer with no clean supply is not silently dropped: `layer_report()`
publishes its target, its realised share and the gap, so "the corpus has
no NLI and no preference data once the benchmark fence is applied" is a
number in the gate instead of an absence nobody counted.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Every stratum axis is the product loader's, so a row is tagged the same
# way wherever it is counted (#T-corpus-rebalance).
from data.mix import (  # noqa: E402,F401 (LAYER_TARGETS/layer_weights re-exported)
    LAYER_TARGETS,
    SOURCES,
    is_hard,
    k_bucket,
    lang_of,
    layer_of,
    layer_weights,
    layers,
    question_type,
)

#: §48 question-type mix.
TYPE_TARGETS = {"choice": 0.55, "noul": 0.30, "score": 0.15}


def layer_supply(supply: dict) -> dict:
    """Trainable questions per §86 layer."""
    out: dict = {}
    for dataset, n in supply.items():
        out[layer_of(dataset)] = out.get(layer_of(dataset), 0) + n
    return out


def _gap_report(counts: dict, total: int, targets: dict, kind: str) -> dict:
    rows = {}
    for name, target in sorted(targets.items()):
        got = counts.get(name, 0)
        share = got / total if total else 0.0
        rows[name] = {"target": target, "share": round(share, 6),
                      "n": got, "gap": round(share - target, 6)}
    for name, got in sorted(counts.items()):
        if name not in rows:
            share = got / total if total else 0.0
            rows[name] = {"target": 0.0, "share": round(share, 6),
                          "n": got, "gap": round(share, 6)}
    empty = sorted(n for n, r in rows.items() if r["n"] == 0 and r["target"])
    return {"kind": kind, "rows": rows, "total": total,
            "max_abs_gap": round(max((abs(r["gap"]) for r in rows.values()),
                                     default=0.0), 6),
            "unsupplied": empty}


def layer_report(counts: dict) -> dict:
    """Realised §86 layer composition against the plan."""
    rep = _gap_report(counts.get("layer", {}), counts.get("rows", 0),
                      LAYER_TARGETS, "§86 layer")
    rep["why_unsupplied"] = {
        "nli": ("no NLI corpus is present in this cut of the registry; "
                "the full one carries `snli` (550 k premise/hypothesis "
                "pairs, CC-BY-SA-4.0)"),
        "preference": ("helpsteer2, the registry's only ordinal corpus, is "
                       "behind the §§18/77 fence; `detox-attack` replaces "
                       "it with annotator-agreement gold that is countable "
                       "from its raw TSV"),
        "adversarial": ("no adversarial/OOD corpus is registered as its own "
                        "layer; the OOD items under "
                        "artifacts/gates/T-data-eval/ are a sealed eval "
                        "probe, not training supply. The §68 hard slice is "
                        "carried INSIDE dbpedia14/goemotions instead, as "
                        "wide option sets built from the measured nearest "
                        "labels of each gold"),
    }
    rep["why_unsupplied"] = {k: v for k, v in rep["why_unsupplied"].items()
                             if k in rep.get("unsupplied", [])}
    return rep


def type_report(counts: dict) -> dict:
    """Realised §48 question-type mix against Choice 55 / Noul 30 / Score 15."""
    rep = _gap_report(counts.get("qtype", {}), counts.get("rows", 0),
                      TYPE_TARGETS, "§48 question type")
    rep["why_unsupplied"] = {
        "score": ("no ordinal corpus is present in this cut: helpsteer2 "
                  "is fenced out by §§18/77 and `detox-attack` — whose "
                  "gold is the measured share of ~10 annotators on a 0-4 "
                  "scale — is the registry's replacement for it"),
    }
    rep["why_unsupplied"] = {k: v for k, v in rep["why_unsupplied"].items()
                             if k in rep.get("unsupplied", [])}
    return rep


def tag(dataset: str, state: str, question: dict) -> dict:
    """The full stratum tuple of one selected question."""
    source = SOURCES[dataset]
    option_texts = [o.get("text", "") for o in question.get("options", [])]
    return {
        "dataset": dataset,
        "family": source.family,
        "layer": layer_of(dataset),
        "qtype": question_type(question.get("kind", ""), option_texts),
        "lang": lang_of(state, source.lang_scope),
        "k": len(option_texts),
        "k_bucket": k_bucket(len(option_texts)),
        "origin": source.origin,
        "hard": is_hard(question, source.layer),
    }
