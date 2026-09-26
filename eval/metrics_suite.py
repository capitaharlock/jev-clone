"""The metric suite every pilot gate publishes — #T-battery-metrics.

One number, chosen after the fact, can be made to look like anything. The
defence is not a better number: it is a FIXED set of figures that travel
together, so the cut that flatters one of them shows up in another.

`report()` is the single reporting function. No gate of the pilot computes
its own accuracy any more; it hands rows here and publishes what comes
back. Everything the task fixed is in that return value:

* **Two accuracies, never one.** `ranking` is the forced choice — the
  argmax over the offered candidates, `unknown` taken out of the race — and
  `abstention` is the published decision over `[K + 1]`. A head that
  abstains on 87.7 % of the rows and gets 0/1000 publishes an accuracy of
  0.0 beside a forced 0.009: the pair is what makes the 0.0 readable as a
  threshold artefact instead of as prudence. Ranking and abstention are
  measured apart because they are different properties.
* **Coverage and precision among the answered.** Precision alone rewards
  answering less; coverage alone rewards answering more.
* **Macro by family.** A global mean passes with one family at zero.
* **Chance beside every figure** (R2), computed from the K that figure was
  measured at, plus the per-K table.
* **Joint success by `variant_group`.** A counterfactual pair counts only
  when BOTH halves are right; getting one of the two is a 0, because a head
  that answers the base case from a prior gets one of the two for free.
* **Permutation invariance**, carried over from the `#T-option-text`
  control and kept EXPLICITLY distinct from state/question tracking: the
  2026-09-24 measurement separated them — the scores follow the option text
  under permutation and the model still misses the decisive change, so a
  green permutation check is not evidence of tracking and never stands in
  for one.
* **NLL, Brier and calibration**, with the temperature fitted on a
  development cut and VERIFIED on the test cut — a calibration that never
  left the cut it was fitted on is a fit, not a calibration.

Every published figure carries its `n`, its `cardinality`, its `chance` and
its 95 % Wilson interval. `check()` is the runner's rule: a report missing
any mandatory metric, or the chance of its K, or the n of its cut, FAILS —
it does not warn. Same for a `reading` that contradicts the numbers it sits
next to (`reading_errors()`, wired into `eval.gate_rules` as C6).

Softmax weights are relative to the candidates offered. They are never
published as an absolute probability that an answer is true, and `check()`
refuses a report that renames them into one.

CLI (stdlib only, no torch):

    python3 -m eval.metrics_suite check <report.json>
    python3 -m eval.metrics_suite reading <gate.json>
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.calib import brier, ece, nll, softmax, wilson_interval  # noqa: E402

FORMAT = "jev.metrics.v1"

#: What a softmax weight is, stamped on every report so no reader has to
#: guess. Renaming it into an absolute probability is a `check()` error.
SOFTMAX_SEMANTICS = (
    "softmax weights are RELATIVE to the candidates offered on that row: "
    "they redistribute over the K columns that were shown and say nothing "
    "about the probability that an answer is true in the world. Offering a "
    "different candidate set moves every weight"
)
#: Keys that would turn that relative weight into an absolute claim.
FORBIDDEN_KEYS = frozenset({
    "probability_of_truth", "p_true", "absolute_probability",
    "prob_correct_absolute", "truth_probability",
})

#: The sections `report()` always emits and `check()` always demands.
MANDATORY_SECTIONS = (
    "cut", "chance", "ranking", "abstention", "by_family", "macro",
    "counterfactual", "permutation_invariance", "tracking", "calibration",
    "softmax_semantics",
)
#: The companions R2 demands beside any published accuracy.
FIGURE_KEYS = ("n", "cardinality", "chance", "accuracy", "accuracy_ci95")


class IncoherentReport(ValueError):
    """Raised by `require()`: a report that may not be published."""


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _round(value, digits: int = 6):
    return None if value is None else round(value, digits)


# ------------------------------------------------------------------- rows

def _row(raw) -> dict:
    """One scored row, normalised. Missing fields are errors, not defaults."""
    missing = [f for f in ("row_id", "family", "variant_group", "k",
                           "gold_index", "pred", "probs") if f not in raw]
    if missing:
        raise IncoherentReport(
            f"row {raw.get('row_id', '?')!r} is missing {missing}: a row with "
            "no family or no variant_group cannot be macro-averaged or paired")
    k = int(raw["k"])
    if k < 2:
        raise IncoherentReport(
            f"row {raw['row_id']!r} offers K={k}: a decision needs 2 options")
    probs = list(raw["probs"])
    if len(probs) not in (k, k + 1):
        raise IncoherentReport(
            f"row {raw['row_id']!r} has {len(probs)} weights for K={k}: "
            "expected K (no `unknown` column) or K + 1")
    gold = raw["gold_index"]
    if gold is not None and not 0 <= int(gold) < k:
        raise IncoherentReport(
            f"row {raw['row_id']!r} has gold_index {gold} outside [0, {k})")
    return {
        "row_id": str(raw["row_id"]),
        "family": str(raw["family"]),
        "variant_group": str(raw["variant_group"]),
        "k": k,
        "gold_index": None if gold is None else int(gold),
        "pred": int(raw["pred"]),
        "probs": probs,
        "answerable": gold is not None,
    }


def _forced(row: dict) -> int:
    """Argmax over the OFFERED candidates — `unknown` out of the race."""
    probs = row["probs"][:row["k"]]
    return max(range(row["k"]), key=lambda j: probs[j])


def _abstained(row: dict) -> bool:
    return row["pred"] >= row["k"]


def _decided_right(row: dict) -> bool:
    """The published decision over `[K + 1]`, abstentions included."""
    if _abstained(row):
        return not row["answerable"]
    return row["answerable"] and row["pred"] == row["gold_index"]


def _forced_right(row: dict) -> bool:
    return row["answerable"] and _forced(row) == row["gold_index"]


# ---------------------------------------------------------------- figures

def figure(hits: int, n: int, chance: float | None, cardinality,
           what: str) -> dict:
    """A published number, with the four companions it may never travel
    without: its n, its K, its chance and its 95 % Wilson interval."""
    lo, hi = wilson_interval(hits, n) if n else (0.0, 1.0)
    return {
        "what": what,
        "n": n,
        "hits": hits,
        "cardinality": cardinality,
        "chance": _round(chance),
        "accuracy": _round(hits / n) if n else None,
        "accuracy_ci95": [_round(lo), _round(hi)],
        "beats_chance": bool(n and chance is not None and lo > chance),
        "beats_chance_decided_by": "accuracy_ci95[0] > chance",
    }


def chance_table(rows: list) -> dict:
    """Chance by K, and the two cut-level rates a mixed-K cut has.

    `chance` is the uniform guesser over the OFFERED candidates, which is
    what every accuracy in this file is compared against (the convention
    `eval/fullspace.py` already publishes). `chance_including_unknown` is
    the same guesser when `unknown` is one more column to pick — published
    beside it because on a cut where abstention is the right answer for
    some rows the two are not the same number.
    """
    by_k: dict = {}
    for row in rows:
        by_k.setdefault(str(row["k"]), {"k": row["k"], "rows": 0})
        by_k[str(row["k"])]["rows"] += 1
    for entry in by_k.values():
        entry["chance"] = _round(1.0 / entry["k"])
        entry["chance_including_unknown"] = _round(1.0 / (entry["k"] + 1))
    n = len(rows)
    return {
        "by_k": dict(sorted(by_k.items(), key=lambda kv: kv[1]["k"])),
        "chance": _round(sum(1.0 / r["k"] for r in rows) / n) if n else None,
        "chance_including_unknown":
            _round(sum(1.0 / (r["k"] + 1) for r in rows) / n) if n else None,
        "mean_k": _round(sum(r["k"] for r in rows) / n, 4) if n else None,
        "k_min": min((r["k"] for r in rows), default=None),
        "k_max": max((r["k"] for r in rows), default=None),
        "how": "chance is the uniform guesser over the offered candidates, "
               "averaged per row; a mixed-K cut has no single K, so mean_k "
               "and the by_k table are published together",
    }


def _cardinality(rows: list):
    ks = {r["k"] for r in rows}
    if len(ks) == 1:
        return next(iter(ks))
    return _round(sum(r["k"] for r in rows) / len(rows), 4) if rows else None


def _chance_of(rows: list):
    return (_round(sum(1.0 / r["k"] for r in rows) / len(rows))
            if rows else None)


# ------------------------------------------- ranking / abstention, apart

def ranking_section(rows: list) -> dict:
    """The forced choice. Abstention is NOT in this number."""
    answerable = [r for r in rows if r["answerable"]]
    hits = sum(1 for r in answerable if _forced_right(r))
    fig = figure(hits, len(answerable), _chance_of(answerable),
                 _cardinality(answerable),
                 "forced accuracy: argmax over the offered candidates, "
                 "`unknown` taken out of the race")
    fig["measured_on"] = ("the answerable rows only — a row whose gold was "
                          "withheld has no correct option to rank")
    fig["excluded_unanswerable_rows"] = len(rows) - len(answerable)
    fig["measurable"] = bool(answerable)
    if not answerable:
        fig["why_unmeasurable"] = ("no answerable row on this cut: there is "
                                   "no forced choice to score, and a K and a "
                                   "chance rate are not invented for it")
    return fig


def abstention_section(rows: list) -> dict:
    """The published decision, plus coverage and precision among answered.

    An abstention rate is not a verdict on its own. 87.7 % abstention beside
    0/1000 answered correctly is a random ranking hidden behind a threshold,
    and the only way to see that is to read coverage, precision and the
    forced accuracy at the same time.
    """
    n = len(rows)
    answered = [r for r in rows if not _abstained(r)]
    hits = sum(1 for r in rows if _decided_right(r))
    fig = figure(hits, n, _chance_of(rows), _cardinality(rows),
                 "accuracy including abstention: the published decision "
                 "over [K + 1], an abstention on an answerable row counted "
                 "as wrong")
    precision_hits = sum(1 for r in answered if _decided_right(r))
    plo, phi = (wilson_interval(precision_hits, len(answered))
                if answered else (0.0, 1.0))
    clo, chi = wilson_interval(len(answered), n) if n else (0.0, 1.0)
    fig.update({
        "abstain_rate": _round(1.0 - len(answered) / n) if n else None,
        "coverage": {
            "what": "share of rows the head answered",
            "n": n, "answered": len(answered),
            "cardinality": _cardinality(rows), "chance": _chance_of(rows),
            "coverage": _round(len(answered) / n) if n else None,
            "coverage_ci95": [_round(clo), _round(chi)],
        },
        "precision_among_answered": {
            "what": "accuracy restricted to the rows the head answered",
            "n": len(answered), "hits": precision_hits,
            "cardinality": _cardinality(answered),
            "chance": _chance_of(answered),
            "accuracy": (_round(precision_hits / len(answered))
                         if answered else None),
            "accuracy_ci95": [_round(plo), _round(phi)],
            "beats_chance": bool(answered and _chance_of(answered) is not None
                                 and plo > _chance_of(answered)),
            "beats_chance_decided_by": "accuracy_ci95[0] > chance",
            "why_not_alone": "precision among the answered rises by "
                             "answering less; it is only readable beside "
                             "coverage",
        },
        "unanswerable_rows": {
            "n": sum(1 for r in rows if not r["answerable"]),
            "correctly_abstained": sum(1 for r in rows
                                       if not r["answerable"]
                                       and _abstained(r)),
        },
    })
    return fig


# ------------------------------------------------------ macro by family

def family_section(rows: list) -> tuple:
    """Per family, both accuracies; and the macro mean of each.

    The global mean is a weighted average, so a family at zero disappears
    into a bigger one. The macro mean gives every family the same vote and
    is the figure that notices.
    """
    families: dict = {}
    for row in rows:
        families.setdefault(row["family"], []).append(row)
    by_family = {}
    for name in sorted(families):
        group = families[name]
        answerable = [r for r in group if r["answerable"]]
        by_family[name] = {
            "ranking": figure(
                sum(1 for r in answerable if _forced_right(r)),
                len(answerable), _chance_of(answerable),
                _cardinality(answerable), f"forced accuracy, family {name}"),
            "abstention": figure(
                sum(1 for r in group if _decided_right(r)), len(group),
                _chance_of(group), _cardinality(group),
                f"accuracy including abstention, family {name}"),
        }

    def mean(section: str, key: str = "accuracy"):
        vals = [f[section][key] for f in by_family.values()
                if f[section][key] is not None]
        return _round(sum(vals) / len(vals)) if vals else None

    macro = {
        "what": "unweighted mean over families — every family one vote",
        "n_families": len(by_family),
        "n": len(rows),
        "cardinality": _cardinality(rows),
        "chance": mean("abstention", "chance"),
        "accuracy": mean("abstention"),
        "accuracy_ranking": mean("ranking"),
        "chance_ranking": mean("ranking", "chance"),
        "worst_family": (min(by_family, key=lambda f:
                             by_family[f]["abstention"]["accuracy"]
                             if by_family[f]["abstention"]["accuracy"]
                             is not None else 1.0)
                         if by_family else None),
        "accuracy_ci95": None,
        "ci95_why_absent": "a macro mean over families is not a binomial "
                           "proportion: its interval is each family's own, "
                           "published in by_family, not a Wilson bound here",
    }
    return by_family, macro


# ------------------------------------------------- counterfactual pairing

def counterfactual_section(rows: list) -> dict:
    """Joint success by `variant_group`: both halves right, or zero.

    The base case of a counterfactual pair is answerable from a prior — the
    variant is the one that needs the state read. Scoring them separately
    hands out half the credit for free, so a group counts as a success only
    when EVERY member of it is right, and a group where exactly one half is
    right scores 0 and is reported as a `one_half_only`.
    """
    groups: dict = {}
    for row in rows:
        groups.setdefault(row["variant_group"], []).append(row)
    paired = {g: rs for g, rs in groups.items() if len(rs) >= 2}
    singles = sorted(g for g, rs in groups.items() if len(rs) < 2)

    joint = half = 0
    chances = []
    for members in paired.values():
        right = sum(1 for r in members if _decided_right(r))
        if right == len(members):
            joint += 1
        elif right:
            half += 1
        product = 1.0
        for r in members:
            product *= 1.0 / r["k"]
        chances.append(product)
    n = len(paired)
    chance = _round(sum(chances) / n) if n else None
    lo, hi = wilson_interval(joint, n) if n else (0.0, 1.0)
    return {
        "what": "joint success by variant_group: every member of the group "
                "right, or the group scores 0",
        "n": n,
        "hits": joint,
        "groups_with_one_half_only": half,
        "cardinality": _cardinality([r for rs in paired.values() for r in rs]),
        "chance": chance,
        "chance_how": "product of the per-member uniform rates, averaged "
                      "over groups — guessing both halves of a pair",
        "accuracy": _round(joint / n) if n else None,
        "accuracy_ci95": [_round(lo), _round(hi)],
        "beats_chance": bool(n and chance is not None and lo > chance),
        "beats_chance_decided_by": "accuracy_ci95[0] > chance",
        "measurable": bool(n),
        "why_unmeasurable": (None if n else
                             "this cut carries no variant_group with two or "
                             "more members: joint success has nothing to "
                             "score and no chance rate is invented for it"),
        "unpaired_groups": len(singles),
        "unpaired_group_ids": singles[:8],
        "rule": "a group with exactly one half right scores 0: half of a "
                "counterfactual pair is what a prior gets for free",
    }


# ------------------------------- permutation invariance vs. tracking

def _control(block, what: str, distinct_from: str) -> dict:
    """A control measured elsewhere, carried in with its provenance.

    It is never synthesised here: a control this module invented would be a
    number nobody measured. A missing one is left as `measured: false` and
    `check()` fails the report.
    """
    if not isinstance(block, dict) or not block:
        return {"what": what, "measured": False, "distinct_from": distinct_from,
                "why": "no measurement handed to report(): this control is "
                       "made elsewhere and carried in, never invented here"}
    out = {"what": what, "measured": True, "distinct_from": distinct_from}
    out.update(block)
    return out


PERMUTATION_WHAT = (
    "permuting the offered candidates leaves each candidate's weight where "
    "it was, realigned by id — the #T-option-text control"
)
TRACKING_WHAT = (
    "changing the decisive fact in the state, or the question asked of it, "
    "changes the answer"
)
DISTINCT = (
    "these are TWO properties and the 2026-09-24 measurement separates "
    "them: the scores follow the option text under permutation and the "
    "model still misses the decisive change. A green permutation check is "
    "not evidence of tracking and never stands in for one"
)


# ------------------------------------------------------------ calibration

def calibration_section(rows: list, calibration: dict | None) -> dict:
    """NLL, Brier and ECE, at the temperature that was actually fitted.

    `fitted_on` and `verified_on` are carried from the caller and must be
    different cuts: a temperature fitted and reported on the same rows is a
    fit, not a calibration, and `check()` says so.
    """
    scored = [r for r in rows if r["answerable"]]
    temp = 1.0
    if isinstance(calibration, dict) and calibration.get("temperature"):
        temp = float(calibration["temperature"])
    probs, labels = [], []
    for row in scored:
        weights = row["probs"][:row["k"]]
        total = sum(weights)
        share = [w / total if total else 1.0 / row["k"] for w in weights]
        # Re-tempering goes through the log: the weights handed in are
        # already a softmax, and `log p` recovers the logits up to the
        # constant a softmax drops anyway.
        probs.append(share if temp == 1.0
                     else softmax([math.log(max(p, 1e-12)) for p in share],
                                  temp))
        labels.append(row["gold_index"])
    out = {
        "what": "NLL, Brier and ECE over the offered candidates",
        "n": len(scored),
        "cardinality": _cardinality(scored),
        "chance": _chance_of(scored),
        "temperature": temp,
        "nll": _round(nll(probs, labels)) if probs else None,
        "brier": _round(brier(probs, labels)) if probs else None,
        "nll_at_chance": (_round(-math.log(max(_chance_of(scored)
                                                  or 1e-12, 1e-12)))
                          if scored else None),
        "softmax_semantics": SOFTMAX_SEMANTICS,
    }
    if probs:
        bins = ece(probs, labels)
        out["ece"] = _round(bins["ece"])
        out["reliability_bins"] = bins["bins"]
    else:
        out["ece"] = None
    src = calibration if isinstance(calibration, dict) else {}
    out["fitted_on"] = src.get("fitted_on")
    out["verified_on"] = src.get("verified_on")
    out["threshold"] = src.get("threshold")
    out["threshold_strategy"] = src.get("strategy")
    if not src:
        out["why_uncalibrated"] = (
            "no calibration handed to report(): temperature 1.0 is the raw "
            "head, not a calibrated one, and check() fails the report")
    return out


# --------------------------------------------------------- the one entry

def report(rows: list, *, cut: dict, model_version: str | None = None,
           task: str | None = None, calibration: dict | None = None,
           permutation: dict | None = None, tracking: dict | None = None,
           notes: list | None = None) -> dict:
    """THE reporting function. Every pilot gate publishes this and nothing
    it computed itself.

    `rows` are scored rows — see `_row()` for the shape. `cut` describes
    where they came from (name, seal, split); it is copied through with the
    cut's `n` filled in from the rows actually scored, because the n of a
    published figure is the rows behind it and not the rows intended.
    """
    parsed = [_row(r) for r in rows]
    by_family, macro = family_section(parsed)
    cut_out = dict(cut or {})
    cut_out["n"] = len(parsed)
    cut_out.setdefault("rows_scored", len(parsed))
    return {
        "format": FORMAT,
        "artifact": "metrics-suite",
        "task": task,
        "generated_utc": utcnow(),
        "model_version": model_version,
        "cut": cut_out,
        "chance": chance_table(parsed),
        "ranking": ranking_section(parsed),
        "abstention": abstention_section(parsed),
        "reported_apart": "ranking and abstention are two measurements and "
                          "are published side by side: neither is the "
                          "headline on its own",
        "by_family": by_family,
        "macro": macro,
        "counterfactual": counterfactual_section(parsed),
        "permutation_invariance": _control(permutation, PERMUTATION_WHAT,
                                           TRACKING_WHAT),
        "tracking": _control(tracking, TRACKING_WHAT, PERMUTATION_WHAT),
        "controls_are_distinct": DISTINCT,
        "calibration": calibration_section(parsed, calibration),
        "softmax_semantics": SOFTMAX_SEMANTICS,
        "notes": list(notes or []),
    }


# ------------------------------------------------------------ the runner

def _figure_errors(node, where: str) -> list:
    """Every figure carries its n, its K, its chance and its interval.

    The one exemption is a section that DECLARES itself unmeasurable on
    this cut — no answerable row to rank, no `variant_group` with two
    members to pair. It still publishes its n (zero) and says why; what it
    does not do is invent a cardinality and a chance rate for rows that are
    not there. A section that is merely missing its numbers, with no such
    declaration, fails as before.
    """
    out = []
    empty = node.get("measurable") is False and node.get("n") == 0
    for key in FIGURE_KEYS:
        if key not in node:
            out.append({"rule": "C5", "severity": "error", "at": where,
                        "missing": key,
                        "why": f"{where} publishes a figure without its "
                               f"{key}: every figure carries its n, its K, "
                               "its chance and its 95 % interval"})
        elif node[key] is None and key in ("n", "cardinality", "chance") \
                and not empty:
            out.append({"rule": "C5", "severity": "error", "at": where,
                        "missing": key,
                        "why": f"{where} has {key}: null — the n of a cut "
                               "and the chance of its K are not optional"})
    ci = node.get("accuracy_ci95")
    if not (isinstance(ci, list) and len(ci) == 2
            and all(isinstance(v, (int, float)) for v in ci)):
        out.append({"rule": "C5", "severity": "error", "at": where,
                    "missing": "accuracy_ci95",
                    "why": f"{where} has no two-ended 95 % interval"})
    return out


def check(doc: dict) -> list:
    """The rule: a report missing a mandatory metric FAILS, it never warns.

    Returns the errors; empty means publishable. `require()` raises.
    """
    errors = []
    if not isinstance(doc, dict):
        return [{"rule": "C5", "severity": "error", "at": "/",
                 "why": "not an object"}]
    if doc.get("format") != FORMAT:
        errors.append({"rule": "C5", "severity": "error", "at": "/format",
                       "why": f"not a {FORMAT} report: the suite is the only "
                              "place a pilot figure is computed"})
    for section in MANDATORY_SECTIONS:
        if section not in doc or doc[section] in (None, {}, ""):
            errors.append({"rule": "C5", "severity": "error",
                           "at": f"/{section}", "missing": section,
                           "why": f"mandatory section `{section}` absent: "
                                  "the suite is published whole or not at all"})
    for section in ("ranking", "abstention", "counterfactual"):
        node = doc.get(section)
        if isinstance(node, dict):
            errors.extend(_figure_errors(node, f"/{section}"))
    abst = doc.get("abstention")
    if isinstance(abst, dict):
        for sub in ("coverage", "precision_among_answered"):
            if not isinstance(abst.get(sub), dict):
                errors.append({"rule": "C5", "severity": "error",
                               "at": f"/abstention/{sub}", "missing": sub,
                               "why": "coverage and precision among the "
                                      "answered are published together or "
                                      "neither is readable"})
    fam = doc.get("by_family")
    if isinstance(fam, dict):
        if not fam:
            errors.append({"rule": "C5", "severity": "error",
                           "at": "/by_family", "missing": "families",
                           "why": "no family breakdown: a global mean passes "
                                  "with one family at zero"})
        for name, node in fam.items():
            for section in ("ranking", "abstention"):
                if isinstance(node, dict) and isinstance(node.get(section), dict):
                    errors.extend(_figure_errors(node[section],
                                                 f"/by_family/{name}/{section}"))
    cut = doc.get("cut")
    if isinstance(cut, dict) and not isinstance(cut.get("n"), int):
        errors.append({"rule": "C5", "severity": "error", "at": "/cut/n",
                       "missing": "n", "why": "the cut publishes no n"})
    ch = doc.get("chance")
    if isinstance(ch, dict) and not ch.get("by_k"):
        errors.append({"rule": "C5", "severity": "error", "at": "/chance/by_k",
                       "missing": "by_k",
                       "why": "no chance-by-K table: R2 wants the chance of "
                              "the K each figure was measured at"})
    for name in ("permutation_invariance", "tracking"):
        node = doc.get(name)
        if isinstance(node, dict) and node.get("measured") is not True:
            errors.append({"rule": "C5", "severity": "error", "at": f"/{name}",
                           "missing": name,
                           "why": f"the {name} control was not measured: it "
                                  "is carried in from where it is measured, "
                                  "and a report without it is incomplete"})
    cal = doc.get("calibration")
    if isinstance(cal, dict):
        for key in ("nll", "brier", "ece"):
            if cal.get(key) is None:
                errors.append({"rule": "C5", "severity": "error",
                               "at": f"/calibration/{key}", "missing": key,
                               "why": "NLL, Brier and calibration error are "
                                      "published together"})
        fitted, verified = cal.get("fitted_on"), cal.get("verified_on")
        if not fitted or not verified:
            errors.append({"rule": "C5", "severity": "error",
                           "at": "/calibration", "missing": "fitted_on/verified_on",
                           "why": "a temperature with no development cut it "
                                  "was fitted on and no cut it was verified "
                                  "on is not a calibration"})
        elif fitted == verified:
            errors.append({"rule": "C5", "severity": "error",
                           "at": "/calibration", "missing": "verification",
                           "why": f"fitted and verified on the same cut "
                                  f"({fitted!r}): that is a fit, not a "
                                  "calibration"})
    for path, key, _ in _walk(doc):
        if key in FORBIDDEN_KEYS:
            errors.append({"rule": "C5", "severity": "error", "at": path,
                           "why": f"`{key}` publishes a softmax weight as an "
                                  "absolute probability of truth; the weight "
                                  "is relative to the offered candidates"})
    if doc.get("softmax_semantics") not in (None, SOFTMAX_SEMANTICS) \
            and "relative" not in str(doc.get("softmax_semantics", "")).lower():
        errors.append({"rule": "C5", "severity": "error",
                       "at": "/softmax_semantics",
                       "why": "the softmax semantics note no longer says the "
                              "weights are relative to the offered candidates"})
    errors.extend(reading_errors(doc))
    return errors


def require(doc: dict) -> dict:
    errors = check(doc)
    if errors:
        raise IncoherentReport(json.dumps(errors, indent=2))
    return doc


# ----------------------------------------------- C6: the reading rule

def _walk(obj, path: str = ""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            here = f"{path}/{key}"
            yield here, key, value
            yield from _walk(value, here)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from _walk(value, f"{path}[{i}]")


#: Keys whose value is prose a human will read as the artifact's verdict.
READING_KEYS = frozenset({"reading", "verdict_line", "summary", "headline",
                          "verdict_reading", "conclusion"})

#: An affirmative claim about where the interval sits relative to chance.
CLAIMS = (
    ("contains_chance", re.compile(
        r"contains? (the )?chance|interval contains it|contiene el azar|"
        r"indistinguishable from chance|indistinguible del azar", re.I)),
    ("below_chance", re.compile(
        r"below (the )?chance|under (the )?chance|por debajo del azar|"
        r"excludes? (the )?chance[^.;]{0,24}?from below|"
        r"excluye el azar por debajo", re.I)),
    ("above_chance", re.compile(
        r"above (the )?chance|beats? chance|clears? (its )?chance|"
        r"supera el azar|por encima del azar", re.I)),
)

#: A sentence names WHICH of the two accuracies it is about, and the rule
#: reads the one it names. `unknown` out of the race is the FORCED figure —
#: this repo calls it `accuracy_options_only` and its verdict flag
#: `ranking_beats_chance` — and the decision over `[K + 1]` is the primary.
#: Conflating them is the exact mistake the rule exists to catch: on
#: #T-fullspace-objective one interval contains chance and the other
#: excludes it from below, in the same artifact, on the same rows.
_FORCED_WORDS = re.compile(
    r"forced|options[ _-]only|ranking|elecci[oó]n forzada|"
    r"out of the race|sin `?unknown`?", re.I)
_PRIMARY_WORDS = re.compile(r"\bprimar(y|ia)\b|published decision", re.I)
#: How far around a claim the rule looks for the word that names its subject.
_SUBJECT_WINDOW = 160
_NEGATOR = re.compile(r"(does not|do not|doesn't|don't|never|not|no|"
                      r"neither|nor|sin|tampoco|ning[uú]n\w*|no)\s+\S*\s*$",
                      re.I)

#: A claim that a hypothesis has been eliminated AS A CAUSE.
_CAUSAL = re.compile(
    r"(discard(?:s|ed)?|rules? out|ruled out|descartad[oa]s?|"
    r"queda descartad[oa])"
    r"[^.;]{0,60}?(as (the )?cause|como causa|as a cause)", re.I)
#: The field in which an artifact declares it establishes no cause (R9).
_NO_CAUSE = re.compile(r"does not establish cause|no establece causa|"
                       r"limits spend and does not establish", re.I)
#: What lifts the R9 bar, and the ONLY thing that does: a structured field
#: saying the arm was actually repeated. Prose does not count — every one of
#: these artifacts already says in words that a causal claim would need
#: another seed, and reading that sentence as evidence of one is how the
#: claim got published in the first place.
REPLICATION_KEYS = ("replicated", "seeds", "n_seeds")


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _figure_of(node) -> dict | None:
    """A readable figure: a chance and a two-ended interval on one object."""
    if not isinstance(node, dict):
        return None
    ci = node.get("accuracy_ci95") or node.get("ci95")
    chance = node.get("chance")
    if (_is_number(chance) and isinstance(ci, list) and len(ci) == 2
            and all(_is_number(v) for v in ci)):
        return {"chance": chance, "ci95": [ci[0], ci[1]],
                "accuracy": node.get("accuracy")}
    return None


def _forced_figure_of(node) -> dict | None:
    """The forced figure of an object: `unknown` taken out of the race."""
    if not isinstance(node, dict):
        return None
    ci = node.get("accuracy_options_only_ci95")
    chance = node.get("chance")
    if (_is_number(chance) and isinstance(ci, list) and len(ci) == 2
            and all(_is_number(v) for v in ci)):
        return {"chance": chance, "ci95": [ci[0], ci[1]],
                "accuracy": node.get("accuracy_options_only")}
    return None


def _primary_nodes(doc) -> list:
    """The objects a sentence means when it says "the primary"."""
    if not isinstance(doc, dict):
        return []
    primary = doc.get("primary")
    out = []
    if isinstance(primary, dict):
        out.append(("/primary", primary))
        for key in ("arm", "dev", "value", "headline"):
            if isinstance(primary.get(key), dict):
                out.append((f"/primary/{key}", primary[key]))
    return out


def _names_which(text: str, span: tuple | None) -> tuple:
    """Which of the two accuracies this claim names, by NEAREST word.

    A sentence mentions both — "the interval that contains chance is the
    forced one, `unknown` out of the race" sits in the same paragraph as
    "the published decision is a threshold on top of a ranking" — so a
    window that merely CONTAINS a keyword picks the wrong subject and the
    rule fires on a correct sentence. The word that names the subject of a
    claim is the one closest to it; ties go to the published decision,
    which is what an unqualified "the interval" means.

    Returns `(forced, named)`: whether the forced figure is the subject,
    and whether the sentence named a subject at all.
    """
    at = span[0] if span else 0
    end = span[1] if span else len(text)

    def nearest(pattern) -> int | None:
        best = None
        for m in pattern.finditer(text):
            if m.start() >= at and m.end() <= end:
                continue  # the claim phrase itself is not its own subject
            d = 0 if at <= m.start() <= end else min(abs(m.start() - end),
                                                     abs(at - m.end()))
            if d <= _SUBJECT_WINDOW and (best is None or d < best):
                best = d
        return best

    f, p = nearest(_FORCED_WORDS), nearest(_PRIMARY_WORDS)
    if f is None and p is None:
        return False, False
    if f is None:
        return False, True
    if p is None:
        return True, True
    return f < p, True


def _subject(doc, chain: tuple, text: str, span: tuple | None = None) -> tuple:
    """Which numbers a claim is talking about — resolved per claim.

    The sentence says which of the two accuracies it means, so the rule
    reads that one: a window around the claim naming the forced choice
    resolves to `accuracy_options_only` and its interval, one naming the
    primary resolves to the published decision. Without either word the
    nearest enclosing object that publishes a figure is the subject, and
    when nothing publishes one the claim is left alone — this rule catches
    an artifact contradicting itself, it does not guess at absent numbers.
    """
    forced, named = _names_which(text, span)
    pick = _forced_figure_of if forced else _figure_of
    tag = "forced (options only)" if forced else "published decision"
    # A suite report names its own two figures: `/ranking` is the forced
    # choice and `/abstention` the published decision, so a reading inside
    # one is checked against the section it is talking about.
    if isinstance(doc, dict) and doc.get("format") == FORMAT:
        section = doc.get("ranking" if forced else "abstention")
        got = _figure_of(section)
        if got:
            return got, f"/{'ranking' if forced else 'abstention'} — {tag}"
    if named:
        for where, node in _primary_nodes(doc):
            got = pick(node)
            if got:
                return got, f"{where} — {tag}"
    for node in reversed(chain):
        got = pick(node)
        if got:
            return got, f"enclosing figure — {tag}"
    return None, None


def _claims_in(text: str) -> list:
    out = []
    for name, pattern in CLAIMS:
        for m in pattern.finditer(text):
            before = text[max(0, m.start() - 28):m.start()]
            out.append({"claim": name, "phrase": m.group(0),
                        "span": (m.start(), m.end()),
                        "negated": bool(_NEGATOR.search(before))})
    return out


def _holds(claim: str, negated: bool, chance: float, ci: list) -> bool:
    lo, hi = ci
    if claim == "contains_chance":
        truth = lo <= chance <= hi
    elif claim == "below_chance":
        truth = hi < chance
    else:
        truth = lo > chance
    return truth != negated


def reading_errors(doc) -> list:
    """C6 — a gate whose `reading` contradicts its own numbers FAILS.

    Two contradictions, both self-contained: the reading says where the
    interval sits relative to chance and the interval sits elsewhere; or
    the reading eliminates a hypothesis AS A CAUSE in an artifact that
    declares in its own R9 field that it establishes none.

    A negated claim ("does not clear chance") is read as its complement and
    checked the same way. A claim whose subject has no published figure is
    left alone: this rule catches artifacts that contradict themselves, it
    does not guess at numbers that are not there.
    """
    errors = []

    def visit(node, path: str, chain: tuple) -> None:
        if isinstance(node, dict):
            chain = chain + (node,)
            for key, value in node.items():
                here = f"{path}/{key}"
                if key in READING_KEYS and isinstance(value, str):
                    errors.extend(_check_reading(doc, node, chain, here, value))
                visit(value, here, chain)
        elif isinstance(node, list):
            for i, value in enumerate(node):
                visit(value, f"{path}[{i}]", chain)

    visit(doc, "", ())
    return errors


def _replicated(chain: tuple) -> bool:
    """Did the artifact DECLARE a replication, in a field and not in prose?"""
    for node in chain:
        for key in REPLICATION_KEYS:
            value = node.get(key)
            if value is True:
                return True
            if _is_number(value) and value >= 2:
                return True
    return False


def _check_reading(doc, node, chain, path: str, text: str) -> list:
    out = []
    for claim in _claims_in(text):
        fig, where = _subject(doc, chain, text, claim["span"])
        if not fig:
            continue
        if not _holds(claim["claim"], claim["negated"], fig["chance"],
                      fig["ci95"]):
            out.append({
                "rule": "C6", "severity": "error", "at": path,
                "reading": text[:240],
                "claim": claim["claim"],
                "negated": claim["negated"],
                "phrase": claim["phrase"],
                "subject": where,
                "accuracy": fig["accuracy"],
                "accuracy_ci95": fig["ci95"],
                "chance": fig["chance"],
                "why": "the reading states where the interval sits "
                       "relative to chance and its own numbers put it "
                       "somewhere else — a gate whose reading "
                       "contradicts its numbers FAILS, it does not warn",
            })
    if _CAUSAL.search(text):
        declares_no_cause = any(
            isinstance(v, str) and _NO_CAUSE.search(v)
            for o in chain for v in o.values())
        if declares_no_cause and not _replicated(chain):
            out.append({
                "rule": "C6", "severity": "error", "at": path,
                "reading": text[:240],
                "claim": "cause_discarded",
                "why": "the reading eliminates a hypothesis AS A CAUSE in an "
                       "artifact whose own R9 field says it establishes no "
                       "cause: one seed and one budget discard that arm's "
                       "OBSERVED BENEFIT at that budget, not the cause",
            })
    return out


# -------------------------------------------------------------------- CLI

def main(argv: list) -> int:
    ap = argparse.ArgumentParser(prog="eval.metrics_suite")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="run the suite's rule over a report")
    c.add_argument("path")
    r = sub.add_parser("reading", help="C6 only, over any gate artifact")
    r.add_argument("path")
    args = ap.parse_args(argv)

    doc = json.loads(Path(args.path).read_text())
    errors = check(doc) if args.cmd == "check" else reading_errors(doc)
    print(json.dumps({"path": args.path, "n_errors": len(errors),
                      "errors": errors, "pass": not errors}, indent=2,
                     ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
