"""The corpus mix descriptor: one registry, hard caps, a versioned manifest
(#T-corpus-rebalance).

Audit finding E: of 2 681 195 converted examples, `civil-comments` alone is
1 999 514 — **74.6 %** — and it is a fixed-label binary task. It teaches
nothing about choosing among dynamic options, which is the one property the
product sells. The strategy already wrote the guardrails down (a dataset
never over 15 %, a family never over 30 %, "nunca 89 % sin que el pipeline lo
marque como error"); they existed only inside `tools/mix_1m`, which does not
feed `training/python/train_decision.py`. So the trainer never saw them.

This module is where they live now, and it is the ONLY place that decides
what enters the training mixture and in what proportion:

* **The registry** (`SOURCES`) names every trainable corpus with its family,
  origin, language scope and — the axis that matters here — whether its
  options are DYNAMIC (a label pool a row samples K from) or a FIXED binary
  / ordinal scale. The weights reward the first kind.
* **The guardrails are a hard error.** `verify_mix()` raises
  `MixGuardrailError` naming the dataset or family and its exact share. The
  loader calls it on the REALISED composition, after the unseen-label
  holdout has cut rows, so a mixture cannot drift past a cap between the
  plan and the batches.
* **The allocator** is a water-filling pass: shares proportional to the
  product weights, clamped by supply, by the 15 % dataset cap and by the
  30 % family cap, with the remainder spilled to whoever still has headroom.
  A target the caps cannot cover raises `MixInfeasible` — with the arithmetic
  that proves it, because "add more sources" is the only fix.
* **The manifest is versioned and reproducible.** Selection is a stateless
  seeded hash over `(seed, dataset, row_index, question_id)`, so the mixture
  is a pure function of `seed + keep fractions + the per-shard sha256` that
  the manifest records. No member list has to be shipped to rebuild it bit
  for bit; `members_sha256` proves the rebuild matched.

Why sampling is now the rule and not the exception
--------------------------------------------------
Every source carries a keep fraction, including the small ones (theirs is
1.0). `civil-comments` is not special-cased: it lands at the same cap as
everyone else and its keep fraction falls out of the arithmetic. The old
per-job `sample` in `data/train_baseline.py::JOBS` was keyed on
`hash(q["id"])` — a Python hash (salted per process, so not reproducible)
of a question id that `civil-comments` reuses verbatim on all 1 999 514 rows
(`civil-toxicity-all-0`), which means it kept either everything or nothing.
`keep_row()` below is keyed on the ROW and hashed with sha256.

A note on feasibility, because the number is not obvious
--------------------------------------------------------
With a 15 % dataset cap and a 30 % family cap, a family contributes at most
`0.15 * min(2, datasets_in_family)` of any mixture. Covering 100 % therefore
needs `sum over families of min(2, n) >= 1 / 0.15 = 6.67` — seven "cap
units". Drop one source from `SOURCES` and the sum falls to 6: no mixture of
any size satisfies the guardrails, and `build_mix()` says so instead of
quietly rebalancing to 90 %.

Cap units are necessary and not sufficient, which is the finding #T-mix-1m
had to pay for. Seven sources cleared 1.05 cap units and still admitted only
39 970 rows, because a source can only fill its 15 % while it HAS 15 % to
give: `email-triage` has 4 000 questions, so past a 40 000-row mixture its
shortfall ate the entire 5 % of slack. The ceiling is therefore the largest
`T` with `sum_d min(supply_d, 0.15 T) >= T` under the family caps, which is
what `max_feasible_target()` binary-searches — and why the fix was five more
independent corpora (`data/convert_widen.py`), not another seed.

CLI::

    python3 -m data.mix show              # registry + cached supply
    python3 -m data.mix build [--target N] [--seed S]
    python3 -m data.mix verify            # re-check the written manifest
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREFETCH_DIR = os.path.join(ROOT, "artifacts", "data-prefetch")
MIX_DIR = os.path.join(ROOT, "artifacts", "mix")

#: bump when the registry, the caps or the selection function change: a
#: manifest of an older version describes a mixture this code cannot rebuild.
MIX_VERSION = "decision-mix-v3"
MANIFEST_FORMAT = 2
DEFAULT_SEED = 20260921

# -- guardrails (strategy §§65-66) ----------------------------------------
#: no single dataset may exceed this share of the mixture
MAX_DATASET_FRACTION = 0.15
#: no task family may exceed this share of the mixture
MAX_FAMILY_FRACTION = 0.30
#: synthetic origin stays a minority of the mixture
MAX_SYNTHETIC_FRACTION = 0.50
#: human-authored supervision must stay a real presence
MIN_HUMAN_FRACTION = 0.20
#: `tools/mix_1m` §68 only: hard/OOD slice of the assembled stage
MIN_HARD_FRACTION = 0.10
#: the product property: rows offering a dynamic option pool must be MOST
#: of the mixture, not a minority of it (task "Done when", third bullet)
MIN_DYNAMIC_FRACTION = 0.50
#: allocate against a slightly tighter cap, so a source whose supply grows
#: between the plan and the run still lands inside the real cap
CAP_SAFETY_MARGIN = 0.0075
#: the hash-keep is binomial around its quota, and a mixture that is 0.3 pp
#: over a hard cap is a failed run. So every source is drawn with this much
#: headroom and then TRIMMED to its quota in file order: the realised
#: composition equals the planned one exactly, at every mixture size.
DRAW_HEADROOM = 1.15


class MixGuardrailError(ValueError):
    """A hard cap is breached: mix construction aborts, never warns."""


class MixInfeasible(ValueError):
    """No mixture of this size satisfies the caps with this supply."""


# -- the registry ----------------------------------------------------------

@dataclass(frozen=True)
class Source:
    """One trainable corpus and everything the caps need to know about it."""

    id: str
    family: str
    origin: str          # human | programmatic | synthetic
    lang_scope: str      # en | multi | <locale>
    #: True when a row's options are drawn from a label pool (the product
    #: property: K varies, the option TEXT is what gets scored). False for a
    #: fixed binary or an ordinal scale, where the option set IS the task.
    dynamic_options: bool
    #: relative pull before the caps bite; see `PRODUCT_WEIGHT`
    weight: float
    #: the §86 layer this corpus belongs to, so the composition of
    #: `decision-mix-clean-1m` is countable against the strategy's plan
    #: instead of being re-declared in a second registry (`tools/mix_1m`).
    layer: str = ""
    shards: tuple = ()
    note: str = ""
    #: an experimental source is registered (so its shards, family and caps
    #: are described in exactly one place) but NEVER joins a default scan:
    #: `default_datasets()` leaves it out, and only an explicit selection
    #: (`train_decision --only-dataset`) trains on it. Registering an
    #: experiment must not move a single byte of a running or queued run.
    experimental: bool = False

    def paths(self, root: str = PREFETCH_DIR) -> list:
        """Shards of this source under `root`.

        The declared `shards` are paths of the real corpus; any OTHER root
        (a test fixture, a copy of the corpus elsewhere) is read as one
        `<id>.jsonl` per source, so a fixture never has to reproduce a
        source's file layout to stand in for it.
        """
        if self.shards and os.path.abspath(root) == PREFETCH_DIR:
            return [os.path.join(ROOT, s) for s in self.shards]
        return [os.path.join(root, f"{self.id}.jsonl")]


#: weight by how much a corpus teaches "point at the right option text".
#: A large dynamic pool (tens of labels, hard siblings) is the real thing; a
#: four-label dynamic task is a weaker version of it; a fixed binary or an
#: ordinal scale teaches the calibration head, not the pointer.
PRODUCT_WEIGHT = {"dynamic_large": 1.0, "dynamic_small": 0.7, "fixed": 0.35}

SOURCES = {
    "banking77": Source(
        "banking77", "intent", "human", "en", True,
        PRODUCT_WEIGHT["dynamic_large"], layer="intent-multi",
        note="77 sibling intents — the hardest dynamic pool in the corpus"),
    "massive": Source(
        "massive", "intent", "human", "multi", True,
        PRODUCT_WEIGHT["dynamic_large"], layer="intent-multi",
        note="60 intents across 4 locales; carries the multilingual axis"),
    "email-triage": Source(
        "email-triage", "intent", "synthetic", "es", True,
        PRODUCT_WEIGHT["dynamic_small"], layer="intent-multi",
        note="4 labels, Spanish only (audit finding D): kept small on purpose"),
    "huffpost": Source(
        "huffpost", "topic", "human", "en", True,
        PRODUCT_WEIGHT["dynamic_large"], layer="intent-multi",
        note="41 categories; 11 of them are the unseen-label holdout"),
    "synth-v1": Source(
        "synth-v1", "grounded", "synthetic", "multi", True,
        PRODUCT_WEIGHT["dynamic_small"], layer="grounded-synth",
        shards=tuple(f"artifacts/synth/v1/synth-v1-{i:03d}.jsonl"
                     for i in range(6)),
        note="teacher-adjudicated distillation corpus; K varies 2-5"),
    "boolq": Source(
        "boolq", "qa", "human", "en", False, PRODUCT_WEIGHT["fixed"],
        layer="human-expert",
        note="yes/no: a two-label pool IS the question, never dynamic"),
    "civil-comments": Source(
        "civil-comments", "toxicity", "human", "en", False,
        PRODUCT_WEIGHT["fixed"], layer="human-expert",
        note="finding E: 74.6 % of the raw corpus, capped at 15 % here"),
    "helpsteer2": Source(
        "helpsteer2", "preference", "human", "en", False,
        PRODUCT_WEIGHT["fixed"], layer="preference",
        note="ordinal score 0-4 shared by 5 attributes; the scale IS the "
             "option set, so it is exempt from the global-label-space check"),
    "prog-gold": Source(
        "prog-gold", "knowledge", "programmatic", "multi", True,
        PRODUCT_WEIGHT["dynamic_large"], layer="programmatic",
        shards=tuple(f"artifacts/prog_gold/v1/prog-gold-v1-{i:03d}.jsonl"
                     for i in range(4)),
        note="#T-prog-gold: Wikidata-grounded gold, every answer verified "
             "against the graph; the probe shard is a robustness fixture "
             "and is deliberately NOT a training shard"),
    # -- the #T-mix-1m widening (data/convert_widen.py) --------------------
    # Five independent corpora, none of them touching the §§18/77 fence.
    # They exist because the caps, not the data, were the binding
    # constraint: with seven sources the registry admitted 39 970 rows of
    # the 1 M §86 asks for, and only independent supply moves that.
    "dbpedia14": Source(
        "dbpedia14", "topic", "human", "en", True,
        PRODUCT_WEIGHT["dynamic_large"], layer="human-expert",
        note="560 k DBpedia ontology abstracts over 14 curated classes; "
             "the SECOND source in the `topic` family, and the wide-K "
             "half of it carries measured nearest-label distractors"),
    "snli": Source(
        "snli", "nli", "human", "en", False, PRODUCT_WEIGHT["fixed"],
        layer="nli",
        note="550 k premise/hypothesis pairs: the 3-way relation IS the "
             "option set, so it is exempt from the global-label-space "
             "check; the boolean twin keeps the §48 Noul share alive"),
    "goemotions": Source(
        "goemotions", "emotion", "human", "en", True,
        PRODUCT_WEIGHT["dynamic_large"], layer="human-expert",
        note="28 emotions over 211 k rater judgements; the K=12 question "
             "of each row is the measured nearest neighbourhood of its "
             "gold, which is where the §68 hard slice comes from"),
    "swag": Source(
        "swag", "commonsense", "programmatic", "en", True,
        PRODUCT_WEIGHT["dynamic_small"], layer="adversarial",
        note="73 k grounded situations with four continuations; the §86 "
             "adversarial layer, which had no supply at all. Its three "
             "distractors survived adversarial filtering against a model "
             "ensemble (Zellers et al. 2018) — a property of the corpus, "
             "not a difficulty this registry asserted. Origin is "
             "programmatic and not human: the gold caption is human, the "
             "option SET is machine-constructed"),
    "episodic-div": Source(
        "episodic-div", "episodic", "synthetic", "en", True,
        PRODUCT_WEIGHT["dynamic_small"], layer="episodic",
        shards=tuple(f"artifacts/episodic-div/episodic-div-{i:03d}.jsonl"
                     for i in range(20)),
        experimental=True,
        note="#T-labelspace-div: 20 000 miniature taxonomies (~52 rows "
             "each) at 1 M total volume — the many-small-spaces alternative "
             "to decision-mix-clean-1m's 9 shared taxonomies. EXPERIMENTAL: "
             "never in a default scan, only via --only-dataset"),
    "detox-attack": Source(
        "detox-attack", "calibration", "human", "en", False,
        PRODUCT_WEIGHT["fixed"], layer="preference",
        note="Wikipedia talk-page comments rated by ~10 annotators each; "
             "the gold is the measured share of annotators on a 0-4 "
             "scale, which is the §48 Score supply HelpSteer2 was fenced "
             "out of — the scale IS the option set, so it is exempt from "
             "the global-label-space check"),
}

#: every id the mixture may draw from, in registry order
TRAINABLE_DATASETS = tuple(SOURCES)


def default_datasets(datasets=None) -> list:
    """The ids a scan plans over when nobody names any: the registry minus
    experimental sources (#T-labelspace-div). Explicit lists pass through
    untouched, so an experiment is selectable but never a default."""
    if datasets is not None:
        return list(datasets)
    return [d for d in TRAINABLE_DATASETS if not SOURCES[d].experimental]

#: corpora the eval firewall refuses outright (`data.firewall`): they are
#: named here only so a mix manifest records WHY they are absent.
NEVER_TRAINABLE = {
    "synth-loop": "quarantined (finding C): 190 skeletons, split by i % 10",
    "logiqa": "reasoning benchmark, eval-only (finding G)",
    "reclor": "reasoning benchmark, eval-only (finding G)",
}

# -- `decision-mix-clean-1m`: ONE recipe, two consumers (#T-mix-1m) -------
#: `tools/mix_1m/run_mix.py` publishes the manifest from these three
#: numbers and `train_decision --fence-clean` assembles the corpus from
#: the same three, so both land on the same `members_sha256`.
#:
#: They are here, in the mixture authority, because the alternative was
#: measured: the trainer used to default its mixture target to the cap
#: CEILING and plan at `cap_margin=0.0`, while the mixer asked for exactly
#: `CLEAN_1M_TARGET` at `CLEAN_1M_CAP_MARGIN`. Two construction paths, two
#: corpora — and a `--max-samples 1000000` run trained 39 981 rows 25
#: times over while its manifest said 1 M.
CLEAN_1M_TARGET = 1_000_000
#: the seed of the published manifest (`artifacts/mix-1m/
#: decision-mix-clean-1m-seed20260922.manifest.json`)
CLEAN_1M_SEED = 20260922
#: 100 rows in a million: `realise()` trims at ROW granularity, so a
#: source whose rows carry two or three questions stops a member or two
#: short of its quota and the realised total lands just under the target.
#: A source planned at exactly 15 % of the target is then over 15 % of
#: that smaller total. Planning against a cap this much tighter is what
#: makes the HARD caps hold on what was actually built.
CLEAN_1M_CAP_MARGIN = 0.0001


#: `tools/mix_1m/fence.py` fences these out of its own §86 assembly because
#: they overlap the Jevals suite. The product path (`data/firewall.py`) does
#: not, and already trains banking77. The disagreement is recorded in every
#: manifest rather than silently resolved — see `fence_conflicts()`.
MIX_1M_FENCED = ("banking77", "helpsteer2", "pubmedqa")


def fence_conflicts() -> list:
    return [{"dataset": d, "in_product_mix": d in SOURCES,
             "fenced_by": "tools/mix_1m/fence.py FENCED_DATASETS",
             "reason": "Jevals benchmark overlap (§§18, 77)"}
            for d in MIX_1M_FENCED if d in SOURCES]


def family_of(dataset: str) -> str:
    return SOURCES[dataset].family


def families(datasets=None) -> dict:
    """family -> the datasets in it, for the cap arithmetic."""
    out: dict = {}
    for d in default_datasets(datasets):
        out.setdefault(SOURCES[d].family, []).append(d)
    return out


def layer_of(dataset: str) -> str:
    """The §86 layer of a source, or `unassigned` if the registry has none."""
    return SOURCES[dataset].layer or "unassigned"


def layers(datasets=None) -> dict:
    """§86 layer -> the datasets in it."""
    out: dict = {}
    for d in default_datasets(datasets):
        out.setdefault(layer_of(d), []).append(d)
    return out


#: §86 layer proportions of `decision-mix-v1` (the 5 M plan). They are a
#: PULL, never a cap: `plan_mix(weights=...)` starts the allocator here and
#: the §§65-66 guardrails clamp whatever comes out.
LAYER_TARGETS = {
    "human-expert": 0.30,    # 1.5M Tasksource/P3 human/expert-derived
    "nli": 0.10,             # 0.5M DocNLI/NLI
    "intent-multi": 0.10,    # 0.5M intent/topic/multilingual
    "preference": 0.10,      # 0.5M preference/ordinal
    "programmatic": 0.20,    # 1.0M Wikidata/rules
    "grounded-synth": 0.14,  # 0.7M grounded synthetic
    "adversarial": 0.06,     # 0.3M adversarial/OOD
}


def layer_weights(datasets=None) -> dict:
    """The §86 plan spread over the sources of each layer.

    A layer's target is split evenly among its members, so the LAYER and
    not the number of files inside it carries the weight. A source whose
    layer the strategy never planned for keeps an epsilon pull, so it is
    drawable but never preferred.
    """
    out: dict = {}
    for layer, members in layers(datasets).items():
        target = LAYER_TARGETS.get(layer, 0.0)
        for dataset in members:
            out[dataset] = (target / len(members)) if target else 1e-6
    return out


def clean_datasets(datasets=None) -> list:
    """The registry minus the benchmark fence (§§18, 77).

    `decision-mix-clean-1m` is assembled from THIS list; the product mix is
    not, and `fence_conflicts()` records the disagreement in every manifest
    rather than resolving it silently.
    """
    return [d for d in default_datasets(datasets)
            if d not in MIX_1M_FENCED]


# -- shared stratum axes (also imported by tools/mix_1m) -------------------

_SCORE_OPT = re.compile(r"^score \d+$")
_LANG_TAG = re.compile(r"^\[([a-z]{2}(?:-[A-Z]{2})?)\]")


def question_type(kind: str, option_texts: list) -> str:
    """The §48 type axis: choice | noul (K<=2) | score (ordinal scale)."""
    if kind == "score" or (
        len(option_texts) >= 3 and all(_SCORE_OPT.match(o) for o in option_texts)
    ):
        return "score"
    if len(option_texts) <= 2 or kind == "boolean":
        return "noul"
    return "choice"


def lang_of(state: str, scope: str) -> str:
    """`[it-IT] svegliami…` -> `it-IT`; otherwise the source's own scope."""
    m = _LANG_TAG.match(state or "")
    if m:
        return m.group(1)
    return "en" if scope == "multi" else scope


def is_hard(question: dict, layer: str = "") -> bool:
    """The §68 hard/OOD axis, decided from what a row already carries.

    Every component is observable BEFORE training, which is what makes the
    ">= 10 % hard" guardrail countable on the assembled mixture instead of
    being a property nobody measures:

    * the whole `adversarial` layer (§86 layer D) is hard by construction;
    * a wide option set (K >= 9) — the regime where pointing is not a
      two-way guess;
    * a teacher that was not confident (`teacher_conf < 0.7`);
    * an explicit distractor / "none of these" / "todo lo anterior" option,
      which is the trust-and-unknown axis of §95;
    * a question the generator itself labelled `difficulty: hard`
      (#T-prog-gold writes this per question, verified against the graph).
    """
    if layer == "adversarial":
        return True
    options = question.get("options", [])
    if len(options) >= 9:
        return True
    conf = question.get("teacher_conf")
    if isinstance(conf, (int, float)) and not isinstance(conf, bool) \
            and conf < 0.7:
        return True
    quality = question.get("quality")
    if isinstance(quality, dict) and quality.get("difficulty") == "hard":
        return True
    blob = " ".join(f"{o.get('id', '')} {o.get('text', '')}"
                    for o in options).lower()
    return ("distractor" in blob or "mal planteada" in blob
            or "todo lo anterior" in blob or "none of the above" in blob)


def k_bucket(k: int) -> str:
    for hi, name in ((4, "2-4"), (8, "5-8"), (16, "9-16"), (32, "17-32"),
                     (64, "33-64")):
        if k <= hi:
            return name
    return "65-255"


def simpson(share_map: dict) -> float:
    """Simpson diversity 1 - sum(p^2): 0 is a monopoly, 1 is perfectly flat."""
    return 1.0 - sum(p * p for p in share_map.values())


def shares(counts: dict) -> dict:
    total = sum(counts.values()) or 1
    return {k: round(v / total, 6) for k, v in sorted(counts.items())}


# -- seeded, stateless selection ------------------------------------------

def keep_row(seed: int, dataset: str, row_index: int, question_id: str,
             fraction: float) -> bool:
    """Deterministic keep decision, keyed on the ROW, hashed with sha256.

    Stateless on purpose: the same `(seed, fraction)` reproduces the same
    mixture from the same bytes without shipping a member list, and it does
    not depend on `PYTHONHASHSEED` the way `hash()` does.
    """
    if fraction >= 1.0:
        return True
    if fraction <= 0.0:
        return False
    digest = hashlib.sha256(
        f"{seed}\x00{dataset}\x00{row_index}\x00{question_id}".encode()
    ).hexdigest()
    return int(digest[:12], 16) / float(16 ** 12) < fraction


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                return h.hexdigest()
            h.update(block)


# -- guardrails ------------------------------------------------------------

def _pct(x: float) -> str:
    return f"{100.0 * x:.2f} %"


def mix_report(counts: dict, total: int | None = None,
               dataset_cap: float = MAX_DATASET_FRACTION,
               family_cap: float = MAX_FAMILY_FRACTION) -> dict:
    """Per-dataset and per-family shares against the caps, without raising.

    `counts` is `{dataset: n}`. `breaches` carries one sentence per cap
    broken, naming WHICH source or family and by HOW much — the whole point
    of finding E is that a 74.6 % share was reported as a number nobody had
    to act on. `verify_mix()` turns those sentences into the error.
    """
    total = sum(counts.values()) if total is None else total
    if total <= 0:
        return {"total": 0, "checks": {}, "by_family": {},
                "breaches": ["empty mixture: nothing to verify"]}

    by_family: dict = {}
    for dataset, n in counts.items():
        fam = SOURCES[dataset].family if dataset in SOURCES else "unregistered"
        by_family[fam] = by_family.get(fam, 0) + n

    breaches, checks = [], {}
    for dataset, n in sorted(counts.items()):
        frac = n / total
        ok = frac <= dataset_cap + 1e-9
        checks[f"dataset:{dataset}"] = {
            "n": n, "share": round(frac, 6), "cap": dataset_cap, "ok": ok}
        if not ok:
            breaches.append(
                f"dataset {dataset!r} is {_pct(frac)} of the mixture "
                f"({n:,}/{total:,} rows), hard cap {_pct(dataset_cap)}")
    for fam, n in sorted(by_family.items()):
        frac = n / total
        ok = frac <= family_cap + 1e-9
        checks[f"family:{fam}"] = {
            "n": n, "share": round(frac, 6), "cap": family_cap, "ok": ok,
            "datasets": sorted(d for d in counts
                               if SOURCES.get(d) and SOURCES[d].family == fam)}
        if not ok:
            breaches.append(
                f"family {fam!r} is {_pct(frac)} of the mixture "
                f"({n:,}/{total:,} rows), hard cap {_pct(family_cap)}")
    return {"total": total, "checks": checks, "breaches": breaches,
            "by_family": dict(sorted(by_family.items()))}


def verify_mix(counts: dict, total: int | None = None,
               dataset_cap: float = MAX_DATASET_FRACTION,
               family_cap: float = MAX_FAMILY_FRACTION) -> dict:
    """`mix_report`, but a breach is a `MixGuardrailError`, never a flag.

    This is what the loader calls: mix construction aborts here.
    """
    report = mix_report(counts, total, dataset_cap, family_cap)
    if report["breaches"]:
        raise MixGuardrailError(
            "mix guardrail breach — construction aborted: "
            + "; ".join(report["breaches"]))
    return report


def verify_guardrails(counts: dict, total: int) -> dict:
    """The §§65-66 caps over a `tools/mix_1m` stage (its stratum counts).

    Kept here so `data/` and `tools/mix_1m/` cannot drift apart on what the
    numbers are. `counts` carries the `dataset`, `family`, `origin` and
    `hard` axes of `tools.mix_1m.strata.tag`.
    """
    rep: dict = {"total": total, "checks": {}}

    def frac(n: int) -> float:
        return n / total if total else 0.0

    for ds, n in sorted(counts.get("dataset", {}).items()):
        f = frac(n)
        rep["checks"][f"dataset:{ds}<=15%"] = {
            "frac": round(f, 4), "ok": f <= MAX_DATASET_FRACTION + 1e-9}
    for fam, n in sorted(counts.get("family", {}).items()):
        f = frac(n)
        rep["checks"][f"family:{fam}<=30%"] = {
            "frac": round(f, 4), "ok": f <= MAX_FAMILY_FRACTION + 1e-9}
    syn = sum(n for d, n in counts.get("origin", {}).items() if d == "synthetic")
    hum = sum(n for d, n in counts.get("origin", {}).items() if d == "human")
    hard = counts.get("hard", 0)
    rep["checks"]["synthetic<=50%"] = {
        "frac": round(frac(syn), 4),
        "ok": frac(syn) <= MAX_SYNTHETIC_FRACTION + 1e-9}
    rep["checks"]["human>=20%"] = {
        "frac": round(frac(hum), 4),
        "ok": frac(hum) >= MIN_HUMAN_FRACTION - 1e-9}
    rep["checks"]["hard>=10%"] = {
        "frac": round(frac(hard), 4),
        "ok": frac(hard) >= MIN_HARD_FRACTION - 1e-9}
    bad = [k for k, v in rep["checks"].items() if not v["ok"]]
    if bad:
        raise MixGuardrailError(f"guardrail breach: {bad} :: {rep['checks']}")
    return rep


# -- the allocator ---------------------------------------------------------

def cap_units(datasets=None, dataset_cap: float = MAX_DATASET_FRACTION,
              family_cap: float = MAX_FAMILY_FRACTION) -> float:
    """Largest share of a mixture these sources can cover under the caps.

    A family of n datasets contributes `min(family_cap, n * dataset_cap)`.
    Below 1.0 no mixture of any size satisfies the guardrails.
    """
    return sum(min(family_cap, len(ds) * dataset_cap)
               for ds in families(datasets).values())


def allocate(target: int, supply: dict, weights: dict | None = None,
             dataset_cap: float = MAX_DATASET_FRACTION,
             family_cap: float = MAX_FAMILY_FRACTION) -> dict:
    """Water-fill `target` rows over `supply` under both caps.

    Shares start proportional to the product weights; a source that hits its
    supply, its dataset cap or its family's cap is frozen at that value and
    the remainder is re-spread over whoever still has headroom. Unfillable
    means `MixInfeasible`, with the arithmetic that says why.
    """
    datasets = [d for d, n in supply.items() if n > 0]
    if not datasets:
        raise MixInfeasible("no supply in any registered source")
    unknown = [d for d in datasets if d not in SOURCES]
    if unknown:
        raise MixInfeasible(f"unregistered sources in supply: {sorted(unknown)}")

    coverage = cap_units(datasets, dataset_cap, family_cap)
    if coverage < 1.0 - 1e-9:
        fam = {f: len(ds) for f, ds in families(datasets).items()}
        raise MixInfeasible(
            f"the caps cannot be satisfied by these {len(datasets)} sources: "
            f"families {fam} cover at most {_pct(coverage)} of any mixture "
            f"(a family contributes min({_pct(family_cap)}, n x "
            f"{_pct(dataset_cap)})). Add an independent source — no target "
            f"size fixes this.")

    weights = weights or {d: SOURCES[d].weight for d in datasets}
    ds_ceiling = {d: min(supply[d], int(target * dataset_cap))
                  for d in datasets}
    fam_ceiling = {f: int(target * family_cap)
                   for f in families(datasets)}
    fam_of = {d: SOURCES[d].family for d in datasets}

    quota = {d: 0 for d in datasets}
    free = set(datasets)
    remaining = target
    for _ in range(4 * len(datasets) + 8):
        if remaining <= 0 or not free:
            break
        wsum = sum(weights[d] for d in free) or 1.0
        moved = 0
        for d in sorted(free, key=lambda x: (-weights[x], x)):
            want = quota[d] + int(math.floor(remaining * weights[d] / wsum))
            fam_used = sum(quota[x] for x in datasets if fam_of[x] == fam_of[d])
            fam_head = fam_ceiling[fam_of[d]] - fam_used + quota[d]
            new = max(quota[d], min(want, ds_ceiling[d], fam_head))
            moved += new - quota[d]
            quota[d] = new
            if new >= min(ds_ceiling[d], fam_head):
                free.discard(d)
        remaining = target - sum(quota.values())
        if moved == 0:
            break
    # largest-remainder top-up: hand the rounding dust to whoever has room
    for d in sorted(datasets, key=lambda x: (-weights[x], x)):
        while remaining > 0:
            fam_used = sum(quota[x] for x in datasets if fam_of[x] == fam_of[d])
            if quota[d] >= ds_ceiling[d] or fam_used >= fam_ceiling[fam_of[d]]:
                break
            quota[d] += 1
            remaining -= 1
    if remaining > 0:
        head = {d: ds_ceiling[d] - quota[d] for d in datasets}
        raise MixInfeasible(
            f"target {target:,} rows: {remaining:,} unfillable under the "
            f"{_pct(dataset_cap)} dataset / {_pct(family_cap)} family caps. "
            f"Supply {dict(sorted(supply.items()))}, headroom left "
            f"{dict(sorted(head.items()))}. Lower the target or add a source.")
    return {d: quota[d] for d in sorted(quota) if quota[d] > 0}


def max_feasible_target(supply: dict,
                        dataset_cap: float = MAX_DATASET_FRACTION,
                        family_cap: float = MAX_FAMILY_FRACTION) -> int:
    """Largest target `allocate()` can fill with this supply. Binary search.

    Monotone in the target: below the ceiling every source has headroom, at
    it the smallest supplies are exhausted.
    """
    if cap_units(list(supply), dataset_cap, family_cap) < 1.0 - 1e-9:
        return 0
    lo, hi = 0, max(8, sum(supply.values()))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        try:
            allocate(mid, supply, dataset_cap=dataset_cap,
                     family_cap=family_cap)
        except MixInfeasible:
            hi = mid - 1
        else:
            lo = mid
    return lo


# -- supply scan (one pass per file, cached) -------------------------------

def _scan_one(dataset: str, paths: list) -> dict:
    """Rows, questions, gold counts, K / language / type axes, shard shas."""
    out: dict = {"dataset": dataset, "shards": [], "rows": {}, "questions": {},
                 "gold_counts": {}, "k_counts": {}, "lang_counts": {},
                 "qtype_counts": {}, "pool": {}, "questions_per_row": {}}
    scope = SOURCES[dataset].lang_scope
    for path in paths:
        if not os.path.exists(path):
            continue
        out["shards"].append({"path": os.path.relpath(path, ROOT),
                              "sha256": sha256_file(path),
                              "bytes": os.path.getsize(path)})
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                split = row.get("split") or "unknown"
                out["rows"][split] = out["rows"].get(split, 0) + 1
                questions = row.get("questions", [])
                n_q = str(len(questions))
                out["questions_per_row"][n_q] = \
                    out["questions_per_row"].get(n_q, 0) + 1
                out["questions"][split] = \
                    out["questions"].get(split, 0) + len(questions)
                if split != "train":
                    continue
                lang = lang_of(row.get("state", ""), scope)
                for q in questions:
                    options = q.get("options", [])
                    for opt in options:
                        out["pool"].setdefault(opt["id"],
                                               opt.get("text", opt["id"]))
                    gold = q.get("answer")
                    if gold is not None:
                        out["gold_counts"][gold] = \
                            out["gold_counts"].get(gold, 0) + 1
                    k = str(len(options))
                    out["k_counts"][k] = out["k_counts"].get(k, 0) + 1
                    out["lang_counts"][lang] = \
                        out["lang_counts"].get(lang, 0) + 1
                    qt = question_type(q.get("kind", ""),
                                       [o.get("text", "") for o in options])
                    out["qtype_counts"][qt] = out["qtype_counts"].get(qt, 0) + 1
    out["train_questions"] = out["questions"].get("train", 0)
    out["train_rows"] = out["rows"].get("train", 0)
    out["pool_size"] = len(out["pool"])
    out["total_questions"] = sum(out["questions"].values())
    return out


def scan_supply(datasets=None, root: str = PREFETCH_DIR,
                cache_path: str | None = None, refresh: bool = False) -> dict:
    """Per-source supply, cached on (path, size, mtime) of every shard.

    `civil-comments.jsonl` is 1 GB; a rebuild that re-reads it for every
    experiment is how a guardrail stops being run.
    """
    datasets = default_datasets(datasets)
    cache_path = cache_path or os.path.join(MIX_DIR, "supply-cache.json")
    cache: dict = {}
    if os.path.exists(cache_path) and not refresh:
        try:
            with open(cache_path) as fh:
                cache = json.load(fh)
        except (OSError, ValueError):
            cache = {}

    out: dict = {}
    dirty = False
    for dataset in datasets:
        paths = [p for p in SOURCES[dataset].paths(root) if os.path.exists(p)]
        stamp = [[os.path.relpath(p, ROOT), os.path.getsize(p),
                  os.path.getmtime(p)] for p in paths]
        hit = cache.get(dataset)
        if hit and hit.get("stamp") == stamp:
            out[dataset] = hit["scan"]
            continue
        if not paths:
            out[dataset] = {"dataset": dataset, "missing": True,
                            "train_questions": 0, "shards": [], "pool": {},
                            "gold_counts": {}, "k_counts": {},
                            "lang_counts": {}, "qtype_counts": {},
                            "rows": {}, "questions": {}, "pool_size": 0,
                            "total_questions": 0, "train_rows": 0,
                            "questions_per_row": {}}
            continue
        scan = _scan_one(dataset, paths)
        out[dataset] = scan
        cache[dataset] = {"stamp": stamp, "scan": scan}
        dirty = True
    if dirty:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as fh:
            json.dump(cache, fh, sort_keys=True)
    return out


def effective_supply(scan: dict, seen_labels: dict | None = None) -> dict:
    """Trainable questions per source AFTER the unseen-label holdout cuts.

    A row whose gold is a held-out label never trains (#T-unseen-labels), so
    planning a mixture on the raw counts would plan a share the loader then
    cannot fill — and the realised composition would drift past the cap.
    """
    out = {}
    for dataset, s in scan.items():
        if s.get("missing"):
            continue
        keep = (seen_labels or {}).get(dataset)
        if keep is None:
            out[dataset] = s["train_questions"]
        else:
            keep = set(keep)
            out[dataset] = sum(n for label, n in s["gold_counts"].items()
                               if label in keep)
    return {d: n for d, n in out.items() if n > 0}


# -- the descriptor --------------------------------------------------------

@dataclass
class MixSpec:
    """The mixture a training run consumes. Serialised as the manifest."""

    version: str = MIX_VERSION
    seed: int = DEFAULT_SEED
    target: int = 0
    quotas: dict = field(default_factory=dict)
    supply: dict = field(default_factory=dict)
    keep_fractions: dict = field(default_factory=dict)
    seen_labels: dict = field(default_factory=dict)
    shards: dict = field(default_factory=dict)

    @property
    def datasets(self) -> list:
        return sorted(self.quotas)

    def keep(self, dataset: str):
        """`(row_index, question_id) -> bool` for one source, stateless."""
        fraction = self.keep_fractions.get(dataset, 1.0)
        seen = self.seen_labels.get(dataset)
        seen = set(seen) if seen is not None else None

        def _keep(row_index: int, question_id: str, gold=None) -> bool:
            if seen is not None and gold is not None and gold not in seen:
                return False
            return keep_row(self.seed, dataset, row_index, question_id,
                            fraction)
        return _keep

    def to_dict(self) -> dict:
        return asdict(self)


def plan_mix(supply: dict, target: int | None = None, seed: int = DEFAULT_SEED,
             seen_labels: dict | None = None, scan: dict | None = None,
             cap_margin: float = CAP_SAFETY_MARGIN,
             weights: dict | None = None,
             dataset_cap: float | None = None,
             family_cap: float | None = None) -> MixSpec:
    """Quotas + keep fractions. No corpus is read: pure arithmetic.

    `weights` overrides the registry's product weights — `tools/mix_1m`
    passes the §86 layer targets through it so the clean corpus is pulled
    towards the strategy's composition. It changes the PULL, never the
    caps: the same `allocate()` clamps the result either way.

    `dataset_cap`/`family_cap` replace the §§65-66 values outright, for an
    experiment that has to state a different cap out loud (see
    `MixtureStream`). `cap_margin` is ignored for whichever of the two is
    given, so a run cannot end up planning against one pair of caps and
    being verified against another.
    """
    margin_cap = (MAX_DATASET_FRACTION - cap_margin if dataset_cap is None
                  else dataset_cap)
    margin_fam = (MAX_FAMILY_FRACTION - cap_margin if family_cap is None
                  else family_cap)
    if target is None:
        target = max_feasible_target(supply, margin_cap, margin_fam)
        if target <= 0:
            raise MixInfeasible(
                "no feasible mixture: " + _infeasible_reason(supply))
    quotas = allocate(target, supply, weights=weights, dataset_cap=margin_cap,
                      family_cap=margin_fam)
    keep_fractions = {d: min(1.0, DRAW_HEADROOM * quotas[d] / supply[d])
                      for d in quotas}
    shards = {d: (scan or {}).get(d, {}).get("shards", []) for d in quotas}
    return MixSpec(version=MIX_VERSION, seed=seed, target=target,
                   quotas=quotas, supply={d: supply[d] for d in quotas},
                   keep_fractions=keep_fractions,
                   seen_labels={d: sorted(v) for d, v in
                                (seen_labels or {}).items() if d in quotas},
                   shards=shards)


def _infeasible_reason(supply: dict) -> str:
    try:
        allocate(1, supply)
    except MixInfeasible as exc:
        return str(exc)
    return "unknown"


# -- realisation: read the corpus once and count what the caps see ---------

def realise(spec: MixSpec, root: str = PREFETCH_DIR, on_member=None) -> dict:
    """Enumerate the selected members and count every diversity axis.

    This is the pass that proves the plan: `members_sha256` is a digest of
    the ordered `(dataset, row_index, question_id)` keys, so a rebuild from
    the same seed and the same shard shas is comparable bit for bit.

    `on_member(dataset, row_index, row, question)` observes every selected
    member as it is counted. It exists so a caller that needs another axis
    — `tools/mix_1m` runs the §§18/77 benchmark fence and the §86 stratum
    tagging through it — reads the SAME selection this function counts,
    instead of re-deciding membership in a second loop that can drift.
    """
    digest = hashlib.sha256()
    counts = {"dataset": {}, "family": {}, "origin": {}, "qtype": {},
              "lang": {}, "k": {}, "k_bucket": {}, "layer": {}, "hard": 0,
              "dynamic": 0, "rows": 0}
    per_dataset: dict = {}
    ledger: dict = {}
    for dataset in spec.datasets:
        source = SOURCES[dataset]
        keep = spec.keep(dataset)
        quota = spec.quotas[dataset]
        kept = 0
        index = -1
        for path in source.paths(root):
            if not os.path.exists(path) or kept >= quota:
                continue
            shard = ledger.setdefault(
                os.path.relpath(path, ROOT),
                {"dataset": dataset, "examples": 0, "state_tokens": 0,
                 "candidate_tokens": 0, "k_sum": 0})
            with open(path) as fh:
                for line in fh:
                    if kept >= quota:  # drawn with headroom, trimmed here
                        break
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if row.get("split") != "train":
                        continue
                    index += 1
                    state = row.get("state", "")
                    lang = lang_of(state, source.lang_scope)
                    selected = [q for q in row.get("questions", [])
                                if keep(index, q.get("id", ""),
                                        q.get("answer"))]
                    # trim at ROW granularity, so the loader (which holds
                    # whole rows) stops at exactly the same member
                    if not selected or kept + len(selected) > quota:
                        if kept + len(selected) > quota:
                            break
                        continue
                    state_tokens = len(state.split())
                    for q in selected:
                        qid = q.get("id", "")
                        options = q.get("options", [])
                        k = len(options)
                        shard["examples"] += 1
                        shard["state_tokens"] += state_tokens
                        shard["candidate_tokens"] += sum(
                            len(o.get("text", "").split()) for o in options)
                        shard["k_sum"] += k
                        if on_member is not None:
                            on_member(dataset, index, row, q)
                        qt = question_type(
                            q.get("kind", ""),
                            [o.get("text", "") for o in options])
                        digest.update(
                            f"{dataset}\x00{index}\x00{qid}\n".encode())
                        counts["dataset"][dataset] = \
                            counts["dataset"].get(dataset, 0) + 1
                        counts["family"][source.family] = \
                            counts["family"].get(source.family, 0) + 1
                        counts["origin"][source.origin] = \
                            counts["origin"].get(source.origin, 0) + 1
                        counts["qtype"][qt] = counts["qtype"].get(qt, 0) + 1
                        counts["lang"][lang] = counts["lang"].get(lang, 0) + 1
                        counts["k"][str(k)] = counts["k"].get(str(k), 0) + 1
                        bucket = k_bucket(k)
                        counts["k_bucket"][bucket] = \
                            counts["k_bucket"].get(bucket, 0) + 1
                        layer = source.layer or "unassigned"
                        counts["layer"][layer] = \
                            counts["layer"].get(layer, 0) + 1
                        counts["hard"] += int(is_hard(q, source.layer))
                        counts["dynamic"] += int(source.dynamic_options)
                        counts["rows"] += 1
                        kept += 1
        per_dataset[dataset] = {
            "kept": kept, "quota": spec.quotas[dataset],
            "supply": spec.supply[dataset],
            "keep_fraction": round(spec.keep_fractions[dataset], 8),
            "family": source.family, "origin": source.origin,
            "dynamic_options": source.dynamic_options,
            "weight": source.weight, "note": source.note}
    counts["per_dataset"] = per_dataset
    for shard in ledger.values():
        n = max(shard["examples"], 1)
        shard["mean_k"] = round(shard["k_sum"] / n, 3)
        shard["tokens"] = shard["state_tokens"] + shard["candidate_tokens"]
    counts["tokens_by_shard"] = dict(sorted(ledger.items()))
    counts["tokens"] = {
        "examples": sum(v["examples"] for v in ledger.values()),
        "state_tokens": sum(v["state_tokens"] for v in ledger.values()),
        "candidate_tokens": sum(v["candidate_tokens"] for v in
                                ledger.values()),
        "total_tokens": sum(v["tokens"] for v in ledger.values()),
        "mean_k": round(sum(v["k_sum"] for v in ledger.values())
                        / max(counts["rows"], 1), 3),
        "unit": "whitespace tokens of the state text and of every option "
                "text, counted on the rows the mixture actually selected",
    }
    counts["members_sha256"] = digest.hexdigest()
    return counts


def diversity_dashboard(counts: dict) -> dict:
    """Shares + Simpson index on every axis the strategy asks to publish."""
    axes = {}
    for axis in ("dataset", "family", "origin", "qtype", "k_bucket", "lang",
                 "layer"):
        axes[axis] = shares(counts.get(axis, {}))
    return {"shares": axes,
            "simpson": {a: round(simpson(s), 4) for a, s in axes.items()}}


def composition_report(counts: dict) -> dict:
    total = counts.get("rows", 0)
    dynamic = counts.get("dynamic", 0)
    return {
        "rows": total,
        "by_dataset": dict(sorted(counts.get("dataset", {}).items())),
        "by_family": dict(sorted(counts.get("family", {}).items())),
        "by_origin": dict(sorted(counts.get("origin", {}).items())),
        "by_qtype": dict(sorted(counts.get("qtype", {}).items())),
        "by_k": dict(sorted(counts.get("k", {}).items(),
                            key=lambda t: int(t[0]))),
        "by_lang": dict(sorted(counts.get("lang", {}).items())),
        "by_layer": dict(sorted(counts.get("layer", {}).items())),
        "hard_rows": counts.get("hard", 0),
        "hard_share": round(counts.get("hard", 0) / total, 6) if total else 0.0,
        "dynamic_option_rows": dynamic,
        "dynamic_option_share": round(dynamic / total, 6) if total else 0.0,
        "top_dataset_share": (round(max(counts.get("dataset", {}).values())
                                    / total, 6) if total else 0.0),
    }


def check_composition(counts: dict) -> dict:
    """Every "Done when" of #T-corpus-rebalance, as a pass/fail block."""
    total = counts.get("rows", 0)
    guard = mix_report(counts.get("dataset", {}), total)
    dataset_shares = shares(counts.get("dataset", {}))
    family_shares = shares(counts.get("family", {}))
    origin_shares = shares(counts.get("origin", {}))
    dynamic_share = counts.get("dynamic", 0) / total if total else 0.0
    checks = {
        "guardrails_hold": {
            "pass": not guard["breaches"],
            "breaches": guard["breaches"],
            "enforced_by": "data.mix.verify_mix (raises MixGuardrailError)"},
        "no_dataset_over_cap": {
            "pass": all(s <= MAX_DATASET_FRACTION + 1e-9
                        for s in dataset_shares.values()),
            "cap": MAX_DATASET_FRACTION, "shares": dataset_shares},
        "no_family_over_cap": {
            "pass": all(s <= MAX_FAMILY_FRACTION + 1e-9
                        for s in family_shares.values()),
            "cap": MAX_FAMILY_FRACTION, "shares": family_shares},
        "dynamic_options_are_the_majority": {
            "pass": dynamic_share > MIN_DYNAMIC_FRACTION,
            "floor": MIN_DYNAMIC_FRACTION,
            "share": round(dynamic_share, 6),
            "why": ("the product property is choosing among options that "
                    "vary per row; fixed-label binaries must not be most "
                    "of what the head sees")},
        "synthetic_is_a_minority": {
            "pass": origin_shares.get("synthetic", 0.0)
            <= MAX_SYNTHETIC_FRACTION + 1e-9,
            "cap": MAX_SYNTHETIC_FRACTION,
            "share": origin_shares.get("synthetic", 0.0)},
    }
    return {"pass": all(c["pass"] for c in checks.values()),
            "checks": checks, "guardrails": guard}


# -- manifest --------------------------------------------------------------

def manifest_path(version: str = MIX_VERSION) -> str:
    return os.path.join(MIX_DIR, version, "manifest.json")


def build_manifest(spec: MixSpec, counts: dict, scan: dict | None = None,
                   elapsed_s: float | None = None) -> dict:
    report = composition_report(counts)
    checks = check_composition(counts)
    return {
        "format": MANIFEST_FORMAT,
        "version": spec.version,
        "task": "T-corpus-rebalance",
        "seed": spec.seed,
        "target": spec.target,
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(elapsed_s, 1) if elapsed_s is not None else None,
        "guardrails": {
            "max_dataset_fraction": MAX_DATASET_FRACTION,
            "max_family_fraction": MAX_FAMILY_FRACTION,
            "max_synthetic_fraction": MAX_SYNTHETIC_FRACTION,
            "min_dynamic_fraction": MIN_DYNAMIC_FRACTION,
            "cap_safety_margin": CAP_SAFETY_MARGIN,
            "enforcement": ("hard error: data.mix.verify_mix raises "
                            "MixGuardrailError and mix construction aborts"),
            "cap_units": round(cap_units(spec.datasets), 4),
        },
        "selection": {
            "function": "data.mix.keep_row",
            "key": "sha256(seed \\0 dataset \\0 row_index \\0 question_id)",
            "keep_fractions": {d: round(v, 8)
                               for d, v in sorted(spec.keep_fractions.items())},
            "note": ("stateless: seed + keep fractions + the shard sha256 "
                     "below rebuild this mixture without a member list"),
        },
        "sources": {
            d: {"shards": spec.shards.get(d, []),
                "quota": spec.quotas[d],
                "supply": spec.supply[d],
                "kept": counts["per_dataset"].get(d, {}).get("kept", 0),
                "share": round(counts["per_dataset"].get(d, {}).get("kept", 0)
                               / max(counts["rows"], 1), 6),
                "family": SOURCES[d].family,
                "layer": layer_of(d),
                "origin": SOURCES[d].origin,
                "lang_scope": SOURCES[d].lang_scope,
                "dynamic_options": SOURCES[d].dynamic_options,
                "weight": SOURCES[d].weight,
                "pool_size": (scan or {}).get(d, {}).get("pool_size"),
                "note": SOURCES[d].note}
            for d in spec.datasets},
        "excluded": NEVER_TRAINABLE,
        "fence_conflicts": fence_conflicts(),
        "seen_labels": {d: len(v) for d, v in sorted(spec.seen_labels.items())},
        "composition": report,
        "diversity": diversity_dashboard(counts),
        "members_sha256": counts["members_sha256"],
        "checks": checks,
    }


def write_manifest(manifest: dict, path: str | None = None) -> str:
    path = path or manifest_path(manifest.get("version", MIX_VERSION))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def load_manifest(path: str | None = None) -> dict:
    with open(path or manifest_path()) as fh:
        return json.load(fh)


def spec_from_manifest(manifest: dict) -> MixSpec:
    """Rebuild the descriptor a run consumes from a written manifest."""
    if manifest.get("version") != MIX_VERSION:
        raise ValueError(
            f"manifest version {manifest.get('version')!r} was written by "
            f"another registry; this code builds {MIX_VERSION!r}")
    sources = manifest["sources"]
    return MixSpec(
        version=manifest["version"], seed=manifest["seed"],
        target=manifest["target"],
        quotas={d: s["quota"] for d, s in sources.items()},
        supply={d: s["supply"] for d, s in sources.items()},
        keep_fractions=dict(manifest["selection"]["keep_fractions"]),
        seen_labels={}, shards={d: s.get("shards", [])
                                for d, s in sources.items()})


def build(target: int | None = None, seed: int = DEFAULT_SEED,
          datasets=None, root: str = PREFETCH_DIR,
          seen_labels: dict | None = None, write: bool = True,
          refresh: bool = False, cap_margin: float = CAP_SAFETY_MARGIN,
          weights: dict | None = None) -> tuple:
    """Scan, plan, realise, verify, write. Returns `(spec, manifest)`."""
    t0 = time.perf_counter()
    scan = scan_supply(datasets, root, refresh=refresh)
    supply = effective_supply(scan, seen_labels)
    spec = plan_mix(supply, target, seed, seen_labels, scan, cap_margin,
                    weights)
    counts = realise(spec, root)
    verify_mix(counts["dataset"], counts["rows"])  # hard error, never a flag
    manifest = build_manifest(spec, counts, scan, time.perf_counter() - t0)
    if write:
        write_manifest(manifest)
    return spec, manifest


# -- gate ------------------------------------------------------------------

GATE_DIR = os.path.join(ROOT, "artifacts", "gates", "T-corpus-rebalance")

#: the numbers audit-2026-09-21 finding E published, kept verbatim so the
#: gate's "before" column is the operator's own number, not a re-derivation
AUDIT_BEFORE = {"total_examples": 2_681_195, "civil_comments": 1_999_514,
                "civil_comments_share": 0.746,
                "source": ".meshkore/docs/audit-2026-09-21.md finding E"}


def before_composition(scan: dict) -> dict:
    """What the corpus looks like with no mixture applied at all."""
    supply = {d: s["train_questions"] for d, s in scan.items()
              if not s.get("missing") and s["train_questions"]}
    total = sum(supply.values())
    by_family: dict = {}
    dynamic = 0
    for d, n in supply.items():
        by_family[SOURCES[d].family] = by_family.get(SOURCES[d].family, 0) + n
        dynamic += n if SOURCES[d].dynamic_options else 0
    top = max(supply, key=lambda d: supply[d]) if supply else None
    return {
        "what": "every trainable source, whole, as the trainer would read it",
        "rows": total,
        "by_dataset": dict(sorted(supply.items())),
        "by_family": dict(sorted(by_family.items())),
        "dynamic_option_share": round(dynamic / total, 6) if total else 0.0,
        "top_dataset": top,
        "top_dataset_share": round(supply[top] / total, 6) if total else 0.0,
        "guardrail_verdict": _verdict(supply, total),
        "audit_finding_e": AUDIT_BEFORE,
    }


def _verdict(counts: dict, total: int) -> str:
    breaches = mix_report(counts, total)["breaches"]
    return "; ".join(breaches) if breaches else "within the caps"


def run_gate(target: int | None = None, seed: int = DEFAULT_SEED,
             root: str = PREFETCH_DIR, write: bool = True) -> dict:
    """Build the mixture, prove it reproduces, write the gate. Hard-fails.

    Reproducibility is checked the way the manifest claims it: a second
    selection from the same seed and the same shard shas must yield the
    same ordered member digest, with no member list in between.
    """
    t0 = time.perf_counter()
    scan = scan_supply(root=root)
    before = before_composition(scan)
    spec, manifest = build(target=target, seed=seed, root=root)
    rebuild = realise(spec, root)
    reproduced = rebuild["members_sha256"] == manifest["members_sha256"]

    checks = dict(manifest["checks"]["checks"])
    checks["manifest_reproduces_the_mixture"] = {
        "pass": reproduced,
        "members_sha256": manifest["members_sha256"],
        "rebuilt_sha256": rebuild["members_sha256"],
        "rebuilt_from": "seed + keep fractions + the shard sha256 in the "
                        "manifest; no member list was read"}
    checks["civil_comments_is_no_longer_the_corpus"] = {
        "pass": (manifest["composition"]["by_dataset"].get("civil-comments", 0)
                 / max(manifest["composition"]["rows"], 1)
                 <= MAX_DATASET_FRACTION + 1e-9),
        "before_share": before["by_dataset"].get("civil-comments", 0)
        / max(before["rows"], 1),
        "after_share": manifest["composition"]["by_dataset"]
        .get("civil-comments", 0) / max(manifest["composition"]["rows"], 1),
        "cap": MAX_DATASET_FRACTION}

    gate = {
        "format": 1,
        "task": "T-corpus-rebalance",
        "pass": all(c["pass"] for c in checks.values()),
        "mix_version": manifest["version"],
        "seed": manifest["seed"],
        "manifest": os.path.relpath(manifest_path(), ROOT),
        "before": before,
        "after": {**manifest["composition"],
                  "guardrail_verdict": _verdict(
                      manifest["composition"]["by_dataset"],
                      manifest["composition"]["rows"])},
        "guardrails": manifest["guardrails"],
        "diversity": manifest["diversity"],
        "fence_conflicts": manifest["fence_conflicts"],
        "excluded": manifest["excluded"],
        "checks": checks,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if write:
        os.makedirs(GATE_DIR, exist_ok=True)
        with open(os.path.join(GATE_DIR, "gate.json"), "w") as fh:
            json.dump(gate, fh, indent=2, sort_keys=True)
            fh.write("\n")
    return gate


# -- CLI -------------------------------------------------------------------

def _show(root: str = PREFETCH_DIR) -> int:
    scan = scan_supply(root=root)
    supply = effective_supply(scan)
    total = sum(supply.values())
    print(f"{'dataset':16s} {'family':11s} {'train q':>10s} {'share':>8s} "
          f"{'dyn':>4s}  pool")
    for d in sorted(supply, key=lambda x: -supply[x]):
        s = SOURCES[d]
        print(f"{d:16s} {s.family:11s} {supply[d]:10,d} "
              f"{100.0 * supply[d] / total:7.2f}% "
              f"{'yes' if s.dynamic_options else 'no':>4s}  "
              f"{scan[d]['pool_size']}")
    print(f"{'TOTAL':16s} {'':11s} {total:10,d}")
    print(f"\ncap units: {cap_units(list(supply)):.3f} "
          f"(needs >= 1.0)   max feasible mixture: "
          f"{max_feasible_target(supply):,}")
    return 0


def main(argv: list) -> int:
    cmd = argv[1] if len(argv) > 1 else "show"
    if cmd == "show":
        return _show()
    if cmd == "verify":
        manifest = load_manifest()
        report = check_composition({
            "rows": manifest["composition"]["rows"],
            "dataset": manifest["composition"]["by_dataset"],
            "family": manifest["composition"]["by_family"],
            "origin": manifest["composition"]["by_origin"],
            "dynamic": manifest["composition"]["dynamic_option_rows"]})
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["pass"] else 1
    if cmd not in ("build", "gate"):
        print(f"unknown command {cmd!r}; use show, build, gate or verify",
              file=sys.stderr)
        return 2
    target = None
    seed = DEFAULT_SEED
    rest = argv[2:]
    for i, arg in enumerate(rest):
        if arg == "--target" and i + 1 < len(rest):
            target = int(rest[i + 1])
        if arg == "--seed" and i + 1 < len(rest):
            seed = int(rest[i + 1])
    if cmd == "gate":
        gate = run_gate(target=target, seed=seed)
        print(json.dumps({"before": gate["before"]["by_dataset"],
                          "after": gate["after"]["by_dataset"],
                          "pass": gate["pass"]}, indent=2, sort_keys=True))
        for name, check in gate["checks"].items():
            print(f"[gate] {name}: {'PASS' if check['pass'] else 'FAIL'}")
        return 0 if gate["pass"] else 1
    _, manifest = build(target=target, seed=seed)
    print(json.dumps({"version": manifest["version"],
                      "rows": manifest["composition"]["rows"],
                      "by_dataset": manifest["composition"]["by_dataset"],
                      "by_family": manifest["composition"]["by_family"],
                      "dynamic_option_share":
                          manifest["composition"]["dynamic_option_share"],
                      "members_sha256": manifest["members_sha256"],
                      "pass": manifest["checks"]["pass"]},
                     indent=2, sort_keys=True))
    return 0 if manifest["checks"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
