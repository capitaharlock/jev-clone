"""Primary metric: accuracy + ECE on labels NEVER SEEN in training
(#T-unseen-labels, finding F of the 2026-09-21 audit).

Everything this repo had published until today measured memorisation of a
FIXED label space. The product promises the opposite: point at the right
option in an option set it has never been trained on. This module publishes
the only number that answers that, and it publishes it ALWAYS PAIRED —
`seen` next to `unseen` — because an unseen number alone cannot tell
"generalises" from "the cut was easy".

Protocol — holdouts by LABEL, never by row
------------------------------------------
* **HuffPost**: train on 30 of the 41 categories, evaluate on the other 11.
* **Banking77**: hold out whole SIBLING PAIRS (the hard negatives
  `#T-optset-sampler` mines), so the holdout is the hard half.
* **MASSIVE**: label holdout as above, plus the cross-lingual arm — train on
  4 locales, evaluate on 2 (`TRAIN_LOCALES` / `EVAL_LOCALES`).
* **boolq**: a two-label pool (yes/no) IS the question; exempt, `seen`-only,
  stated as such.
* **LogiQA / ReClor**: eval-only, never in train (`#T-halt-contam`
  firewall). The external thermometer, published with its date and its
  `model_version`.

The label holdout itself is the trainer's (`training.python.train_decision.
build_holdout`): the same carve the run actually trained against, not a
re-derivation that might disagree with it.

Rows: group splits, never row indexes
-------------------------------------
The published cut and the temperature-fitting cut are separated with
`eval.splits` (#T-split-domain): the unit is the group
`(normalized skeleton, domain, language)`, the partition is seeded, and it
is sealed with a sha256 manifest under `artifacts/splits/`. No row index
decides anything, and no group appears on both sides.

The predictor is the model
--------------------------
`eval.calib` no longer temperature-scales a parameter-free cosine scorer.
The logits here come from the trained pointer head of #T-train-real, loaded
from its safetensors checkpoint, and every artifact carries that
checkpoint's `model_version` — the gate FAILS without it.

Honesty rule
------------
If a number is bad it is published bad. A gate with `pass: false` and a
truthful table is the correct outcome of this task; a green built on a weak
cut is a failure.

CLI:
    CKPT=artifacts/checkpoints/decision/train-real-v1/stage-000250000
    .venv-train/bin/python -m eval.unseen gate --checkpoint $CKPT
    python3 -m eval.unseen holdout-clean        # no torch needed
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.optset import (PREFETCH_DIR, Sample, SamplerConfig,  # noqa: E402
                         iter_rows, question_text)
from eval import calib as C  # noqa: E402
from eval import splits as S  # noqa: E402
from training.python import train_decision as T  # noqa: E402

TASK = "T-unseen-labels"
GATE_DIR = ROOT / "artifacts" / "gates" / TASK
GATE_PATH = GATE_DIR / "gate.json"

#: seed of the metric/calibration partition of each eval split. Fixed here,
#: before the first measurement; changing it re-seals every split loudly.
UNSEEN_SEED = 20260921
#: 70 % of the GROUPS carry the published metric, 30 % fit the temperature
METRIC_FRAC = 0.7
CALIB_FRAC = 0.3

#: MASSIVE ships its locale in the state prefix: ``[it-IT] svegliami …``
MASSIVE_LOCALES = ("de-DE", "en-US", "es-ES", "fr-FR", "it-IT", "pt-PT")
#: the cross-lingual arm: the last two locales in lexicographic order. The
#: rule is mechanical and was fixed before any number was measured, so the
#: eval pair cannot be chosen after seeing which one flatters the model.
EVAL_LOCALES = MASSIVE_LOCALES[-2:]
TRAIN_LOCALES = MASSIVE_LOCALES[:-2]

#: eval-only benchmarks (firewalled out of training by #T-halt-contam)
EVAL_ONLY = ("logiqa", "reclor")
#: ReClor ships its TEST split with the gold hidden upstream (label -1, the
#: leaderboard clean room), so the eval-only number is published on its
#: official VALIDATION split — 500 labelled rows — and said so. LogiQA
#: ships a labelled test split and uses it.
EVAL_ONLY_SPLIT = {"logiqa": "test", "reclor": "calibration"}
EVAL_ONLY_SPLIT_NOTE = {
    "logiqa": "official test split",
    "reclor": "official validation split: the test split's gold is hidden "
              "upstream (label -1), so it cannot be scored at all",
}
#: the task's own threshold for the external thermometer: a 4-option
#: benchmark answered above 0.25 by a model that never saw it means the
#: pointer head is doing its job
THERMOMETER_FLOOR = 0.25

#: rows read per dataset before restriction (every eval split fits under it)
ROWS_CAP = 40_000
#: samples scored per (dataset, cut). MASSIVE gets more because the locale
#: cuts are carved out of the same entries instead of re-running the model.
MAX_SAMPLES = {"massive": 4000}
DEFAULT_MAX_SAMPLES = 2000
MAX_CALIB_SAMPLES = 3000
MAX_EVAL_ONLY = 1500

DATASETS = ("banking77", "huffpost", "massive", "boolq")

_LOCALE = re.compile(r"^\[([a-z]{2}-[A-Z]{2})\]")


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def locale_of(state: str) -> str:
    """The MASSIVE locale tag of a state, or "" for a corpus without one."""
    m = _LOCALE.match(state or "")
    return m.group(1) if m else ""


# -- the metric / calibration partition (groups, sealed) -------------------

def eval_side(state: str, domain: str) -> str:
    """"test" (published) or "calibration" (fits the temperature).

    The decision is taken over the row's GROUP digest — skeleton, domain,
    language — exactly as `#T-split-domain` requires. A row index is never
    read, and a group never lands on both sides.
    """
    return S.split_of_digest(S.group_digest(state, domain), UNSEEN_SEED,
                             METRIC_FRAC, CALIB_FRAC)


def seal_metric_split(domain: str, path: Path | None = None,
                      force: bool = False) -> dict:
    """Seal (or re-verify) the group partition of one dataset's eval split.

    Sealed bytes are never silently regenerated: if a manifest is already
    on disk the ids are recomputed and compared, and a mismatch raises.
    """
    src = path or Path(T.PREFETCH_DIR) / f"{domain}.jsonl"
    eval_split = T.EVAL_SPLIT.get(domain, "test")
    rows = [(rid, state) for rid, state, shipped
            in S.iter_rows(src, domain, None) if shipped == eval_split]
    res = S.SplitResult(f"{TASK}-{domain}", domain, UNSEEN_SEED, METRIC_FRAC,
                        CALIB_FRAC)
    for rid, state in rows:
        key = S.group_key(state, domain)
        dig = key.digest()
        side = S.split_of_digest(dig, UNSEEN_SEED, METRIC_FRAC, CALIB_FRAC)
        res.ids.setdefault(side, []).append(rid)
        res.groups.setdefault(side, set()).add(dig)
        res.group_rows[dig] = res.group_rows.get(dig, 0) + 1
        res.samples.setdefault(dig, key.skeleton[:200])
    for side in res.ids:
        res.ids[side].sort()
    base = S.SPLITS_DIR / res.name
    if (base / "manifest.json").exists() and not force:
        old = S.load_split(res.name)
        want = {f"{side}.ids": S.sha256_text("".join(f"{i}\n" for i in ids))
                for side, ids in sorted(res.ids.items())}
        got = {fn: meta["sha256"]
               for fn, meta in old["manifest"]["files"].items()}
        if want != got:
            raise ValueError(
                f"sealed split {res.name!r} disagrees with the partition "
                "recomputed now: the fence moved. Re-seal on purpose "
                "(--reseal) or find out what changed")
        man = old["manifest"]
    else:
        man = S.seal_split(res, source=src, source_rows=len(rows),
                           force=force)
    shared = sorted(res.groups.get("test", set())
                    & res.groups.get("calibration", set()))
    return {"name": res.name, "counts": man["counts"],
            "shared_groups_metric_vs_calibration": len(shared),
            "shared_examples": shared[:5],
            "files": man["files"],
            "seed": man["seed"], "grouped_by": man["grouped_by"],
            "fracs": man["fracs"],
            "manifest_sha256": man["manifest_sha256"],
            "manifest": f"artifacts/splits/{res.name}/manifest.json"}


# -- sample construction ---------------------------------------------------

def eval_config(base: SamplerConfig | None = None) -> SamplerConfig:
    """Eval never emits `unknown` rows: an unknown row has no gold to point
    at. The model's own abstention is measured through `abstain_rate` and
    the `unknown` risk-coverage curve instead."""
    from dataclasses import asdict
    cfg = base or SamplerConfig()
    return SamplerConfig(**{**asdict(cfg), "unknown_fraction": 0.0})


def samples_of(sampler, dataset: str, side: str, limit: int,
               locales: tuple | None = None,
               seed: int = UNSEEN_SEED) -> list:
    """Compose the eval samples of one side of the group partition.

    Row ids keep their ordinal in the UNRESTRICTED row list, so a row's
    option set does not change when the other side is taken — the option
    set is seeded by `(seed, dataset, row, question)`.

    When the side holds more rows than `limit`, the rows are picked by a
    seeded shuffle, never by file order: MASSIVE is written locale after
    locale, so "the first N rows" would be a single-locale cut wearing a
    mixture's name.
    """
    eligible = []
    for n, row in enumerate(sampler.rows[dataset]):
        state = row.get("state", "")
        if eval_side(state, dataset) != side:
            continue
        if locales and locale_of(state) not in locales:
            continue
        eligible.append((n, row))
    random.Random(f"{seed}\x00pick\x00{dataset}\x00{side}").shuffle(eligible)
    out = []
    for n, row in eligible:
        rid = f"{dataset}-{n}"
        for question in row.get("questions", []):
            sample = sampler.compose(dataset, row, question, rid)
            sampler._shuffle(sample, 0)
            out.append(sample)
            if len(out) >= limit:
                return out
    return out


def cut_samples(holdout: dict, dataset: str, which: str, side: str,
                limit: int, cfg: SamplerConfig | None = None) -> list:
    """Samples of one (dataset, seen|unseen, side) cut.

    `which == "unseen"` restricts BOTH the row pool and the distractor pool
    to held-out labels, so the model cannot win the cut by elimination.
    """
    h = holdout[dataset]
    keep = set(h.seen if which == "seen" else h.unseen)
    if len(keep) < 2:
        return []
    sampler = T.make_sampler(dataset, keep, T.EVAL_SPLIT.get(dataset, "test"),
                             eval_config(cfg), ROWS_CAP, T.PREFETCH_DIR)
    return samples_of(sampler, dataset, side, limit)


def calibration_entries(engine, holdout: dict,
                        cfg: SamplerConfig | None = None,
                        max_samples: int = MAX_CALIB_SAMPLES) -> list:
    """Entries of the calibration side — SEEN labels only.

    A held-out label may not be fitted on, not even a temperature: that
    would put the held-out text back into the fitted surface.
    """
    per = max(1, max_samples // len(DATASETS))
    out = []
    for dataset in DATASETS:
        samples = cut_samples(holdout, dataset, "seen", "calibration", per,
                              cfg)
        out.extend(C.entries_from_samples(engine, samples, "calibration",
                                          "seen", dataset))
    return out


def eval_only_samples(dataset: str, cardinality: int | None = None,
                      limit: int = MAX_EVAL_ONLY, split: str | None = None,
                      seed: int = UNSEEN_SEED) -> list:
    """Samples of an EVAL-ONLY benchmark, straight from its own file.

    `data.optset.OptionSetSampler` refuses logiqa/reclor at the firewall
    barrier — correctly: they must never be sampled for training. The
    option set here is the benchmark's own (no mined distractors), with the
    order reshuffled deterministically so a gold-position bias in the
    source cannot be mistaken for skill. Rows whose gold is hidden
    upstream (`answer == "unknown"`) are dropped: they cannot be scored.
    """
    split = split or EVAL_ONLY_SPLIT.get(dataset, "test")
    out = []
    for n, row in enumerate(iter_rows(dataset, split, None, PREFETCH_DIR)):
        for question in row.get("questions", []):
            options = list(question.get("options") or [])
            answer = question.get("answer")
            if cardinality is not None and len(options) != cardinality:
                continue
            if not options or answer is None:
                continue
            rng = random.Random(f"{seed}\x00{dataset}\x00{n}"
                                f"\x00{question.get('id', '')}")
            rng.shuffle(options)
            ids = [o["id"] for o in options]
            if answer not in ids:
                continue
            out.append(Sample(
                dataset=dataset, row_id=f"{dataset}-{n}",
                question_id=str(question.get("id", "")),
                state=row.get("state", ""), question=question_text(question),
                options=[{"id": o["id"], "text": o["text"]} for o in options],
                answer=answer, gold_index=ids.index(answer)))
    random.Random(f"{seed}\x00pick\x00{dataset}").shuffle(out)
    return out[:limit]


# -- gate criterion 1: no unseen label text in any training row -----------

def unseen_text_absence(holdout: dict, rows_cap: int | None = None,
                        sample_rows: int = 20_000,
                        cfg: SamplerConfig | None = None) -> dict:
    """No held-out label may appear in ANY training row, by NORMALISED TEXT.

    Three layers, all by text and never by id:

    1. **pools** — the restricted distractor pool of each training sampler
       is checked exhaustively. This is the proof: an option can only ever
       come from this pool, so an empty intersection here means no training
       option set can contain a held-out label.
    2. **row golds** — every training row's gold label, exhaustively over
       the rows the trainer loaded.
    3. **composed samples** — a bounded run of the sampler's real output,
       belt-and-braces against a restriction bug that the two static
       checks would not see.

    A fourth, INFORMATIONAL layer counts held-out label texts occurring
    inside the training STATES (a headline may legitimately contain the
    words "world news"). That is input text, not label-space leakage, so it
    is reported and never gates.
    """
    banned: dict = {}
    for dataset, h in sorted(holdout.items()):
        for label in h.unseen:
            key = T.normalise_label(h.texts.get(label, label))
            banned.setdefault(key, []).append(f"{dataset}:{label}")
    train = T.train_samplers(holdout, eval_config(cfg), rows_cap,
                             T.PREFETCH_DIR)
    pool_hits, gold_hits, sample_hits, state_hits = [], [], [], []
    checked = {"pool_labels": 0, "row_golds": 0, "samples": 0, "states": 0}
    for dataset, sampler in sorted(train.items()):
        texts = holdout[dataset].texts
        for option in sampler.pools[dataset]:
            checked["pool_labels"] += 1
            key = T.normalise_label(option["text"])
            if key in banned:
                pool_hits.append({"dataset": dataset, "option": option["id"],
                                  "label": banned[key][0]})
        for row in sampler.rows[dataset]:
            checked["row_golds"] += 1
            for question in row.get("questions", []):
                gold = question.get("answer")
                key = T.normalise_label(texts.get(gold, gold or ""))
                if key in banned:
                    gold_hits.append({"dataset": dataset,
                                      "question": question.get("id"),
                                      "label": banned[key][0]})
        per = max(1, sample_rows // max(len(train), 1))
        for sample in _bounded(sampler, dataset, per):
            checked["samples"] += 1
            for option in sample.options:
                if T.normalise_label(option["text"]) in banned:
                    sample_hits.append({"dataset": dataset,
                                        "row_id": sample.row_id,
                                        "option": option["id"]})
        for row in sampler.rows[dataset][:per]:
            checked["states"] += 1
            norm = T.normalise_label(row.get("state", ""))
            for key, labels in banned.items():
                if key and key in norm:
                    state_hits.append({"dataset": dataset,
                                       "label": labels[0]})
                    break
    hits = pool_hits + gold_hits + sample_hits
    return {
        "pass": not hits,
        "compared_by": "normalised label TEXT, never by option id",
        "banned_texts": len(banned),
        "checked": checked,
        "n_hits": len(hits),
        "hits": {"distractor_pool": pool_hits[:10], "row_gold": gold_hits[:10],
                 "composed_sample": sample_hits[:10]},
        "informational_state_mentions": {
            "n": len(state_hits), "examples": state_hits[:5],
            "note": "a held-out label's words occurring inside a training "
                    "STATE is input text, not label-space leakage; counted, "
                    "never gating"},
    }


def _bounded(sampler, dataset: str, limit: int):
    n = 0
    for sample in sampler.epoch(0):
        yield sample
        n += 1
        if n >= limit:
            return


# -- gate criterion: the cross-lingual arm --------------------------------

def cross_lingual_status(train_gate: Path | None = None) -> dict:
    """Did the training run actually carve the locale holdout?

    The protocol asks for 4 train locales and 2 eval locales. That carve
    belongs to the TRAINING run: it cannot be manufactured at eval time,
    because a locale the run trained on has been seen whatever this module
    does with it. So the status is read off the training gate — and it
    flips to satisfied on its own the day a run publishes
    `holdout.locales`.
    """
    path = train_gate or (ROOT / "artifacts" / "gates" / "T-train-real"
                          / "gate.json")
    carved, run = None, None
    try:
        gate = json.loads(path.read_text())
        run = gate.get("run_id")
        carved = (gate.get("holdout") or {}).get("locales")
    except (OSError, ValueError):
        pass
    ok = bool(carved) and sorted(carved.get("unseen", [])) == sorted(
        EVAL_LOCALES)
    return {
        "pass": ok,
        "protocol": {"train_locales": list(TRAIN_LOCALES),
                     "eval_locales": list(EVAL_LOCALES),
                     "rule": "the last two locales in lexicographic order, "
                             "fixed before the first measurement"},
        "carved_by_training_run": carved,
        "run_id": run,
        "blocker": None if ok else (
            f"run {run!r} trained on all {len(MASSIVE_LOCALES)} MASSIVE "
            "locales: its train split contains it-IT and pt-PT rows, so no "
            "eval can turn them into unseen locales. The per-locale numbers "
            "below are therefore SEEN-locale and are an upper bound on "
            "cross-lingual generalisation. Unblocking it needs one training "
            "run that excludes EVAL_LOCALES from the train samplers — not a "
            "change in this module"),
    }


# -- the report ------------------------------------------------------------

def cut_report(entries: list, cal: dict | None,
               strategies: tuple = ("unknown", "maxprob", "margin")) -> dict:
    """One published cut: raw + calibrated metrics, risk-coverage, CI."""
    if not entries:
        return {"n": 0, "skipped": "no rows in this cut"}
    out = {"raw": C.metrics_of(entries)}
    scored = entries
    if cal:
        scored = C.calibrated(cal, entries)
        out["calibrated"] = C.metrics_of(scored)
        out["calibrated"]["temperature_source"] = _majority(
            [e["temp_source"] for e in scored])
    out["risk_coverage"] = {s: C.risk_coverage(scored, s)
                            for s in strategies}
    out["risk_ci95_unknown"] = C.bootstrap_risk_ci(scored, "unknown")
    out["n"] = len(entries)
    return out


def _majority(values: list) -> str:
    counts: dict = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(sorted(counts), key=lambda k: counts[k]) if counts else ""


def _pair(seen: dict, unseen: dict, labels: tuple | None = None) -> dict:
    """The gap that makes the pair mean something.

    `labels` is `(n_seen_labels, n_unseen_labels)`. It is published next to
    the two mean-K values because the cuts are NOT equally hard by
    construction: the unseen cut draws its distractors from the held-out
    pool alone (so the model cannot win by elimination), and that pool is
    smaller than the seen one. A pair where unseen beats seen usually means
    the unseen cut was the easier draw, and the reader has to be able to
    see that from the artifact instead of guessing it.
    """
    if not seen.get("raw") or not unseen.get("raw"):
        return {}
    s, u = seen["raw"], unseen["raw"]
    return {
        "comparability": {
            "mean_k_seen": s["mean_k"], "mean_k_unseen": u["mean_k"],
            "labels_seen": (labels or (None, None))[0],
            "labels_unseen": (labels or (None, None))[1],
            "chance_seen": s["chance"], "chance_unseen": u["chance"],
            "note": "distractors of the unseen cut come only from the "
                    "held-out labels: a smaller pool, so the two cuts are "
                    "not equally hard. Read the pair with both K and both "
                    "chance rates in hand",
        },
        "accuracy_seen": s["accuracy"], "accuracy_unseen": u["accuracy"],
        "accuracy_drop": round(s["accuracy"] - u["accuracy"], 6),
        "ece_seen": s["ece"], "ece_unseen": u["ece"],
        "ece_rise": round(u["ece"] - s["ece"], 6),
        "brier_seen": s["brier"], "brier_unseen": u["brier"],
        "unseen_beats_chance": bool(
            u["accuracy_ci95"][0] > u["chance"]),
        "unseen_ranking_beats_chance": bool(
            u["accuracy_options_only_ci95"][0] > u["chance_options_only"]),
    }


def headline(table: dict) -> dict:
    """The compact seen/unseen pair the dashboard puts at the top."""
    out = {"cuts": {}}
    for name, cut in sorted(table.items()):
        pair = cut.get("pair") or {}
        if not pair:
            continue
        out["cuts"][name] = {
            "accuracy": [pair["accuracy_seen"], pair["accuracy_unseen"]],
            "ece": [pair["ece_seen"], pair["ece_unseen"]],
            "brier": [pair["brier_seen"], pair["brier_unseen"]],
            "n": [cut["seen"]["n"], cut["unseen"]["n"]],
            "unseen_beats_chance": pair["unseen_beats_chance"],
        }
    overall = table.get("ALL", {})
    if overall.get("pair"):
        out["overall"] = overall["pair"]
    return out


def compose_gate(model_version: str, ckpt_dir: str, table: dict,
                 eval_only: dict, absence: dict, cross: dict,
                 seals: dict, cal: dict, holdout: dict,
                 n_scored: int) -> dict:
    """Every criterion, then `pass` as their AND. No number is softened."""
    paired = {
        "pass": all(
            cut["seen"].get("n") and cut["unseen"].get("n")
            for name, cut in table.items()
            if not cut.get("exempt")),
        "rule": "every non-exempt cut publishes seen AND unseen",
        "cuts": {name: {"seen_n": cut["seen"].get("n", 0),
                        "unseen_n": cut["unseen"].get("n", 0),
                        "exempt": cut.get("exempt")}
                 for name, cut in sorted(table.items())},
    }
    from_model = {
        "pass": bool(model_version) and bool(cal.get("model_version"))
        and cal["predictor"]["name"] == "pointer-decision-head",
        "model_version": model_version,
        "checkpoint": ckpt_dir,
        "predictor": cal.get("predictor", {}).get("name"),
        "rule": "the metric is computed by the trained pointer head of "
                "#T-train-real; an artifact without model_version fails",
        "not": "cosine-char3-softmax (the parameter-free scorer this task "
               "removed from eval/calib.py)",
    }
    sealed = {
        "pass": all(s["shared_groups_metric_vs_calibration"] == 0
                    for s in seals.values()),
        "rule": "the metric/calibration partition is a GROUP split "
                "(skeleton+domain+language), seeded and sha256-sealed; "
                "no row index decides anything",
        "seed": UNSEEN_SEED,
        "splits": seals,
    }
    criteria = {
        "unseen_text_absent_from_train": absence,
        "metrics_from_trained_checkpoint": from_model,
        "paired_seen_unseen_table": paired,
        "group_split_sealed": sealed,
        "cross_lingual_holdout": cross,
    }
    failed = sorted(k for k, v in criteria.items() if not v.get("pass"))
    return {
        "format": 1,
        "task": TASK,
        "generated_utc": utcnow(),
        "model_version": model_version,
        "checkpoint": ckpt_dir,
        "calibration": {
            "artifact": os.path.relpath(C.MODEL_CALIB, ROOT),
            "model_version": cal.get("model_version"),
            "temperature": cal.get("global", {}).get("temperature"),
            "nll_before": cal.get("global", {}).get("nll_before"),
            "nll_after": cal.get("global", {}).get("nll_after"),
            "fit_split": cal.get("fit_split"),
            "fit_cut": cal.get("fit_cut"),
            "n_fit": cal.get("n_fit"),
        },
        "pass": not failed,
        "failed_criteria": failed,
        "verdict": _verdict(failed, table),
        "honesty": "numbers are published as measured — a bad number is "
                   "published bad, and a pass:false with a truthful table "
                   "is the correct outcome of this gate",
        "protocol": {
            "label_holdout": T.holdout_report(holdout),
            "cross_lingual": cross["protocol"],
            "row_partition": {"by": ["skeleton", "domain", "language"],
                              "seed": UNSEEN_SEED,
                              "metric_frac": METRIC_FRAC,
                              "calibration_frac": CALIB_FRAC,
                              "never": "row index (i % 10)"},
            "eval_only": {"datasets": list(EVAL_ONLY),
                          "firewall": "#T-halt-contam: refused at "
                                      "data.optset.assert_trainable",
                          "thermometer_floor": THERMOMETER_FLOOR},
        },
        "criteria": criteria,
        "n_scored": n_scored,
        "table": table,
        "eval_only_results": eval_only,
        "headline": headline(table),
    }


def _verdict(failed: list, table: dict) -> str:
    allcut = (table.get("ALL") or {}).get("pair") or {}
    core = ""
    if allcut:
        core = (f"unseen accuracy {allcut['accuracy_unseen']} vs seen "
                f"{allcut['accuracy_seen']} (drop "
                f"{allcut['accuracy_drop']}), unseen ECE "
                f"{allcut['ece_unseen']} vs seen {allcut['ece_seen']}")
    if not failed:
        return f"PASS — {core}"
    return (f"FAIL [{', '.join(failed)}] — {core}. The table is published "
            "as measured; the failing criteria say what is missing, not "
            "that the numbers were massaged")


# -- the run ---------------------------------------------------------------

def run(ckpt_dir: str, device: str = "auto", reseal: bool = False,
        max_samples: dict | None = None, write: bool = True) -> dict:
    """Score every cut once, fit the calibrator, write the gate."""
    caps = {**MAX_SAMPLES, **(max_samples or {})}
    engine, manifest = T.load_checkpoint(ckpt_dir, device)
    model_version = manifest.get("model_version")
    holdout = T.build_holdout()
    cfg = SamplerConfig()

    seals = {d: seal_metric_split(d, force=reseal) for d in DATASETS}

    cal_entries = calibration_entries(engine, holdout, cfg)
    cal = C.fit_calibrator(cal_entries, C.predictor_card(manifest, ckpt_dir))
    if write:
        C.save_calibration(cal)

    table: dict = {}
    entries_by_cut: dict = {"seen": [], "unseen": []}
    massive: dict = {}
    for dataset in DATASETS:
        limit = caps.get(dataset, DEFAULT_MAX_SAMPLES)
        cut: dict = {}
        for which in ("seen", "unseen"):
            samples = cut_samples(holdout, dataset, which, "test", limit, cfg)
            entries = C.entries_from_samples(engine, samples, "test", which,
                                             dataset)
            if dataset == "massive":
                massive[which] = entries
            entries_by_cut[which].extend(entries)
            cut[which] = cut_report(entries, cal)
        cut["pair"] = _pair(cut["seen"], cut["unseen"],
                            (len(holdout[dataset].seen),
                             len(holdout[dataset].unseen)))
        if not holdout[dataset].unseen:
            cut["exempt"] = holdout[dataset].exempt
        table[dataset] = cut

    # the cross-lingual arm, carved out of the MASSIVE entries already
    # scored: no second pass over the model, same rows, split by locale
    for name, locales in (("massive@train-locales", TRAIN_LOCALES),
                          ("massive@eval-locales", EVAL_LOCALES)):
        cut = {}
        for which in ("seen", "unseen"):
            sub = [e for e in massive.get(which, [])
                   if e.get("locale") in locales]
            cut[which] = cut_report(sub, cal)
        cut["pair"] = _pair(cut["seen"], cut["unseen"],
                            (len(holdout["massive"].seen),
                             len(holdout["massive"].unseen)))
        cut["locales"] = list(locales)
        cut["caveat"] = ("these locales were IN the training split of "
                         f"run {manifest.get('run_id')}: the cut is "
                         "seen-locale, and the label holdout is the only "
                         "unseen dimension here")
        table[name] = cut

    table["ALL"] = {
        "seen": cut_report(entries_by_cut["seen"], cal),
        "unseen": cut_report(entries_by_cut["unseen"], cal),
        "note": "the whole mixture, every dataset pooled",
    }
    table["ALL"]["pair"] = _pair(
        table["ALL"]["seen"], table["ALL"]["unseen"],
        (sum(len(h.seen) for h in holdout.values()),
         sum(len(h.unseen) for h in holdout.values())))

    eval_only = {}
    for name, dataset, card in (("logiqa-mc", "logiqa", 4),
                                ("logiqa-nli", "logiqa", 2),
                                ("reclor", "reclor", 4)):
        samples = eval_only_samples(dataset, card, MAX_EVAL_ONLY)
        entries = C.entries_from_samples(engine, samples, "test", "eval-only",
                                         name)
        rep = cut_report(entries, cal)
        if rep.get("n"):
            raw = rep["raw"]
            rep["never_in_train"] = "#T-halt-contam firewall"
            rep["date"] = utcnow()[:10]
            rep["model_version"] = model_version
            rep["cardinality"] = card
            rep["split"] = EVAL_ONLY_SPLIT.get(dataset)
            rep["split_note"] = EVAL_ONLY_SPLIT_NOTE.get(dataset)
            rep["above_chance"] = bool(
                raw["accuracy_ci95"][0] > raw["chance"])
            rep["ranking_above_chance"] = bool(
                raw["accuracy_options_only_ci95"][0]
                > raw["chance_options_only"])
            rep["above_thermometer_floor"] = bool(
                card == 4 and raw["accuracy_options_only_ci95"][0]
                > THERMOMETER_FLOOR)
        eval_only[name] = rep

    absence = unseen_text_absence(holdout, None, cfg=cfg)
    cross = cross_lingual_status()
    n_scored = (len(entries_by_cut["seen"]) + len(entries_by_cut["unseen"])
                + len(cal_entries)
                + sum(r.get("n", 0) for r in eval_only.values()))
    gate = compose_gate(model_version, os.path.relpath(ckpt_dir, ROOT),
                        table, eval_only, absence, cross, seals, cal,
                        holdout, n_scored)
    if write:
        GATE_DIR.mkdir(parents=True, exist_ok=True)
        GATE_PATH.write_text(json.dumps(gate, indent=2) + "\n")
    return gate


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(prog="eval.unseen")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate", help="score every cut and write gate.json")
    g.add_argument("--checkpoint", required=True)
    g.add_argument("--device", default="auto")
    g.add_argument("--reseal", action="store_true",
                   help="re-seal the group partitions on purpose")
    h = sub.add_parser("holdout-clean",
                       help="criterion 1 only (no torch needed)")
    h.add_argument("--rows", type=int, default=0,
                   help="0 = every training row the trainer loads")
    s = sub.add_parser("seal", help="seal the group partitions only")
    s.add_argument("--reseal", action="store_true")
    for p in (g, h, s):
        p.set_defaults(_p=p)
    args = ap.parse_args(argv)
    if args.cmd == "gate":
        gate = run(args.checkpoint, args.device, args.reseal)
        print(json.dumps({"pass": gate["pass"],
                          "failed": gate["failed_criteria"],
                          "verdict": gate["verdict"],
                          "headline": gate["headline"]}, indent=2))
        return 0
    if args.cmd == "seal":
        print(json.dumps({d: seal_metric_split(d, force=args.reseal)["counts"]
                          for d in DATASETS}, indent=2))
        return 0
    rep = unseen_text_absence(T.build_holdout(), args.rows or None)
    print(json.dumps(rep, indent=2))
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
