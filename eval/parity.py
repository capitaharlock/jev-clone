"""Protocol parity with the published Jev recipe — #T-jev-parity.

There is one number from outside this repo on the table: **0.924**
(2 846 / 3 080) on the official BANKING77 test messages with all 77 intents
present. Until now it sat in `eval/fullspace.py` as a reference with
`same_rows: false` stamped beside it and the words "different rows", which
is not what is actually known. What is known is narrower and worse:

* the teacher was **never scored through our pipeline** — its accuracy is
  read off a published report, not produced here;
* there is **no row-by-row evidence** from its side to align against ours,
  so neither "same rows" nor "different rows" is established;
* and the information regime differs (definitions + 24 retrieved examples
  against raw identifiers), which is a difference of protocol even when the
  rows and the cardinality match.

So this module stops describing the gap and enumerates it. It publishes our
number on the 3 080 official rows at K=77 with chance printed beside it, the
95 % interval, a stable identifier per row so "same rows" becomes checkable
from our side, the gold seen/unseen split inside the same K, the three
information-regime arms of `#T-option-text` as numbers, the contamination
audit at the precision the evidence supports, and every protocol difference
as its own entry with `matched: true | false | unverifiable`.

Two numbers travel with every run from here on: the distance to the teacher
(0.924) and the distance to our own first 77-way number (0.0123, 38/3 080,
`artifacts/gates/T-teacher-probe/fullspace.json`), which is kept frozen in
`BASELINE` so a later run cannot quietly re-baseline itself.

Automatic, not manual (done-when 6). `eval.fullspace gate` emits this
artifact from the rows it already scored — no second forward pass, no second
read of the reserved cut — and `eval.scoreboard` attaches the published
headline through `attach()`. Standalone it runs on any checkpoint by path.

CLI:
    CKPT=artifacts/checkpoints/decision/<run>/stage-000250000
    .venv-train/bin/python -m eval.parity gate --checkpoint $CKPT \\
        --device cpu --reason "why the reserved cut is being read"
    python3 -m eval.parity contamination     # stdlib only, no torch
    python3 -m eval.parity show
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TASK = "T-jev-parity"
GATE_DIR = ROOT / "artifacts" / "gates" / TASK
GATE_PATH = GATE_DIR / "parity.json"
BY_CHECKPOINT_DIR = GATE_DIR / "by-checkpoint"
ROWS_DIR = GATE_DIR / "rows"
CONTAMINATION_PATH = GATE_DIR / "contamination.json"

OPTIONTEXT_GATE = ROOT / "artifacts" / "gates" / "T-option-text" / \
    "optiontext.json"

DATASET = "banking77"
OFFICIAL_TEST_ROWS = 3080
CARDINALITY = 77
CHANCE = round(1.0 / CARDINALITY, 6)

#: The first 77-way number this repo ever published, frozen. Every run
#: publishes its distance to THIS as well as to the teacher, so "we moved"
#: is a subtraction a reader can do rather than a claim they have to accept.
#: Copied from the artifact named below, not recomputed: a baseline that is
#: re-derived from whatever code is on disk today is not a baseline.
BASELINE = {
    "what": "the first accuracy this repo published at the full BANKING77 "
            "cardinality, and the one the phase-2 plan is written against",
    "published_as": "0,0123",
    "n": OFFICIAL_TEST_ROWS,
    "hits": 38,
    "cardinality": CARDINALITY,
    "chance": CHANCE,
    "accuracy": 0.012338,
    "accuracy_ci95": [0.009002, 0.016888],
    "beats_chance": False,
    "reading": "indistinguishable from chance — the 95 % interval CONTAINS "
               "the chance rate. Not 'below chance': below would be a "
               "measured bias, and what is measured is the absence of "
               "signal",
    "checkpoint": "artifacts/checkpoints/decision/"
                  "leverstack-d512-prior-ettin-68m-s20260922/"
                  "stage-001000000",
    "artifact": "artifacts/gates/T-teacher-probe/fullspace.json",
    "first_published": "2026-09-23",
    "regime": "trained at the sampler's K=3-8 with the backbone frozen "
              "(phase 1); scored here at K=77. R4: it is a phase-1 "
              "checkpoint measured under the phase-2 metric, which is "
              "exactly what makes it the baseline and not a target",
}


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def teacher() -> dict:
    """The cited teacher number — one definition, in `eval.fullspace`."""
    from eval import fullspace as F

    return dict(F.REFERENCES[DATASET][0])


def references() -> list:
    from eval import fullspace as F

    return [dict(r) for r in F.REFERENCES.get(DATASET, [])]


def _read(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def sample_id(sample) -> str:
    """The identifier a row is named by, everywhere in this repo.

    `dataset:dataset-<n>:<question id>`, where `n` is the row's position
    within its split. It is what `eval.cuts` seals, what
    `eval.fullspace.full_samples` builds, and now what the parity artifact
    publishes a digest of — so "the same rows" is a comparison somebody can
    make instead of a sentence somebody has to believe.
    """
    return f"{sample.dataset}:{sample.row_id}:{sample.question_id}"


# ------------------------------------------------ the differences, one by one

def protocol_differences(regime: dict, contamination: dict) -> list:
    """Every way our measurement differs from the teacher's, enumerated.

    `matched` is the field a reader scans: `true` means the two protocols
    agree on that dimension, `false` means they demonstrably differ, and
    `"unverifiable"` means nobody can check it from either side — which is a
    third state and gets said, not rounded to one of the other two.
    """
    arms = (regime or {}).get("arms") or {}
    budget = (regime or {}).get("budget_finding") or {}
    ident = (contamination or {}).get("labels_found_as_identifier_total")
    gold = (contamination or {}).get("labels_found_as_answer_total")
    return [
        {
            "id": "label-space",
            "dimension": "cardinality of the decision",
            "teacher": "all 77 BANKING77 intents present on every row",
            "ours": f"all {CARDINALITY} intents present on every row, "
                    f"chance {CHANCE}",
            "matched": True,
            "evidence": "every row is built with the whole sorted label "
                        "space as its options (`eval.fullspace."
                        "full_samples`), and `cardinality` is published per "
                        "row-set",
        },
        {
            "id": "split",
            "dimension": "which rows",
            "teacher": "the 3 080 official test messages, per the published "
                       "report",
            "ours": f"the {OFFICIAL_TEST_ROWS} official test messages, "
                    "sealed by sha256 as `T-eval-cardinality-banking77-test`",
            "matched": "unverifiable",
            "evidence": "our side names every row it scored (`rows.ids_"
                        "sha256`, and the per-row file); the teacher's side "
                        "publishes a count and a split name, not a row list, "
                        "so the two sets cannot be compared",
        },
        {
            "id": "scored-by",
            "dimension": "who produced the number",
            "teacher": "the teacher's own harness. It has never been scored "
                       "through this repo's pipeline",
            "ours": "`eval.calib.entries_from_samples` over a checkpoint in "
                    "this repo, named by sha256",
            "matched": False,
            "evidence": "`#T-teacher-probe` is the task that would score the "
                        "teacher on our rows with our scorer; it needs a "
                        "working API key and has not run",
        },
        {
            "id": "row-evidence",
            "dimension": "row-by-row evidence",
            "teacher": "none published: an aggregate accuracy, no per-row "
                       "predictions",
            "ours": "one line per row (id, gold, prediction, p(gold), rank "
                    "of the gold) in the `rows.file` this artifact names",
            "matched": False,
            "evidence": "with no per-row list on the other side there is "
                        "nothing to align ours to, which is precisely why "
                        "`same_rows` is UNVERIFIED rather than false",
        },
        {
            "id": "option-text",
            "dimension": "what the options say (R8)",
            "teacher": "a natural-language definition of each of the 77 "
                       "categories",
            "ours": "the raw label identifier — `Option(id=l, text=l)`, "
                    "`data/adapters.py:146`",
            "matched": False,
            "measured": ("#T-option-text arm B replaces the identifiers "
                         "with the teacher's own definitions: "
                         + _arm_line(arms, "B")),
        },
        {
            "id": "labelled-examples",
            "dimension": "demonstrations in the context (R8)",
            "teacher": "up to 24 BM25-retrieved labelled training examples "
                       "per prediction",
            "ours": "none — the row's own text and nothing else",
            "matched": False,
            "measured": ("#T-option-text arm C retrieves them the same way: "
                         + _arm_line(arms, "C")),
        },
        {
            "id": "context-budget",
            "dimension": "how much of that context survives",
            "teacher": "a context window large enough for the whole prompt",
            "ours": "256 tokens (`train_decision.TRAIN_MAX_LENGTH`), applied "
                    "by `eval/calib.py`",
            "matched": False,
            "measured": (
                f"{budget.get('examples_complete_mean')} of "
                f"{budget.get('examples_offered_mean')} retrieved examples "
                f"survive the {budget.get('window_tokens')}-token window, "
                f"and in the teacher's own order the query itself survives "
                f"in {budget.get('query_survives_share_teacher_order')} of "
                "rows. The teacher's regime does not fit in this window, "
                "which is a result and not an excuse"
                if budget else
                "not measured in this read — see "
                "artifacts/gates/T-option-text/optiontext.json"),
        },
        {
            "id": "abstention",
            "dimension": "may the model decline to answer",
            "teacher": "no: it names one of the 77 categories on every row",
            "ours": "yes: the head scores [K + 1] and `unknown` can win, so "
                    "our published accuracy is not a forced choice",
            "matched": False,
            "evidence": "`abstain_rate` and `accuracy_options_only` (the "
                        "same ranking with `unknown` taken out of the race) "
                        "are published beside the headline, so the forced "
                        "comparison is available without being the headline",
        },
        {
            "id": "weight-updates",
            "dimension": "training on this label space",
            "teacher": "none claimed: no weight updates for the task",
            "ours": "none on this label space — the head trained on 12 other "
                    "sources at the sampler's K=3-8, with banking77 fenced",
            "matched": True,
            "evidence": "`gold_exposure` in this artifact, computed from "
                        "`run.json` and the contamination scan rather than "
                        "asserted",
        },
        {
            "id": "training-data-audit",
            "dimension": "can the training data be inspected",
            "teacher": "no. BANKING77 is public and nobody outside can look "
                       "at what the teacher was trained on, so its number is "
                       "not a held-out claim",
            "ours": "yes for the 12 sources of the mixture: the fence is "
                    "declared in `run.json` and the label strings are "
                    "scanned for literally",
            "matched": False,
            "measured": (f"{ident} of {CARDINALITY} label strings occur as "
                         f"identifiers in the other sources and {gold} as "
                         "gold answers there — the fence excludes the "
                         "DATASET, this scan is what excludes the STRINGS"
                         if ident is not None else
                         "scan not attached to this read"),
        },
        {
            "id": "backbone-exposure",
            "dimension": "what the pretrained backbone had already seen",
            "teacher": "unknown and unauditable",
            "ours": "unknown and unauditable — a public web-scale pretrain "
                    "is not something this repo can inspect either",
            "matched": "unverifiable",
            "evidence": "declared separately from the corpus fence on "
                        "purpose: the fence is evidence, this is an absence "
                        "of evidence, and merging them would let the second "
                        "borrow the credibility of the first",
        },
    ]


def _arm_line(arms: dict, key: str) -> str:
    arm = (arms or {}).get(key)
    if not arm:
        return ("not attached to this read — see "
                "artifacts/gates/T-option-text/optiontext.json")
    return (f"{arm['hits']}/{arm['n']} = {arm['accuracy']} CI "
            f"{arm['accuracy_ci95']} against chance {arm['chance']} at "
            f"K={arm['cardinality']} on the {arm['cut']} cut")


def same_rows_reading(rows: dict) -> dict:
    """What `same_rows: false` means here, spelled out (done-when 3).

    The value stays a boolean `false` because rule R3 says a citation is
    stamped that way and machine readers across this repo test it as one.
    What changes is that the file no longer says "different rows", which
    claims a fact nobody established. Not-verified is a third state and this
    block is where it is written down. The list of what is unverified is
    `eval.fullspace.SAME_ROWS_NOT_VERIFIED`, shared so the two gates cannot
    tell two different stories about the same absence of evidence.
    """
    from eval import fullspace as F

    return {
        "value_means": "NOT ESTABLISHED. It does not say the rows differ — "
                       "nobody has compared them. `false` reads as 'this is "
                       "a citation, not a same-rows measurement' (R3)",
        "verified": False,
        "what_is_verified_on_our_side": [
            f"the rows we scored are named one by one: {rows.get('n')} ids "
            f"of the form {rows.get('scheme')}, digest "
            f"{str(rows.get('ids_sha256'))[:16]}…",
            "they are the sealed official test cut "
            "(`artifacts/splits/T-eval-cardinality-banking77-test`), whose "
            "seal is a content digest and not an intention",
            "one line per row with its gold, our prediction and the gold's "
            "rank is written to the file this artifact names, so anybody "
            "can recompute our accuracy from the rows themselves",
        ],
        "what_is_NOT_verified": list(F.SAME_ROWS_NOT_VERIFIED),
        "what_would_verify_it": "either #T-teacher-probe scoring the teacher "
                                "on these exact ids through our scorer (it "
                                "needs a working API key), or a per-row "
                                "prediction file published by the teacher "
                                "that can be aligned to `rows.file`",
    }


# --------------------------------------------------------------- the rows

def row_records(samples: list, entries: list, seen: set) -> list:
    """One record per scored row — the evidence behind the accuracy."""
    out = []
    for sample, entry in zip(samples, entries):
        k = len(sample.options)
        probs = entry["probs"]
        pred = entry["pred"]
        gold = sample.options[sample.gold_index]["id"]
        p_gold = probs[sample.gold_index]
        out.append({
            "id": sample_id(sample),
            "gold": gold,
            "gold_seen_in_training": gold in seen,
            "predicted": "⟨unknown⟩" if pred >= k else sample.options[pred]["id"],
            "correct": bool(pred == sample.gold_index),
            "p_gold": round(p_gold, 6),
            "gold_rank": 1 + sum(1 for j in range(k) if probs[j] > p_gold),
            "cardinality": k,
        })
    return out


def row_identity(records: list, seal: dict, path: Path | None) -> dict:
    """The block that makes "the same rows" a checkable claim."""
    from eval import splits as S

    ids = [r["id"] for r in records]
    digest = S.sha256_text("".join(f"{i}\n" for i in sorted(ids)))
    sealed_ids = set(seal.get("ids") or [])
    return {
        "scheme": "<dataset>:<dataset>-<row index within the split>:"
                  "<question id>",
        "why": "so 'same rows' is a set comparison and not an assertion — "
               "and so a future run that scores a different subset cannot "
               "publish its number under the same heading",
        "n": len(ids),
        "unique": len(set(ids)) == len(ids),
        "ids_sha256": digest,
        "sample": sorted(ids)[:3],
        "file": os.path.relpath(path, ROOT) if path else None,
        "file_contents": "one JSON object per row: id, gold, whether that "
                         "gold was seen in training, our prediction, "
                         "whether it was right, p(gold) and the gold's rank",
        "sealed_cut": {
            "rows_in_seal": seal.get("rows"),
            "split_sha256": seal.get("split_sha256"),
            "covers_the_whole_sealed_cut": bool(
                sealed_ids and set(ids) == sealed_ids),
            "rows_scored_not_in_seal": sorted(set(ids) - sealed_ids)[:5],
        } if seal else None,
    }


# ------------------------------------------------------- seen / unseen gold

def seen_labels(run: dict, contamination: dict, labels: list) -> dict:
    """Which of the 77 golds this run could have learned, and on what basis.

    Two independent routes make a label 'seen', and they are different
    claims, so they are reported apart:

    * the banking77 DATASET was in the mixture (the fence was off) — every
      label of the space is then seen;
    * the label STRING occurs as a gold answer in one of the other sources,
      which makes that one label seen even with banking77 fenced.

    Everything else is unseen, and `basis` says which route decided it. This
    is what lets the scoreboard split gold-seen from gold-unseen INSIDE the
    same K=77 the moment a future run trains on banking77.
    """
    fence = run.get("fence") or {}
    fenced = list(fence.get("fenced_datasets") or [])
    datasets = list(run.get("datasets") or [])
    dataset_in_mixture = DATASET in datasets and DATASET not in fenced
    as_answer = ((contamination or {}).get("labels_found_as_answer")
                 or {})
    seen = set(labels) if dataset_in_mixture else set(as_answer)
    return {
        "seen": sorted(seen),
        "n_seen": len(seen),
        "n_unseen": len(labels) - len(seen),
        "basis": {
            "dataset_in_mixture": dataset_in_mixture,
            "dataset_evidence": (
                f"run.json: fence.clean_1m={fence.get('clean_1m')}, "
                f"fenced_datasets={fenced}; the run's training sources are "
                f"{datasets}"),
            "label_string_as_gold_in_another_source": sorted(as_answer),
            "label_string_evidence": (contamination or {}).get("artifact"),
        },
        "why_two_routes": "excluding the DATASET is not the same claim as "
                          "the label strings being absent from the other "
                          "sources; a run that fences banking77 and trains "
                          "on a source that happens to use the same intent "
                          "identifiers has seen those labels",
    }


def exposure_split(samples: list, entries: list, seen: set) -> dict:
    """The same K=77 number, split by whether the gold was seen in training.

    Not two cuts and not two cardinalities: the identical rows, scored once,
    partitioned by a property of the gold. With banking77 fenced the seen
    side is empty and says so; the block exists so the day a run trains on
    this label space, the scoreboard does not publish one number over two
    populations.
    """
    from eval import fullspace as F

    pairs = list(zip(samples, entries))
    sides = {"unseen_gold": [], "seen_gold": []}
    for sample, entry in pairs:
        gold = sample.options[sample.gold_index]["id"]
        sides["seen_gold" if gold in seen else "unseen_gold"].append(
            (sample, entry))
    out = {}
    for name, rows in sides.items():
        if not rows:
            out[name] = {
                "n": 0,
                "cardinality": CARDINALITY,
                "chance": CHANCE,
                "why_empty": "no row of this cut has a gold on that side of "
                             "the fence in this run",
            }
            continue
        out[name] = F.tally([s for s, _ in rows], [e for _, e in rows])
    out["how"] = ("the same rows, the same K, the same forward pass — "
                  "partitioned by whether the row's GOLD label was seen in "
                  "training, not by cut and not by cardinality")
    out["covers_every_row"] = (
        (out["seen_gold"].get("n") or 0) + (out["unseen_gold"].get("n") or 0)
        == len(pairs))
    return out


# -------------------------------------------------- the information regime

def information_regime(path: Path | None = None) -> dict:
    """The three #T-option-text arms, hooked in as numbers (done-when 2).

    The task's phrase is "so the difference of information regime is a
    number and not a footnote". That number already exists — one checkpoint,
    one sealed cut, K=77 throughout — so this reads it rather than measuring
    it again. Every arm is carried with the cut it was scored on — the
    reserved-cut reading where there is one, since those are the very rows
    this artifact is about, and the development reading for the arms the
    reserved cut was never read for (arm C, which is the entire
    labelled-examples difference). `arithmetic` says which of them may be
    subtracted from which.
    """
    doc = _read(path or OPTIONTEXT_GATE)
    if not doc:
        return {
            "measured": False,
            "why": "no artifact at "
                   f"{os.path.relpath(path or OPTIONTEXT_GATE, ROOT)}",
            "how": ".venv-train/bin/python -m eval.optiontext run "
                   "--checkpoint <ckpt>",
        }
    conf = doc.get("reserved_cut_confirmation") or {}
    # Every arm, each carrying the cut it was scored on: the reserved-cut
    # reading wins where there is one (those are THESE rows), and the
    # development arms fill in the ones the reserved cut was never read for
    # — arm C among them, which is the whole labelled-examples difference.
    # Dropping them would leave that difference described instead of priced.
    arms, by_cut = {}, {}
    for source in (doc, conf):
        cut = (source.get("cut") or {}).get("name")
        for key, arm in (source.get("arms") or {}).items():
            block = _arm_block(arm, source)
            arms[key] = block
            by_cut.setdefault(cut, {})[key] = block
    return _regime_doc(doc, conf, arms, by_cut, path)


def _arm_block(arm: dict, source: dict) -> dict:
    return {
        "arm": arm.get("arm"),
        "name": arm.get("name"),
        "cut": (source.get("cut") or {}).get("name"),
        "reserved_cut": (source.get("cut") or {}).get("reserved"),
        "option_text": (arm.get("protocol") or {}).get("option_text"),
        "labelled_examples_in_state": (
            (arm.get("protocol") or {}).get("labelled_examples_in_state")),
        "n": arm.get("n"),
        "hits": arm.get("hits"),
        "cardinality": arm.get("cardinality"),
        "chance": arm.get("chance"),
        "accuracy": arm.get("accuracy"),
        "accuracy_ci95": arm.get("accuracy_ci95"),
        "beats_chance": arm.get("beats_chance"),
    }


def _within_cut(by_cut: dict, baseline: str = "A") -> dict:
    """Which arms may be subtracted from which — and the one subtraction.

    Arm A is on the reserved cut and arm C on the development cut, so
    `C - A` across them is not a contrast, it is two measurements with a
    difference of rows baked in. The grouping says so, and the delta is only
    computed inside a group that holds the baseline.
    """
    out = {
        "comparable_within": {cut: sorted(arms)
                              for cut, arms in sorted(by_cut.items())},
        "why": "a difference between two arms is readable only when both "
               "were scored on the same rows; two arms on different cuts "
               "are two measurements and their difference carries the "
               "change of rows as well as the change of protocol",
        "deltas": {},
    }
    for cut, arms in sorted(by_cut.items()):
        if baseline not in arms:
            continue
        base = arms[baseline]
        for key in sorted(k for k in arms if k != baseline):
            arm = arms[key]
            out["deltas"][f"{key} - {baseline} on {cut}"] = {
                "cut": cut,
                "n": arm["n"],
                "cardinality": arm["cardinality"],
                "chance": arm["chance"],
                "point_shift": round((arm["accuracy"] or 0.0)
                                     - (base["accuracy"] or 0.0), 6),
                "intervals_overlap": _overlap(arm["accuracy_ci95"],
                                              base["accuracy_ci95"]),
                "reading": "a point shift inside overlapping intervals is "
                           "not a move: the information regime changed and "
                           "the number did not",
            }
    return out


def _regime_doc(doc: dict, conf: dict, arms: dict, by_cut: dict,
                path) -> dict:
    dev_c = ((doc.get("arms") or {}).get("C") or {}).get("context_budget") or {}
    dev_ct = (((doc.get("arms") or {}).get("C-teacher-order") or {})
              .get("context_budget") or {})
    budget = {}
    if dev_c:
        budget = {
            "window_tokens": dev_c.get("window_tokens"),
            "examples_offered_mean": dev_c.get("examples_offered_mean"),
            "examples_complete_mean": dev_c.get("examples_complete_mean"),
            "examples_complete_share": dev_c.get("examples_complete_share"),
            "query_survives_share": dev_c.get("query_survives_share"),
            "query_survives_share_teacher_order": dev_ct.get(
                "query_survives_share"),
            "measured_on": (doc.get("cut") or {}).get("name"),
            "reading": "R8 — the teacher's regime does not fit in this "
                       "window, so 'we reproduced the teacher's protocol' "
                       "is false by budget before it is false by anything "
                       "else",
        }
    verdict = doc.get("verdict") or {}
    return {
        "measured": True,
        "artifact": os.path.relpath(path or OPTIONTEXT_GATE, ROOT),
        "task": "#T-option-text",
        "question": "how much of the distance to the teacher is bought by "
                    "WHAT THE MODEL IS SHOWN",
        "arms_quoted_from": ("each arm carries its own cut: "
                             + ", ".join(f"{k} on {v['cut']}"
                                         for k, v in sorted(arms.items()))
                             + ". The reserved cut is the rows this artifact "
                             "scores; a development arm beside it is the "
                             "same protocol measured on other rows of the "
                             "same population, and is labelled as such"),
        "checkpoint": (conf or doc).get("checkpoint"),
        "arms": arms,
        "arms_by_cut": by_cut,
        "arithmetic": _within_cut(by_cut),
        "budget_finding": budget,
        "point_shift_best_vs_identifiers": verdict.get(
            "point_shift_best_vs_baseline"),
        "share_of_gap_to_teacher_explained": verdict.get(
            "share_of_gap_explained_by_input"),
        "arms_that_move_the_number": verdict.get("arms_that_move_the_number"),
        "reading": verdict.get("line"),
    }


# ----------------------------------------------------------- contamination

def _trie(words: list) -> dict:
    root: dict = {}
    for word in words:
        node = root
        for ch in word:
            node = node.setdefault(ch, {})
        node[""] = {}
    return root


def _trie_pattern(node: dict) -> str:
    alts, terminal = [], False
    for ch in sorted(node):
        if ch == "":
            terminal = True
            continue
        alts.append(re.escape(ch) + _trie_pattern(node[ch]))
    if not alts:
        return ""
    body = alts[0] if len(alts) == 1 else "(?:" + "|".join(alts) + ")"
    return f"(?:{body})?" if terminal else body


def label_regex(labels: list):
    """One pass, both spellings: `card_arrival` and `card arrival`.

    A 77-way alternation makes Python's engine walk every branch; the same
    77 words as a character trie let it dispatch on the first byte, which is
    the difference between minutes and hours over 2.4 GB of corpus. The
    separator class is what catches the readable form as well as the
    identifier, and the lookarounds keep `card_arrival` out of
    `card_arrival_x`.
    """
    pattern = _trie_pattern(_trie(sorted(x.lower() for x in labels)))
    pattern = pattern.replace("_", "[_ ]")
    return re.compile((r"(?<![a-z0-9_])(?:" + pattern + r")(?![a-z0-9_])")
                      .encode())


#: how many offending lines one source may contribute to the artifact
MAX_HIT_EXAMPLES = 8
SCAN_CHUNK = 1 << 23

#: The two spellings a label can appear in, and they are NOT the same
#: finding. `card_arrival` in another source is an identifier — that source
#: could have had this exact label in its own option space, which puts it in
#: the gradient. `card arrival` is an English phrase, and the phrase "exchange
#: rate" occurring in a news headline is a collision of language, not
#: contamination. Merging the two counts would turn the second into the
#: first, so they are counted apart and the reading says which one it found.
FORM_IDENTIFIER = "identifier"
FORM_PHRASE = "readable_phrase"


def _form(matched: str, label: str) -> str:
    return FORM_IDENTIFIER if matched == label.lower() else FORM_PHRASE


def _label_of(matched: str) -> str:
    """`card arrival` and `card_arrival` both name the label `card_arrival`."""
    return matched.replace(" ", "_")


def _bump(store: dict, label: str, key: str) -> None:
    store.setdefault(label, {})
    store[label][key] = store[label].get(key, 0) + 1


def _where(line: bytes, matched: str) -> dict:
    """Is the hit a LABEL of that source, or a word in somebody's text?

    The distinction is the whole point of the scan: `card_arrival` appearing
    as an option id in another dataset means that label was trainable there;
    the same string inside a comment means a phrase collided. Rare enough to
    parse the line when it happens.
    """
    try:
        row = json.loads(line)
    except ValueError:
        return {"where": "unparseable line", "matched": matched}
    where = []
    for question in row.get("questions", []) or []:
        if (question.get("answer") or "").lower() == matched:
            where.append("answer")
        if any((o.get("id") or "").lower() == matched
               for o in question.get("options", []) or []):
            where.append("option_id")
    if matched in (row.get("state") or "").lower():
        where.append("state_text")
    return {
        "where": sorted(set(where)) or ["elsewhere in the line"],
        "matched": matched,
        "excerpt": (row.get("state") or "")[:160],
    }


def scan_files(paths: list, labels: list, rx=None) -> dict:
    """Literal occurrences of the label strings in one source's shards.

    Chunked, not line-by-line: 2.4 GB of corpus through `bytes.lower()` and
    one trie regex per 8 MiB block is the difference between minutes and an
    afternoon. The chunk boundary is pulled back to the last newline so no
    match is cut in half, and the per-line work only happens for the blocks
    that actually matched — which, on this corpus, is very few of them.
    """
    rx = rx or label_regex(labels)
    counts: dict = {}
    as_answer: dict = {}
    as_option: dict = {}
    examples: list = []
    size = 0
    started = time.perf_counter()
    for path in paths:
        size += Path(path).stat().st_size
        with open(path, "rb") as fh:
            tail = b""
            while True:
                chunk = fh.read(SCAN_CHUNK)
                if not chunk:
                    break
                buf = tail + chunk
                cut = buf.rfind(b"\n") + 1
                tail, block = ((buf[cut:], buf[:cut]) if cut
                               else (b"", buf))
                low = block.lower()
                if not rx.search(low):
                    continue
                for line in low.split(b"\n"):
                    hits = rx.findall(line)
                    if not hits:
                        continue
                    # occurrences count HITS; placement is decided once per
                    # (line, label), so `as_gold_answer` counts ROWS in
                    # which the label is the gold and not how many times
                    # the string happens to appear in that row's JSON
                    seen_here: dict = {}
                    for hit in hits:
                        text = hit.decode("utf-8", "replace")
                        label = _label_of(text)
                        form = _form(text, label)
                        _bump(counts, label, form)
                        if form == FORM_IDENTIFIER or label not in seen_here:
                            seen_here[label] = text
                    for label, text in seen_here.items():
                        place = _where(line, text)
                        if "answer" in place["where"]:
                            as_answer[label] = as_answer.get(label, 0) + 1
                        if "option_id" in place["where"]:
                            as_option[label] = as_option.get(label, 0) + 1
                        if len(examples) < MAX_HIT_EXAMPLES:
                            examples.append(place)
            for hit in rx.findall(tail.lower()):
                text = hit.decode("utf-8", "replace")
                _bump(counts, _label_of(text), _form(text, _label_of(text)))
    ident = sorted(k for k, f in counts.items()
                   if f.get(FORM_IDENTIFIER))
    return {
        "paths": [os.path.relpath(p, ROOT) for p in paths],
        "bytes": size,
        "seconds": round(time.perf_counter() - started, 2),
        "labels_found": len(counts),
        "labels_found_as_identifier": ident,
        "occurrences": {k: dict(sorted(v.items()))
                        for k, v in sorted(counts.items())},
        "as_gold_answer": dict(sorted(as_answer.items())),
        "as_option_id": dict(sorted(as_option.items())),
        "counts_mean": {
            "occurrences": "hits per label per form",
            "as_gold_answer": "ROWS of this source in which the label is "
                              "the gold answer — the finding that makes it a "
                              "label this run could have trained on",
            "as_option_id": "rows in which the label appears in the option "
                            "space, gold or not",
        },
        "examples": examples,
    }


def _contamination_reading(labels: list, per_source: dict, found: dict,
                           as_identifier: dict, as_answer: dict) -> str:
    """The scan as a sentence, with the three findings kept apart.

    A single "7 of 77 labels occur" would be read as contamination, and on
    this corpus all seven are the English phrase — `exchange rate` in a news
    headline, `age limit` in a comment. The identifier form and the gold
    answer are the two findings that would actually put a label in the
    gradient, so the sentence leads with them.
    """
    n = len(per_source)
    if not found:
        return (f"no banking77 label string occurs, in either form, in any "
                f"of the {n} other training sources")
    head = (f"no banking77 label occurs as an IDENTIFIER in the other {n} "
            "sources"
            if not as_identifier else
            f"{len(as_identifier)} of {len(labels)} labels occur as "
            f"IDENTIFIERS in the other sources ({sorted(as_identifier)}) — "
            "triage these by hand: an identifier is what another source's "
            "own option space would look like")
    gold = ("and none is a gold answer anywhere in them"
            if not as_answer else
            f"and {len(as_answer)} are GOLD ANSWERS there ({sorted(as_answer)}"
            "), which makes those labels trainable for this run and counts "
            "them as seen in `gold_exposure`")
    phrase = [k for k in found if k not in as_identifier]
    tail = ("" if not phrase else
            f". {len(phrase)} occur only as the English phrase "
            f"({sorted(phrase)[:6]}…), which is a collision of language and "
            "not contamination: the model was never shown them as labels")
    return f"{head}, {gold}{tail}"


def contamination_scan(run: dict, labels: list | None = None,
                       root: Path | None = None, write: bool = True,
                       log=print) -> dict:
    """Is any BANKING77 label string literally present in the other sources?

    `run.json` says the banking77 DATASET contributed zero rows
    (`fence.clean_1m`, `fenced_datasets`). That is a real and auditable
    claim and it is NOT the same claim as "no label of this space occurs in
    the training corpus" — a second source using `card_arrival` as its own
    intent id would put that label in the gradient with the fence intact.
    This scans for it instead of assuming either way, and the backbone's
    pretraining exposure is declared separately as unauditable, because it
    is.
    """
    from data import taxonomy as TX
    from data.optset import PREFETCH_DIR, source_paths

    root = str(root or PREFETCH_DIR)
    labels = sorted(labels or TX.load().keys())
    fence = run.get("fence") or {}
    sources = [d for d in (run.get("datasets") or []) if d != DATASET]
    manifest = run.get("data_manifest") or {}
    rx = label_regex(labels)
    per_source, missing = {}, []
    started = time.perf_counter()
    for dataset in sources:
        # the trainer's own resolver: a multi-shard source (synth-v1 is six
        # files, prog-gold four) is not a single `<name>.jsonl`, and looking
        # for one would have quietly skipped two of the twelve sources
        paths = source_paths(dataset, root)
        if not paths:
            missing.append(dataset)
            continue
        rep = scan_files(paths, labels, rx)
        rep["sha256_declared_by_the_run"] = (manifest.get(dataset)
                                             or {}).get("sha256")
        per_source[dataset] = rep
        log(f"[parity] scanned {dataset}: {rep['labels_found']} label "
            f"strings ({len(rep['labels_found_as_identifier'])} as "
            f"identifiers) in {rep['bytes'] / 1e6:.0f} MB "
            f"({rep['seconds']}s)")
    found: dict = {}
    as_identifier: dict = {}
    as_answer: dict = {}
    for dataset, rep in per_source.items():
        for label, forms in rep["occurrences"].items():
            found.setdefault(label, {})[dataset] = forms
            if forms.get(FORM_IDENTIFIER):
                as_identifier.setdefault(label, {})[dataset] = \
                    forms[FORM_IDENTIFIER]
        for label, count in rep["as_gold_answer"].items():
            as_answer.setdefault(label, {})[dataset] = count
    doc = {
        "format": "jev.gate.v1",
        "task": TASK,
        "artifact": "contamination",
        "generated_utc": utcnow(),
        "question": "does any of the 77 BANKING77 label strings occur "
                    "literally in the other training sources",
        "run_id": run.get("run_id"),
        "declared_fence": {
            "clean_1m": fence.get("clean_1m"),
            "fenced_datasets": fence.get("fenced_datasets"),
            "why": fence.get("why"),
            "training_sources": run.get("datasets"),
            "evidence": f"artifacts/runs/{run.get('run_id')}/run.json",
            "proves": "the banking77 DATASET contributed zero rows to the "
                      "mixture",
            "does_not_prove": "that no banking77 LABEL STRING occurs in the "
                              "other sources — a different claim, and the "
                              "one this scan measures",
        },
        "scan": {
            "labels": len(labels),
            "forms": "counted APART, because they are different findings: "
                     "the identifier (`card_arrival`) is a label another "
                     "source could have trained on; the readable form "
                     "(`card arrival`) is an English phrase and its "
                     "appearance in somebody's sentence is a collision of "
                     "language, not contamination. Case-insensitive, with "
                     "boundaries so a label is not matched inside a longer "
                     "identifier",
            "sources_scanned": sorted(per_source),
            "shards_scanned": sum(len(r["paths"])
                                  for r in per_source.values()),
            "sources_missing_locally": missing,
            "resolver": "data.optset.source_paths — the trainer's own, so a "
                        "multi-shard source cannot be silently skipped",
            "scope": "the WHOLE prefetched file of each source, which is a "
                     "superset of the rows the mixture actually drew: a "
                     "scan that misses is then a stronger statement, and a "
                     "scan that hits is triaged by hand",
            "bytes": sum(r["bytes"] for r in per_source.values()),
            "seconds": round(time.perf_counter() - started, 2),
        },
        "labels_found": found,
        "labels_found_total": len(found),
        "labels_found_as_identifier": as_identifier,
        "labels_found_as_identifier_total": len(as_identifier),
        "labels_found_as_answer": as_answer,
        "labels_found_as_answer_total": len(as_answer),
        "per_source": per_source,
        "reading": _contamination_reading(labels, per_source, found,
                                          as_identifier, as_answer),
        "not_auditable": {
            "what": "the pretrained backbone's prior exposure to BANKING77",
            "why": "the backbone is a public checkpoint trained on a corpus "
                   "this repo does not hold and cannot inspect. BANKING77 "
                   "is a public benchmark, so exposure is plausible and "
                   "unmeasurable",
            "consequence": "no claim in this repo — including 'unseen "
                           "labels' — extends to the backbone's pretraining, "
                           "and it is stated separately so the corpus "
                           "fence's evidence is not read as covering it",
            "same_for_the_teacher": "and nobody can inspect the teacher's "
                                    "training data at all, which is why its "
                                    "0.924 is not a held-out claim",
        },
    }
    doc["artifact_path"] = os.path.relpath(CONTAMINATION_PATH, ROOT)
    if write:
        GATE_DIR.mkdir(parents=True, exist_ok=True)
        CONTAMINATION_PATH.write_text(
            json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return doc


def contamination_for(run: dict, refresh: bool = False, log=print) -> dict:
    """The scan, cached on the sources it was run over.

    2.4 GB of corpus per scan is not something a per-run report should pay
    every time, and it only changes when the mixture's sources or their
    bytes change — both of which the cached artifact records, so a stale
    cache is detectable rather than invisible.
    """
    cached = _read(CONTAMINATION_PATH)
    sources = sorted(d for d in (run.get("datasets") or []) if d != DATASET)
    manifest = run.get("data_manifest") or {}
    if cached and not refresh:
        same_sources = cached.get("scan", {}).get("sources_scanned") == sources
        same_bytes = all(
            (cached.get("per_source", {}).get(d) or {})
            .get("sha256_declared_by_the_run")
            == (manifest.get(d) or {}).get("sha256") for d in sources)
        if same_sources and same_bytes:
            cached["reused"] = {
                "from": os.path.relpath(CONTAMINATION_PATH, ROOT),
                "why": "the sources and their declared sha256 are the ones "
                       "this scan was run over; nothing to re-read",
                "generated_utc": cached.get("generated_utc"),
            }
            return cached
    return contamination_scan(run, log=log)


# ------------------------------------------------------------- the distances

def _overlap(a: list, b: list) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def distances(ours: dict) -> dict:
    """Distance to the baseline and to the teacher, in the same block.

    Both are subtractions a reader could do; they are published so that
    nobody has to go and find the other number, and so that a run cannot
    report movement against a baseline it chose after the fact.
    """
    acc = ours.get("accuracy") or 0.0
    ref = teacher()
    base_ci = BASELINE["accuracy_ci95"]
    ours_ci = ours.get("accuracy_ci95") or [0.0, 1.0]
    overlap = _overlap(ours_ci, base_ci)
    return {
        "to_baseline": {
            "baseline": BASELINE["accuracy"],
            "baseline_artifact": BASELINE["artifact"],
            "ours": acc,
            "delta": round(acc - BASELINE["accuracy"], 6),
            "intervals_overlap": overlap,
            "reading": (
                "the two intervals overlap, so this run has not been shown "
                "to differ from the first 77-way number this repo published"
                if overlap else
                "the intervals are disjoint, so the difference from the "
                "first published 77-way number is larger than sampling "
                "noise on these rows"),
        },
        "to_teacher": {
            "teacher": ref["accuracy"],
            "teacher_is_a_citation": True,
            "ours": acc,
            "delta": round(ref["accuracy"] - acc, 6),
            "ratio": round(acc / ref["accuracy"], 6) if ref["accuracy"] else None,
            "reading": "a distance to a CITED number: the protocol "
                       "differences in this artifact are part of it, and "
                       "the arms of #T-option-text say how much of it the "
                       "input information accounts for",
        },
        "chance": {
            "chance": CHANCE,
            "clears_it": ours.get("beats_chance"),
            "reading": "R2 — the distance to chance is the first distance, "
                       "because a number that does not clear it has no "
                       "distance to anything else worth reading",
        },
    }


def parity_line(doc: dict) -> str:
    """The one sentence a reader who stops at the bottom comes away with."""
    ours = doc["ours"]
    dist = doc["distances"]
    diffs = doc["protocol_differences"]
    unmatched = [d["id"] for d in diffs if d["matched"] is False]
    unverifiable = [d["id"] for d in diffs if d["matched"] == "unverifiable"]
    return (
        f"[{doc['cut']['name']}{' RESERVED' if doc['cut']['reserved'] else ''}] "
        f"{ours['hits']}/{ours['n']} = {ours['accuracy']} CI "
        f"{ours['accuracy_ci95']} at K={ours['cardinality']} against chance "
        f"{ours['chance']} — {'clears' if ours['beats_chance'] else 'does not clear'} "
        f"its chance rate; {dist['to_teacher']['delta']} from the teacher's "
        f"cited {dist['to_teacher']['teacher']} and "
        f"{dist['to_baseline']['delta']:+f} from the first published 77-way "
        f"number {dist['to_baseline']['baseline']}. The two protocols differ "
        f"on {len(unmatched)} declared dimensions ({', '.join(unmatched)}) "
        f"and {len(unverifiable)} cannot be checked from either side "
        f"({', '.join(unverifiable)}), so same_rows stays UNVERIFIED: the "
        "teacher has never been scored through this pipeline and publishes "
        "no per-row evidence to align against ours")


# --------------------------------------------------------------- composition

def compose(cut, ckpt_rel: str, manifest: dict, seal: dict, run: dict,
            samples: list, entries: list, contamination: dict,
            regime: dict, rows_path: Path | None, cost: dict,
            reason: str = "") -> dict:
    from eval import fullspace as F

    exposure = seen_labels(run, contamination,
                           [o["id"] for o in samples[0].options])
    seen = set(exposure["seen"])
    records = row_records(samples, entries, seen)
    ours = F.tally(samples, entries)
    doc = {
        "format": "jev.gate.v1",
        "task": TASK,
        "artifact": "parity",
        "generated_utc": utcnow(),
        "question": "what does this checkpoint score on the rows, the label "
                    "space and the split the teacher's 0.924 is quoted on, "
                    "and in what ways is the protocol still not the same",
        "checkpoint": ckpt_rel,
        "checkpoint_sha256": manifest.get("weights_sha256"),
        "model_version": manifest.get("model_version"),
        "run_id": manifest.get("run_id"),
        "tokenizer_hash": manifest.get("tokenizer_hash"),
        "cut": {
            "name": getattr(cut, "name", "banking77-test"),
            "reserved": bool(getattr(cut, "reserved", True)),
            "dataset": getattr(cut, "dataset", DATASET),
            "split": getattr(cut, "split", "test"),
            "rows_official": OFFICIAL_TEST_ROWS,
            "rows_sealed": seal.get("rows"),
            "rows_scored": ours["n"],
            "is_the_whole_official_test_set":
                ours["n"] == OFFICIAL_TEST_ROWS,
            "manifest": seal.get("manifest"),
            "manifest_sha256": seal.get("manifest_sha256"),
            "split_sha256": seal.get("split_sha256"),
            "rule": "R7 — the reserved cut is read rarely and every read is "
                    "logged with its date, its checkpoint and its reason",
        },
        "cardinality": CARDINALITY,
        "chance": CHANCE,
        "ours": ours,
        "baseline": BASELINE,
        "references": references(),
        "distances": distances(ours),
        "rows": row_identity(records, seal, rows_path),
        "gold_exposure": {
            **exposure,
            "by_side": exposure_split(samples, entries, seen),
        },
        "information_regime": regime,
        "contamination": {
            "artifact": os.path.relpath(CONTAMINATION_PATH, ROOT),
            "generated_utc": (contamination or {}).get("generated_utc"),
            "declared_fence": (contamination or {}).get("declared_fence"),
            "labels_found_total": (contamination or {}).get(
                "labels_found_total"),
            "labels_found": (contamination or {}).get("labels_found"),
            "labels_found_as_identifier": (contamination or {}).get(
                "labels_found_as_identifier"),
            "labels_found_as_answer": (contamination or {}).get(
                "labels_found_as_answer"),
            "reading": (contamination or {}).get("reading"),
            "not_auditable": (contamination or {}).get("not_auditable"),
        },
        "same_rows": False,
        "same_rows_reading": None,
        "protocol_differences": protocol_differences(regime, contamination),
        "cost": cost,
        "honesty": [
            "the teacher's 0.924 is a CITATION. It was not produced by this "
            "pipeline and there is no per-row evidence to align against "
            "ours, so this artifact publishes a distance to a quoted "
            "number, not a head-to-head result",
            "every difference of protocol we know of is enumerated above "
            "with `matched: true | false | unverifiable`; the third value "
            "is not a polite way of saying false",
            "R2 — no accuracy in this file appears without its cardinality, "
            "its chance rate and its 95 % interval",
            "R4 — a verdict measured in one training regime does not "
            "transfer to another; the baseline is a phase-1 checkpoint read "
            "under the phase-2 metric and is kept as a line, not a target",
        ],
    }
    doc["same_rows_reading"] = same_rows_reading(doc["rows"])
    if reason:
        doc["reason"] = reason
    doc["parity_line"] = parity_line(doc)
    doc["_records"] = records
    return doc


# ------------------------------------------------------------------ writing

def _slug(ckpt_rel: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", ckpt_rel).strip("_")


def write(doc: dict) -> dict:
    """Publish the artifact, the per-row evidence and the per-checkpoint copy.

    `parity.json` is the latest read, which is what a reader opens.
    `by-checkpoint/<slug>.json` is what `attach()` looks in, so two runs
    reporting at the same time do not overwrite each other's headline.
    """
    records = doc.pop("_records", [])
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    BY_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if records and doc["rows"].get("file"):
        path = ROOT / doc["rows"]["file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                for r in records), encoding="utf-8")
    doc["artifact_path"] = os.path.relpath(GATE_PATH, ROOT)
    doc["also_written_to"] = os.path.relpath(
        BY_CHECKPOINT_DIR / f"{_slug(doc['checkpoint'])}.json", ROOT)
    body = json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
    GATE_PATH.write_text(body)
    (ROOT / doc["also_written_to"]).write_text(body)
    doc["artifact"] = doc["artifact_path"]
    return doc


def published(ckpt_rel: str) -> dict:
    """The parity artifact for ONE checkpoint, or `{}` if there is none."""
    direct = BY_CHECKPOINT_DIR / f"{_slug(ckpt_rel)}.json"
    if direct.exists():
        return _read(direct)
    latest = _read(GATE_PATH)
    return latest if latest.get("checkpoint") == ckpt_rel else {}


def attach(ckpt_rel: str | None) -> dict:
    """The parity headline, for any report that names a checkpoint.

    This is done-when 6: a run's report does not have to remember to include
    the parity number, and it cannot include somebody else's — the block
    either carries the measurement made on THIS checkpoint or says, in the
    report, that none was made and how to make it.
    """
    doc = published(ckpt_rel) if ckpt_rel else {}
    if not doc:
        return {
            "measured": False,
            "task": f"#{TASK}",
            "why": ("no parity artifact for "
                    f"{ckpt_rel!r}: this checkpoint has never been scored "
                    "on the rows the teacher's number is quoted on"),
            "how": (".venv-train/bin/python -m eval.parity gate --checkpoint "
                    f"{ckpt_rel} --reason '<why the reserved cut is read>'"),
            "baseline": BASELINE,
            "teacher": teacher(),
        }
    return {
        "measured": True,
        "task": f"#{TASK}",
        "artifact": doc.get("artifact_path") or os.path.relpath(GATE_PATH,
                                                                ROOT),
        "generated_utc": doc.get("generated_utc"),
        "checkpoint": doc.get("checkpoint"),
        "cut": doc.get("cut"),
        "ours": doc.get("ours"),
        "baseline": doc.get("baseline"),
        "distances": doc.get("distances"),
        "same_rows": doc.get("same_rows"),
        "same_rows_reading": doc.get("same_rows_reading"),
        "protocol_differences": doc.get("protocol_differences"),
        "gold_exposure": {k: v for k, v in (doc.get("gold_exposure") or
                                            {}).items() if k != "basis"},
        "information_regime": {
            k: v for k, v in (doc.get("information_regime") or {}).items()
            if k in ("measured", "artifact", "arms", "budget_finding",
                     "share_of_gap_to_teacher_explained", "reading")},
        "line": doc.get("parity_line"),
    }


# ------------------------------------------------------------------ the gate

def measure(ckpt_dir: str, cut_key: str = "test", limit: int | None = None,
            device: str = "auto", batch_size: int = 16, log=print):
    """Score one checkpoint, by path, on one cut at the full cardinality."""
    from eval import calib as C
    from eval import cuts as K
    from training.python import train_decision as T

    cut = K.CUTS[cut_key]
    seal = K.sealed(cut)
    samples = K.samples(cut, limit)
    if not samples:
        raise ValueError(f"cut {cut.name!r} produced no samples")
    log(f"[parity] {cut.name}: {len(samples)} rows, "
        f"K={len(samples[0].options)}")
    engine, manifest = T.load_checkpoint(ckpt_dir, device)
    started = time.perf_counter()
    entries = C.entries_from_samples(engine, samples, cut.split, cut.name,
                                     cut.dataset, batch_size)
    elapsed = time.perf_counter() - started
    cost = {
        "device": str(engine.device),
        "rows": len(entries),
        "cardinality": len(samples[0].options),
        "seconds": round(elapsed, 2),
        "ms_per_row": round(1000 * elapsed / max(1, len(entries)), 2),
        "note": "the head attends BETWEEN options, so this grows about K² "
                "per row: a cost measured at K=8 does not price K=77",
    }
    return cut, seal, samples, entries, manifest, cost


def build(ckpt_rel: str, cut, seal: dict, samples: list, entries: list,
          manifest: dict, cost: dict, reason: str = "", refresh: bool = False,
          log=print) -> dict:
    """Compose the artifact off rows that are already scored."""
    run = _read(ROOT / "artifacts" / "runs" / str(manifest.get("run_id"))
                / "run.json")
    contamination = contamination_for(run, refresh=refresh, log=log)
    regime = information_regime()
    rows_path = ROWS_DIR / f"{_slug(ckpt_rel)}__{getattr(cut, 'name', 'cut')}.jsonl"
    return compose(cut, ckpt_rel, manifest, seal, run, samples, entries,
                   contamination, regime, rows_path, cost, reason)


def gate(ckpt_dir: str, cut_key: str = "test", limit: int | None = None,
         device: str = "auto", batch_size: int = 16, reason: str = "",
         refresh: bool = False, write_artifact: bool = True,
         log=print) -> dict:
    """Run the whole thing on any checkpoint by path. Logs the R7 read."""
    from eval import cuts as K

    cut, seal, samples, entries, manifest, cost = measure(
        ckpt_dir, cut_key, limit, device, batch_size, log)
    rel = os.path.relpath(ckpt_dir, ROOT)
    doc = build(rel, cut, seal, samples, entries, manifest, cost, reason,
                refresh, log)
    if write_artifact:
        doc = write(doc)
        if cut.reserved:
            K.record_query(rel, reason or
                           "eval.parity gate, no reason given on the command "
                           "line — the read happened anyway and is logged as "
                           "unexplained",
                           os.path.relpath(GATE_PATH, ROOT),
                           rows=len(entries), by="eval.parity")
            doc["test_cut_query_logged"] = os.path.relpath(K.LEDGER_PATH,
                                                           ROOT)
    return doc


def emit(ckpt_rel: str, cut, seal: dict, samples: list, entries: list,
         manifest: dict, cost: dict, reason: str = "", log=print) -> dict:
    """The hook `eval.fullspace` calls: same rows, same forward pass.

    Done-when 6 says the parity gate enters the report of every run without
    manual intervention. The cheapest honest way to do that is to compose it
    from rows a run has ALREADY scored: no second model load, no second read
    of the reserved cut, and therefore no second entry in the R7 ledger for
    a number that came out of the first one.
    """
    doc = build(ckpt_rel, cut, seal, samples, entries, manifest, cost,
                reason, log=log)
    doc["emitted_by"] = {
        "who": "eval.fullspace",
        "how": "composed from the rows that run had already scored — no "
               "extra forward pass and no extra read of the reserved cut",
    }
    return write(doc)


# ----------------------------------------------------------------- rendering

def render(doc: dict) -> str:
    out = []
    add = out.append
    cut = doc["cut"]
    ours = doc["ours"]
    add(f"PARITY · {doc['checkpoint']}")
    add(f"  cut {cut['name']}{' [RESERVED]' if cut['reserved'] else ''} · "
        f"{cut['rows_scored']}/{cut['rows_official']} official rows · "
        f"K={doc['cardinality']} · chance {doc['chance']} · model_version "
        f"{doc['model_version']}")
    add("")
    add(f"  {'who':<34}{'hits':>10}{'acc':>10}{'CI 95 %':>22}{'chance':>9}")
    ci = "[%.4f, %.4f]" % tuple(ours["accuracy_ci95"])
    add(f"  {'ours (this checkpoint)':<34}"
        f"{str(ours['hits']) + '/' + str(ours['n']):>10}"
        f"{ours['accuracy']:>10.4f}{ci:>22}{doc['chance']:>9.4f}")
    b = doc["baseline"]
    bci = "[%.4f, %.4f]" % tuple(b["accuracy_ci95"])
    add(f"  {'baseline (first 77-way, cited)':<34}"
        f"{str(b['hits']) + '/' + str(b['n']):>10}{b['accuracy']:>10.4f}"
        f"{bci:>22}{b['chance']:>9.4f}")
    for ref in doc["references"]:
        who = (ref["who"] + " [citation]")[:33]
        add(f"  {who:<34}{'—':>10}"
            f"{ref['accuracy']:>10.4f}{'—':>22}{doc['chance']:>9.4f}")
    if not cut["is_the_whole_official_test_set"]:
        add(f"  ! {cut['rows_scored']} of the {cut['rows_official']} "
            "official rows: this is NOT the parity row-set and the number "
            "above is not comparable with the teacher's")
    add("")
    add("PROTOCOL DIFFERENCES")
    for diff in doc["protocol_differences"]:
        mark = {True: "same", False: "DIFFERS"}.get(diff["matched"],
                                                    "UNVERIFIABLE")
        add(f"  {mark:<13}{diff['id']:<22}{diff['dimension']}")
    add("")
    add("INFORMATION REGIME (#T-option-text)")
    regime = doc["information_regime"]
    for cut_name, arms in ((regime.get("arithmetic") or {})
                           .get("comparable_within") or {}).items():
        for key in arms:
            arm = (regime.get("arms_by_cut") or {}).get(cut_name, {}).get(key)
            if not arm:
                continue
            ci = "[%.4f, %.4f]" % tuple(arm["accuracy_ci95"])
            add(f"  {key:<17}{str(arm['hits']) + '/' + str(arm['n']):>10}"
                f"{arm['accuracy']:>10.4f}{ci:>22}  {cut_name}"
                f"{' [RESERVED]' if arm['reserved_cut'] else ''}")
    for name, delta in ((regime.get("arithmetic") or {})
                        .get("deltas") or {}).items():
        add(f"  Δ {name:<38}{delta['point_shift']:+.6f}  intervals "
            f"{'overlap' if delta['intervals_overlap'] else 'DISJOINT'}")
    add("")
    exposure = doc["gold_exposure"]
    for side in ("seen_gold", "unseen_gold"):
        rep = exposure["by_side"][side]
        add(f"  gold {side:<12} n={rep.get('n')} "
            f"acc={rep.get('accuracy')} CI={rep.get('accuracy_ci95')} "
            f"chance={rep.get('chance')}")
    add(f"  labels seen in training: {exposure['n_seen']}/"
        f"{exposure['n_seen'] + exposure['n_unseen']}")
    add("")
    add(f"ROWS  {doc['rows']['n']} ids, sha256 "
        f"{doc['rows']['ids_sha256'][:16]}… · {doc['rows']['file']}")
    add(f"CONTAMINATION  {doc['contamination']['reading']}")
    add("")
    add("SAME ROWS: false — " + doc["same_rows_reading"]["value_means"])
    for line in doc["same_rows_reading"]["what_is_NOT_verified"]:
        add(f"  not verified: {line}")
    add("")
    add("LINE")
    add("  " + doc["parity_line"])
    return "\n".join(out)


# ---------------------------------------------------------------------- CLI

def main(argv: list) -> int:
    ap = argparse.ArgumentParser(prog="eval.parity")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("gate", help="score one checkpoint against the "
                                    "published recipe")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cut", default="test", choices=("dev", "test"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--refresh-contamination", action="store_true")
    p.add_argument("--no-write", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--reason", default="",
                   help="why the reserved cut is being read (R7); it goes "
                        "into artifacts/gates/T-eval-cardinality/"
                        "test-queries.json")
    c = sub.add_parser("contamination",
                       help="scan the other training sources for the 77 "
                            "label strings (stdlib only)")
    c.add_argument("--run", default="",
                   help="run id whose run.json declares the sources; "
                        "defaults to the checkpoint of the published "
                        "parity artifact")
    c.add_argument("--no-write", action="store_true")
    sub.add_parser("show", help="the published parity artifact")
    args = ap.parse_args(argv)

    if args.cmd == "show":
        doc = _read(GATE_PATH)
        if not doc:
            print(f"no artifact at {os.path.relpath(GATE_PATH, ROOT)}")
            return 1
        print(render(doc))
        return 0

    if args.cmd == "contamination":
        run_id = args.run or _read(GATE_PATH).get("run_id")
        run = _read(ROOT / "artifacts" / "runs" / str(run_id) / "run.json")
        if not run:
            print(f"no run.json for run id {run_id!r}: name one with --run")
            return 1
        doc = contamination_scan(run, write=not args.no_write)
        print(json.dumps({k: v for k, v in doc.items() if k != "per_source"},
                         indent=2, ensure_ascii=False))
        return 0

    if args.cut == "test" and not args.reason.strip():
        ap.error("--cut test needs --reason: every read of the reserved cut "
                 "is logged with why it was made (R7)")
    doc = gate(args.checkpoint, args.cut, args.limit or None, args.device,
               args.batch_size, args.reason, args.refresh_contamination,
               write_artifact=not args.no_write)
    print(json.dumps(doc, indent=2, ensure_ascii=False) if args.json
          else render(doc))
    if not args.no_write:
        print(f"\nartifact: {doc.get('artifact')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
