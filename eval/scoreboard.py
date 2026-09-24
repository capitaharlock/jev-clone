"""ONE command, ONE checkpoint, ONE table — #T-eval-cardinality.

Before this module the state of a checkpoint lived in three artifacts that
disagreed and nobody reconciled: the trainer's stage eval (n=3 008, K the
sampler's 3-8), the `eval.unseen` gate (n=5 624, same K regime, banking77
included), and `eval.fullspace` (n=3 080, K=77). Three cuts, three numbers,
no statement of which one answers which question — so whichever one was
quoted became the headline by accident.

The hierarchy this publishes, and it is not negotiable inside an artifact:

* **primary** — accuracy at the FULL cardinality of the label space on
  labels the run never trained on, with its chance rate, its counts and its
  95 % interval. This is the number the product promises and the only one
  that may head a report (rules R1/R2).
* **diagnostics** — everything measured at K≤8, the K sweep, the ranking
  with `unknown` taken out of the race. Useful, published, labelled, and
  never at the top.
* **parity** — what an outside number was measured on, and every way our
  protocol differs from it (R3/R8). A citation is not a measurement.
* **cost** — what the number cost to produce, on which device.
* **regime** — the cardinality the checkpoint was TRAINED at (R4). A verdict
  measured in one regime does not transfer to another, so the regime travels
  with every number.

And three diagnostics that tell a collapsed head apart from a head making
semantic mistakes, which no artifact in this repo published before: the
prediction histogram per label, the confusion matrix, and a sample of real
errors.

The cut is the **development** cut by default (`eval.cuts`, rule R7).
Reading the reserved test cut takes `--cut test --reason "…"` and writes an
entry in `artifacts/gates/T-eval-cardinality/test-queries.json`.

CLI:
    CKPT=artifacts/checkpoints/decision/leverstack-d512-prior-ettin-68m-s20260922/stage-001000000
    .venv-train/bin/python -m eval.scoreboard show --checkpoint $CKPT
    .venv-train/bin/python -m eval.scoreboard show --checkpoint $CKPT \\
        --cut test --reason "final read for the #T-option-text writeup"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import cuts as CUTS  # noqa: E402
from eval import fullspace as F  # noqa: E402

TASK = "T-eval-cardinality"
GATE_DIR = ROOT / "artifacts" / "gates" / TASK
GATE_PATH = GATE_DIR / "scoreboard.json"

#: the artifacts this scoreboard reconciles instead of leaving to the reader
UNSEEN_GATE = ROOT / "artifacts" / "gates" / "T-unseen-labels" / "gate.json"
FULLSPACE_GATE = (ROOT / "artifacts" / "gates" / "T-teacher-probe"
                  / "fullspace.json")
BENCH = ROOT / "artifacts" / "gates" / "T-state-cache" / "bench.json"

ERRORS_SAMPLED = 12
TOP_LABELS = 12
TOP_CONFUSIONS = 12


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


# ----------------------------------------------------------- the diagnostics

def prediction_frequency(samples: list, entries: list) -> dict:
    """How often each label is PREDICTED against how often it is the gold.

    A head that has collapsed onto one label and a head that is making
    semantic mistakes both score at chance; only this tells them apart.
    """
    predicted: Counter = Counter()
    gold: Counter = Counter()
    for sample, entry in zip(samples, entries):
        k = len(sample.options)
        pred = entry["pred"]
        predicted["⟨unknown⟩" if pred >= k
                  else sample.options[pred]["id"]] += 1
        gold[sample.options[sample.gold_index]["id"]] += 1
    n = max(1, len(entries))
    top = predicted.most_common(TOP_LABELS)
    return {
        "n": len(entries),
        "labels_in_space": len(samples[0].options) if samples else 0,
        "distinct_predicted": len(predicted),
        "distinct_gold": len(gold),
        "top1_label": top[0][0] if top else None,
        "top1_share": round(top[0][1] / n, 6) if top else None,
        "top5_share": round(
            sum(c for _, c in predicted.most_common(5)) / n, 6),
        "reading": _collapse_reading(predicted, gold, n,
                                     len(samples[0].options) if samples
                                     else 0),
        "predicted": [{"label": lab, "count": c, "share": round(c / n, 6),
                       "gold_count": gold.get(lab, 0)} for lab, c in top],
        "never_predicted": sorted(set(gold) - set(predicted))[:TOP_LABELS],
        "n_never_predicted": len(set(gold) - set(predicted)),
    }


def _collapse_reading(predicted: Counter, gold: Counter, n: int,
                      space: int) -> str:
    """Three regimes, read off coverage and the top label's share.

    All three can sit at the same accuracy, and they are different problems:
    a constant answer, a strong prior over a fraction of the space, and
    errors that are actually about meaning. The reading says which one this
    is and explicitly does NOT say anything about accuracy — that is the
    primary's job and it is one line above.
    """
    if not predicted or not space:
        return "nothing was scored"
    label, count = predicted.most_common(1)[0]
    share = count / n
    uniform = 1.0 / space
    coverage = len(predicted) / space
    if share >= 0.5:
        return (f"collapsed: {share:.1%} of the rows are answered "
                f"{label!r}, which is the gold on {gold.get(label, 0)} of "
                f"{n}. The head is not ranking the space, it is picking a "
                "constant")
    if coverage <= 0.5 or share >= 5 * uniform:
        return (f"skewed: {len(predicted)} of {space} labels are ever "
                f"predicted and {label!r} takes {share:.1%} of the rows "
                f"where uniform would be {uniform:.1%} — a strong prior over "
                "part of the space, not a collapse onto one label and not "
                "an even spread. It says nothing about accuracy")
    return (f"spread: {len(predicted)} of {space} labels predicted, top "
            f"label {share:.1%} against a uniform {uniform:.1%} — the "
            "errors are distributed, which is what a semantic mistake looks "
            "like and NOT what a collapse looks like. It says nothing about "
            "accuracy")


def confusion(samples: list, entries: list) -> dict:
    """gold -> predicted -> count, sparse, plus the pairs that dominate."""
    matrix: dict = {}
    for sample, entry in zip(samples, entries):
        k = len(sample.options)
        pred = entry["pred"]
        g = sample.options[sample.gold_index]["id"]
        p = "⟨unknown⟩" if pred >= k else sample.options[pred]["id"]
        matrix.setdefault(g, {})
        matrix[g][p] = matrix[g].get(p, 0) + 1
    pairs = sorted(((c, g, p) for g, row in matrix.items()
                    for p, c in row.items() if g != p), reverse=True)
    return {
        "format": "gold -> predicted -> count, zero cells omitted",
        "cells": {g: dict(sorted(row.items(), key=lambda kv: -kv[1]))
                  for g, row in sorted(matrix.items())},
        "top_confusions": [{"gold": g, "predicted": p, "count": c}
                           for c, g, p in pairs[:TOP_CONFUSIONS]],
        "n_off_diagonal": sum(c for c, _, _ in pairs),
    }


def error_sample(samples: list, entries: list, n: int = ERRORS_SAMPLED,
                 state_chars: int = 180) -> list:
    """Real rows the model got wrong, with what it preferred and how surely.

    Evenly spaced through the cut, not the first N: the first N are one
    corner of one file.
    """
    wrong = [(s, e) for s, e in zip(samples, entries)
             if e["pred"] != s.gold_index]
    if not wrong:
        return []
    step = max(1, len(wrong) // n)
    out = []
    for sample, entry in wrong[::step][:n]:
        k = len(sample.options)
        pred = entry["pred"]
        probs = entry["probs"]
        out.append({
            "id": entry.get("id"),
            "state": sample.state[:state_chars],
            "gold": sample.options[sample.gold_index]["id"],
            "p_gold": round(probs[sample.gold_index], 6),
            "predicted": ("⟨unknown⟩" if pred >= k
                          else sample.options[pred]["id"]),
            "p_predicted": round(probs[pred], 6),
            "gold_rank": 1 + sum(1 for j in range(k)
                                 if probs[j] > probs[sample.gold_index]),
        })
    return out


# ------------------------------------------------------------ the protocol

def examples_protocol(samples: list) -> dict:
    """R8 — what information the model was actually given, in the artifact.

    Two numbers measured on the same rows at the same K are still not
    comparable if one of the models was handed category definitions and 24
    labelled examples and the other was handed identifiers.
    """
    example = samples[0] if samples else None
    texts = [o["text"] for o in (example.options if example else [])][:3]
    identifiers = all(o["id"] == o["text"]
                      for s in samples[:50] for o in s.options)
    return {
        "labelled_examples_in_state": 0,
        "option_text": ("raw label identifiers (`data/adapters.py:146` "
                        "builds `Option(id=l, text=l)`)" if identifiers
                        else "label text distinct from the identifier"),
        "option_text_examples": texts,
        "state": ("the row's own text only — no category definitions, no "
                  "retrieved neighbours, no demonstrations"),
        "state_truncation_tokens": 256,
        "why_it_is_here": "R8 — the information in the options is part of "
                          "the protocol, and it is the one documented "
                          "difference with the teacher that nobody has "
                          "measured yet (#T-option-text)",
    }


def training_regime(manifest: dict, run: dict) -> dict:
    """R4 — the cardinality the checkpoint was trained at.

    A checkpoint that does not declare it gets `declared: false` and no
    guess: reading the sampler defaults of whatever code is on disk today
    and calling it the run's regime is exactly the inference R4 exists to
    forbid.
    """
    declared = manifest.get("cardinality_regime") or run.get(
        "cardinality_regime")
    if declared:
        return {"declared": True, **declared}
    return {
        "declared": False,
        "why": "this checkpoint predates R4: its manifest does not say what "
               "cardinality it was trained at, and it is not inferred here",
        "how_to_fix": "every run from #T-eval-cardinality onward writes "
                      "`cardinality_regime` into run.json and into every "
                      "checkpoint manifest",
        "loss": manifest.get("loss"),
        "backbone_frozen": (manifest.get("backbone") or {}).get("frozen"),
    }


def label_exposure(manifest: dict, run: dict, dataset: str) -> dict:
    """Seen or unseen — as precisely as the evidence actually supports."""
    fence = run.get("fence") or {}
    fenced = fence.get("fenced_datasets") or []
    datasets = manifest.get("datasets") or run.get("datasets") or []
    return {
        "claim": "unseen" if (dataset in fenced or dataset not in datasets)
                 else "seen",
        "evidence": (f"run.json declares "
                     f"fence.clean_1m={fence.get('clean_1m')} "
                     f"with fenced_datasets={fenced}; the run's 12 training "
                     f"sources are {datasets}"),
        "not_verified": [
            "literal absence of each label string from the other training "
            "sources — dataset exclusion is not string absence "
            "(#T-jev-parity)",
            "the pretrained backbone's prior exposure to this benchmark, "
            "which is not auditable at all",
        ],
    }


# ------------------------------------------------------------- the sections

def primary_section(cut, samples: list, entries: list, seal: dict,
                    manifest: dict, run: dict, ckpt_dir: str,
                    cost: dict) -> dict:
    rep = F.tally(samples, entries)
    return {
        "role": "PRIMARY — the number the product promises",
        "metric": "accuracy at the full cardinality of the label space, on "
                  "labels this run never trained on",
        "cut": cut.name,
        "reserved_cut": cut.reserved,
        "dataset": cut.dataset,
        "split": cut.split,
        "checkpoint": ckpt_dir,
        "checkpoint_sha256": manifest.get("weights_sha256"),
        "model_version": manifest.get("model_version"),
        "tokenizer_hash": manifest.get("tokenizer_hash"),
        "split_sha256": seal.get("split_sha256"),
        "manifest": seal.get("manifest"),
        "manifest_sha256": seal.get("manifest_sha256"),
        "data_sha256": ((run.get("data_manifest") or {})
                        .get(cut.dataset, {}) or {}).get("sha256"),
        **rep,
        "seen_or_unseen": label_exposure(manifest, run, cut.dataset),
        "examples_protocol": examples_protocol(samples),
        "training_regime": training_regime(manifest, run),
        "cost": cost,
    }


def diagnostics_section(manifest: dict) -> dict:
    """The K≤8 numbers, each with the n and the K that make it readable."""
    unseen = _read(UNSEEN_GATE)
    overall = ((unseen.get("table") or {}).get("ALL") or {})
    fullspace = _read(FULLSPACE_GATE)
    metrics = manifest.get("metrics") or {}
    out = {
        "role": "DIAGNOSTIC — none of these may head a report (R1/R2): "
                "their K is the sampler's 3-8, not the label space",
        "eval_unseen_gate": {
            "role": "diagnostic",
            "artifact": os.path.relpath(UNSEEN_GATE, ROOT),
            "answers": "does the head point at an option whose LABEL it "
                       "never trained on, in the K it was trained at",
            "checkpoint": unseen.get("checkpoint"),
            "n": unseen.get("n_scored"),
            "cuts": {name: {"n_seen": (cut.get("seen") or {}).get("n"),
                            "n_unseen": (cut.get("unseen") or {}).get("n")}
                     for name, cut in sorted(
                         (unseen.get("table") or {}).items())},
            "unseen": {k: (overall.get("unseen") or {}).get("raw", {}).get(k)
                       for k in ("n", "mean_k", "chance", "accuracy",
                                 "accuracy_ci95")},
            "not_comparable_with": "any number measured at K=77 — R8 and a "
                                   "different K make it a different task",
        },
        "trainer_stage_eval": {
            "role": "diagnostic",
            "artifact": "the `metrics` block of every checkpoint manifest",
            "answers": "is the run still learning, stage by stage",
            "seen": {k: (metrics.get("seen") or {}).get(k)
                     for k in ("n", "mean_k", "chance", "accuracy",
                               "accuracy_ci95", "beats_chance")},
            "unseen": {k: (metrics.get("unseen") or {}).get(k)
                       for k in ("n", "mean_k", "chance", "accuracy",
                                 "accuracy_ci95", "beats_chance")},
            "why_it_is_not_the_primary": "it shares its regime with the "
                                         "objective, which is what makes it "
                                         "useful during a run and blind to "
                                         "the transfer the product sells",
        },
        "cardinality_sweep": {
            "role": "diagnostic, exploratory",
            "artifact": os.path.relpath(FULLSPACE_GATE, ROOT),
            "checkpoint": fullspace.get("checkpoint"),
            "points": {k: {m: v.get(m) for m in
                           ("n", "cardinality", "hits", "accuracy",
                            "accuracy_ci95", "chance", "beats_chance",
                            "ranking_beats_chance")}
                       for k, v in sorted(
                           (fullspace.get("cardinality_sweep") or {}).items(),
                           key=lambda kv: int(kv[0]))},
            "reading": fullspace.get("cardinality_reading"),
            "no_progression": "the points are published with their counts "
                              "and intervals and no narrative: K=5 and K=20 "
                              "do not clear their chance rate while K=8 and "
                              "K=40 do, which is not a curve",
        },
    }
    return out


def parity_section(dataset: str, ckpt_rel: str | None = None) -> dict:
    """Somebody else's number, and every way our protocol differs from it.

    The enumeration is not written here any more: `eval.parity` owns it and
    this section ATTACHES what that gate published for this very checkpoint
    (done-when 6 of #T-jev-parity — the parity number enters every run's
    report without anybody remembering to put it there). When no parity read
    exists for this checkpoint the attached block says so and names the
    command, which is the one thing a report must never quietly replace with
    somebody else's number.
    """
    from eval import parity as P

    attached = P.attach(ckpt_rel)
    return {
        "role": "PARITY — somebody else's number, and every way the "
                "protocol differs from ours (R3/R8)",
        "references": F.REFERENCES.get(dataset, []),
        "same_rows": False,
        "same_rows_reading": F.SAME_ROWS_READING,
        "differences_declared": [
            f"{d['id']}: {d['dimension']} — teacher: {d['teacher']}; ours: "
            f"{d['ours']} (matched: {d['matched']})"
            for d in (attached.get("protocol_differences") or [])
        ] or [
            "the enumeration lives in #T-jev-parity and no parity read "
            "exists for this checkpoint yet; the list is not copied here "
            "so it cannot drift",
        ],
        "jev_parity": attached,
        "measured_by": {
            "#T-jev-parity": "turns the citation into a measurement with "
                             "the differences enumerated, the per-row "
                             "evidence for our side and the contamination "
                             "audit — attached above as `jev_parity`",
            "#T-teacher-kappa": "agreement with the teacher on the same "
                                "rows — needs a working API key",
            "#T-option-text": "the three option-text arms that price the "
                              "information difference — measured, and "
                              "carried inside the parity artifact",
        },
    }


def cost_section(elapsed: float, n: int, k: int, device: str) -> dict:
    bench = _read(BENCH)
    return {
        "role": "COST — what the number cost, so a regime can be priced "
                "before a run is committed to",
        "device": device,
        "rows": n,
        "cardinality": k,
        "seconds": round(elapsed, 2),
        "ms_per_row": round(1000 * elapsed / max(1, n), 2),
        "rows_per_s": round(n / max(1e-6, elapsed), 2),
        "note": "the head attends BETWEEN options (`model/decision_head.py"
                ":105-110`), so this grows about K² per row: a cost measured "
                "at K=8 does not price a run at K=77",
        "published_serving_bench": {
            "artifact": os.path.relpath(BENCH, ROOT),
            "warm_p50_ms": bench.get("warm_p50_ms"),
            "cold_ms": bench.get("cold_ms"),
        } if bench else None,
    }


# --------------------------------------------------------------- the runner

def build(ckpt_dir: str, cut_key: str = "dev", limit: int | None = None,
          device: str = "auto", batch_size: int = 16, reason: str = "",
          log=print) -> dict:
    """Score one checkpoint on one cut and compose the whole table."""
    from eval import calib as C
    from training.python import train_decision as T

    cut = CUTS.CUTS[cut_key]
    seal = CUTS.sealed(cut)
    samples = CUTS.samples(cut, limit)
    if not samples:
        raise ValueError(f"cut {cut.name!r} produced no samples")
    log(f"[scoreboard] {cut.name}: {len(samples)} rows, "
        f"K={len(samples[0].options)}")

    engine, manifest = T.load_checkpoint(ckpt_dir, device)
    run = _read(ROOT / "artifacts" / "runs"
                / str(manifest.get("run_id")) / "run.json")
    t0 = time.perf_counter()
    entries = C.entries_from_samples(engine, samples, cut.split, cut.name,
                                     cut.dataset, batch_size)
    elapsed = time.perf_counter() - t0
    cost = cost_section(elapsed, len(entries), len(samples[0].options),
                        str(engine.device))

    rel = os.path.relpath(ckpt_dir, ROOT)
    primary = primary_section(cut, samples, entries, seal, manifest, run,
                              rel, cost)
    board = {
        "format": "jev.scoreboard.v1",
        "task": TASK,
        "generated_utc": utcnow(),
        "checkpoint": rel,
        "model_version": manifest.get("model_version"),
        "run_id": manifest.get("run_id"),
        "hierarchy": ["primary", "diagnostics", "parity", "cost"],
        "rule": "no number heads a report without its chance rate, its K, "
                "its n and its 95 % interval (R2); a diagnostic never heads "
                "one at all (R1)",
        "primary": primary,
        "diagnostics": diagnostics_section(manifest),
        "parity": parity_section(cut.dataset, rel),
        "cost": cost,
        "training_regime": primary["training_regime"],
        "prediction_frequency": prediction_frequency(samples, entries),
        "confusion": confusion(samples, entries),
        "errors": error_sample(samples, entries),
        "cuts": CUTS.report(),
        "reconciliation": ".meshkore/docs/cortes-de-evaluacion.md",
    }
    if cut.reserved:
        CUTS.record_query(rel, reason, os.path.relpath(GATE_PATH, ROOT),
                          rows=len(entries), by="eval.scoreboard")
        board["test_cut_query_logged"] = os.path.relpath(
            CUTS.LEDGER_PATH, ROOT)
        board["cuts"] = CUTS.report()
    return board


# ----------------------------------------------------------------- the table

def render(board: dict) -> str:
    """The scoreboard as the operator reads it: the table, not the JSON."""
    p = board["primary"]
    d = board["diagnostics"]
    freq = board["prediction_frequency"]
    out = []
    add = out.append
    add(f"SCOREBOARD · {board['checkpoint']}")
    add(f"  model_version {board['model_version']} · "
        f"weights {str(p.get('checkpoint_sha256'))[:12]} · "
        f"generated {board['generated_utc']}")
    regime = board["training_regime"]
    add("  trained at cardinality: "
        + (json.dumps({k: v for k, v in regime.items() if k != "declared"},
                      ensure_ascii=False) if regime.get("declared")
           else "NOT DECLARED (checkpoint predates R4)"))
    add("")
    add("PRIMARY — full cardinality, "
        f"{p['seen_or_unseen']['claim']} labels")
    add(f"  {'cut':<22}{'K':>4}{'n':>7}{'hits':>6}{'acc':>10}"
        f"{'CI 95 %':>22}{'chance':>9}{'beats':>7}{'abst':>7}")
    lo, hi = p["accuracy_ci95"]
    ci = "[%.4f, %.4f]" % (lo, hi)
    name = p["cut"] + (" [RESERVED]" if p["reserved_cut"] else "")
    add(f"  {name:<22}{p['cardinality']:>4}{p['n']:>7}{p['hits']:>6}"
        f"{p['accuracy']:>10.4f}{ci:>22}{p['chance']:>9.4f}"
        f"{('yes' if p['beats_chance'] else 'no'):>7}"
        f"{p['abstain_rate']:>7.3f}")
    proto = p["examples_protocol"]
    add(f"  split_sha256 {str(p.get('split_sha256'))[:16]}… · "
        f"examples in STATE: {proto['labelled_examples_in_state']} · "
        f"option text: {proto['option_text'].split('(')[0].strip()}")
    add("")
    add("DIAGNOSTICS — never head a report")
    u = d["eval_unseen_gate"]["unseen"]
    add(f"  eval.unseen        n={u.get('n')} mean K={u.get('mean_k')} "
        f"acc={u.get('accuracy')} CI={u.get('accuracy_ci95')} "
        f"chance={u.get('chance')}")
    for side in ("seen", "unseen"):
        m = d["trainer_stage_eval"][side]
        add(f"  trainer {side:<7}    n={m.get('n')} mean K={m.get('mean_k')} "
            f"acc={m.get('accuracy')} CI={m.get('accuracy_ci95')} "
            f"chance={m.get('chance')}")
    for k, pt in d["cardinality_sweep"]["points"].items():
        add(f"  sweep K={k:<3}        n={pt.get('n')} hits={pt.get('hits')} "
            f"acc={pt.get('accuracy')} CI={pt.get('accuracy_ci95')} "
            f"chance={pt.get('chance')} "
            f"clears={'yes' if pt.get('beats_chance') else 'no'}")
    add(f"  reading: {d['cardinality_sweep']['reading']}")
    add("")
    add("PARITY — cited, not measured (same_rows: false, and that means "
        "NOT ESTABLISHED, not 'different rows')")
    for ref in board["parity"]["references"]:
        add(f"  {ref['who']:<34} {ref['accuracy']:.4f}  {ref['setup'][:70]}")
    jev = board["parity"]["jev_parity"]
    add("  #T-jev-parity: " + (jev.get("line") or jev.get("why") or ""))
    add("  differences: " + "; ".join(
        x.split(":")[0] for x in board["parity"]["differences_declared"]))
    add("")
    c = board["cost"]
    add(f"COST  {c['rows']} rows at K={c['cardinality']} on "
        f"{c['device']}: {c['seconds']}s, {c['ms_per_row']} ms/row "
        f"({c['rows_per_s']} rows/s)")
    add("")
    add(f"COLLAPSE  {freq['distinct_predicted']}/{freq['labels_in_space']} "
        f"labels ever predicted · top {freq['top1_label']!r} "
        f"{freq['top1_share']:.1%} · top5 {freq['top5_share']:.1%}")
    add(f"  {freq['reading']}")
    add("  most predicted: " + ", ".join(
        f"{r['label']}×{r['count']}" for r in freq["predicted"][:6]))
    add("CONFUSION  top pairs: " + ", ".join(
        f"{x['gold']}→{x['predicted']}×{x['count']}"
        for x in board["confusion"]["top_confusions"][:6]))
    add("ERRORS")
    for e in board["errors"][:5]:
        add(f"  gold {e['gold']} (p={e['p_gold']}, rank {e['gold_rank']}) → "
            f"{e['predicted']} (p={e['p_predicted']}) · {e['state'][:70]}")
    add("")
    add(f"cuts: dev={board['cuts']['dev'].get('rows')} rows, "
        f"test={board['cuts']['test'].get('rows')} rows RESERVED · "
        f"queries to the reserved cut: "
        f"{len(board['cuts']['queries_to_the_reserved_cut'])}")
    add(f"which cut answers what: {board['reconciliation']}")
    return "\n".join(out)


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(prog="eval.scoreboard")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("show", help="the whole table for one checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cut", default="dev", choices=sorted(CUTS.CUTS))
    p.add_argument("--reason", default="",
                   help="why the reserved cut is being read (required for "
                        "--cut test; it goes in the ledger)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", default="")
    p.add_argument("--no-write", action="store_true")
    p.add_argument("--json", action="store_true",
                   help="print the artifact instead of the table")
    args = ap.parse_args(argv)

    if args.cut == "test" and not args.reason.strip():
        ap.error("--cut test needs --reason: every read of the reserved cut "
                 "is logged with why it was made (R7)")

    board = build(args.checkpoint, args.cut, args.limit or None, args.device,
                  args.batch_size, args.reason)
    if not args.no_write:
        out = Path(args.out) if args.out else GATE_PATH
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(board, indent=2, ensure_ascii=False) + "\n")
        board["artifact"] = os.path.relpath(out, ROOT)
    print(json.dumps(board, indent=2, ensure_ascii=False) if args.json
          else render(board))
    if not args.no_write:
        print(f"\nartifact: {board['artifact']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
