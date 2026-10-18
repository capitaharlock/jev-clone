"""The distance to the teacher, measured on the same rows (#T-teacher-probe).

Every accuracy this repo publishes is scored against a threshold WE wrote.
`#T-release-gate` wants 0.50 unseen because a planning document said so, not
because anything external says that is the bar. This module replaces that
ruler with the only one that answers the operator's question — *is this a
joke, or is it close?* — by putting the teacher and our head on the SAME
rows, the same option sets, the same forced-choice format, and publishing
the subtraction.

What "same rows" means here, precisely
--------------------------------------
The cut is `eval.unseen`'s own `unseen` cut: the label holdout of the run
that produced the checkpoint, its seeded option sets, its group partition.
From it a **stratified subsample** is drawn (seeded, versioned to
`subsample.json`), because the full cut is ~5.6k rows and the teacher is
metered. Our head is then re-scored on that subsample — not read off the
full-cut gate — so the two numbers are the same denominator.

Forced choice, both sides
-------------------------
The teacher is asked for one of the K options; it has no `unknown` slot.
So the like-for-like comparison is our `accuracy_options_only` (argmax over
the K options, ignoring our `unknown` logit). The headline `accuracy` — which
counts our abstentions as wrong — is published beside it, labelled, because
dropping it would flatter us.

Cost
----
Nothing is spent without `--budget-usd`. `--dry-run` builds the subsample,
scores our side, prints the estimated spend and writes nothing to the API.
Answers are cached by `eval.teacher`, so `#T-teacher-kappa` re-reads this
run for free.

CLI:
    CKPT=artifacts/checkpoints/decision/lever-stack-d512-prior/stage-001000000
    python3 -m eval.teacher_probe plan --checkpoint $CKPT          # no spend
    .venv-train/bin/python -m eval.teacher_probe gate \\
        --checkpoint $CKPT --n 400 --budget-usd 0.50
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.optset import SamplerConfig  # noqa: E402
from eval import teacher as TE  # noqa: E402
from eval import unseen as U  # noqa: E402
from eval.calib import wilson_interval  # noqa: E402

TASK = "T-teacher-probe"
GATE_DIR = ROOT / "artifacts" / "gates" / TASK
GATE_PATH = GATE_DIR / "gate.json"
SUBSAMPLE_PATH = GATE_DIR / "subsample.json"
PREDS_PATH = GATE_DIR / "preds.json"

#: the subsample seed. Versioned here and echoed into every artifact: the
#: measurement has to be repeatable without spending the budget again.
SUBSAMPLE_SEED = 20260923
DEFAULT_N = 400


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# -- the cut --------------------------------------------------------------

def unseen_samples(cfg: SamplerConfig | None = None,
                   caps: dict | None = None) -> dict:
    """The `unseen` cut of every dataset that has one, keyed by dataset."""
    holdout = U.T.build_holdout()
    cfg = cfg or SamplerConfig()
    caps = {**U.MAX_SAMPLES, **(caps or {})}
    out = {}
    for dataset in U.DATASETS:
        limit = caps.get(dataset, U.DEFAULT_MAX_SAMPLES)
        samples = U.cut_samples(holdout, dataset, "unseen", "test", limit,
                                cfg)
        if samples:
            out[dataset] = samples
    return out


def stratify(by_dataset: dict, n: int, seed: int = SUBSAMPLE_SEED) -> list:
    """`n` rows split across cuts in proportion to each cut's size.

    Proportional, not equal: an equal split would let the smallest cut
    dominate the pooled number, and the pooled number is the one the
    operator reads first. Every cut still gets at least 25 rows, or all of
    them if it has fewer — a cut with 8 rows in it is noise, and the gate
    says so rather than hiding it.
    """
    total = sum(len(v) for v in by_dataset.values())
    if not total:
        return []
    picked = []
    for dataset in sorted(by_dataset):
        rows = list(by_dataset[dataset])
        want = max(25, round(n * len(rows) / total))
        want = min(want, len(rows))
        random.Random(f"{seed}\x00{dataset}").shuffle(rows)
        picked.extend((dataset, s) for s in rows[:want])
    random.Random(f"{seed}\x00order").shuffle(picked)
    return picked


def subsample_card(picked: list, n_asked: int) -> dict:
    """Everything needed to rebuild this exact subsample without the API."""
    per: dict = {}
    for dataset, _ in picked:
        per[dataset] = per.get(dataset, 0) + 1
    return {
        "seed": SUBSAMPLE_SEED,
        "n_asked": n_asked,
        "n": len(picked),
        "per_cut": per,
        "cut": "unseen (label holdout of the checkpoint's own run)",
        "rows": [{
            "cut": dataset,
            "id": f"{s.dataset}:{s.row_id}:{s.question_id}",
            "options": s.option_ids(),
            "gold": s.answer,
        } for dataset, s in picked],
    }


# -- scoring --------------------------------------------------------------

def score_ours(ckpt_dir: str, picked: list, device: str = "auto") -> dict:
    """Our head on the subsample. Requires the torch stack."""
    from eval import calib as C

    engine, manifest = U.T.load_checkpoint(ckpt_dir, device)
    out = {"model_version": manifest.get("model_version"),
           "run_id": manifest.get("run_id"), "rows": {}}
    by_cut: dict = {}
    for dataset, sample in picked:
        by_cut.setdefault(dataset, []).append(sample)
    for dataset, samples in by_cut.items():
        entries = C.entries_from_samples(engine, samples, "test", "unseen",
                                         dataset)
        for sample, entry in zip(samples, entries):
            k = len(sample.options)
            probs = entry["probs"]
            opts_only = max(range(k), key=lambda j: probs[j])
            out["rows"][entry["id"]] = {
                "cut": dataset,
                "pred": (sample.options[entry["pred"]]["id"]
                         if entry["pred"] < k else U.T.UNKNOWN_ID),
                "pred_options_only": sample.options[opts_only]["id"],
                "gold": sample.answer,
                "k": k,
            }
    return out


def score_teacher(picked: list, client: TE.TeacherClient,
                  log=print) -> dict:
    """The teacher on the same rows. Cache hits cost nothing."""
    from data.optset import question_text

    rows: dict = {}
    errors = 0
    for i, (dataset, sample) in enumerate(picked, 1):
        rid = f"{sample.dataset}:{sample.row_id}:{sample.question_id}"
        try:
            ans = client.decide(sample.state, question_text(
                {"text": sample.question, "id": sample.question_id}),
                sample.options, sample.question_id or "q")
        except TE.BudgetExceeded as exc:
            log(f"[teacher-probe] stopped at row {i}/{len(picked)}: {exc}")
            break
        except TE.TeacherError as exc:
            errors += 1
            log(f"[teacher-probe] row {i} failed: {exc}")
            if errors > max(5, len(picked) // 20):
                raise
            continue
        rows[rid] = {
            "cut": dataset,
            "pred": ans.get("choice"),
            "gold": sample.answer,
            "k": len(sample.options),
            "confidence": ans.get("confidence"),
            "off_menu": ans.get("off_menu", False),
            "cached": ans.get("cached", False),
        }
        if i % 25 == 0:
            log(f"[teacher-probe] {i}/{len(picked)} "
                f"${client.budget.usd:.4f}")
    return {"rows": rows, "errors": errors, "budget": client.budget.card()}


# -- reporting ------------------------------------------------------------

def accuracy(rows: list, field: str = "pred") -> dict:
    if not rows:
        return {"n": 0, "skipped": "no rows"}
    hits = sum(1 for r in rows if r.get(field) == r["gold"])
    lo, hi = wilson_interval(hits, len(rows))
    chance = sum(1.0 / max(1, r["k"]) for r in rows) / len(rows)
    return {
        "n": len(rows),
        "accuracy": round(hits / len(rows), 6),
        "accuracy_ci95": [round(lo, 6), round(hi, 6)],
        "chance": round(chance, 6),
        "beats_chance": bool(lo > chance),
    }


def distance_table(ours: dict, theirs: dict) -> dict:
    """Per-cut teacher − us, on the intersection of rows both answered."""
    shared = sorted(set(ours["rows"]) & set(theirs["rows"]))
    table: dict = {}
    cuts = sorted({ours["rows"][r]["cut"] for r in shared})
    for cut in cuts + ["ALL"]:
        ids = [r for r in shared
               if cut == "ALL" or ours["rows"][r]["cut"] == cut]
        us = [ours["rows"][r] for r in ids]
        them = [theirs["rows"][r] for r in ids]
        ours_head = accuracy(us, "pred")
        ours_forced = accuracy(us, "pred_options_only")
        teach = accuracy(them, "pred")
        if not ids:
            continue
        table[cut] = {
            "n": len(ids),
            "teacher": teach,
            "ours_forced_choice": ours_forced,
            "ours_with_unknown": ours_head,
            "distance": round(teach.get("accuracy", 0.0)
                              - ours_forced.get("accuracy", 0.0), 6),
            "note": ("`distance` is teacher − ours on the LIKE-FOR-LIKE "
                     "comparison: both forced to name one of the K options. "
                     "`ours_with_unknown` is the headline number, which "
                     "counts our abstentions as wrong"),
        }
    worst = max((c for c in table if c != "ALL"),
                key=lambda c: table[c]["distance"], default=None)
    if worst:
        table["ALL"]["worst_cut"] = {
            "cut": worst, "distance": table[worst]["distance"]}
    return table


def verdict(table: dict) -> str:
    all_row = table.get("ALL") or {}
    d = all_row.get("distance")
    t = (all_row.get("teacher") or {}).get("accuracy")
    o = (all_row.get("ours_forced_choice") or {}).get("accuracy")
    if d is None:
        return "no comparable rows: the probe did not measure anything"
    verdicts = [
        f"teacher {t:.3f} vs ours {o:.3f} on the same {all_row['n']} unseen "
        f"rows: {d:+.3f}",
    ]
    if not (all_row.get("ours_forced_choice") or {}).get("beats_chance"):
        verdicts.append("our side does not clear chance on this subsample")
    if d > 0.30:
        verdicts.append("a different league, not a gap to close by tuning")
    elif d > 0.15:
        verdicts.append("same league, clearly behind")
    elif d > 0.05:
        verdicts.append("close; the remaining gap is worth attacking")
    else:
        verdicts.append("within noise of the teacher on this cut")
    return " — ".join(verdicts)


def compose_gate(ckpt_dir: str, ours: dict, theirs: dict, sub: dict,
                 table: dict) -> dict:
    all_row = table.get("ALL") or {}
    return {
        "format": "jev.gate.v1",
        "task": TASK,
        "generated_utc": utcnow(),
        "model_version": ours.get("model_version"),
        "run_id": ours.get("run_id"),
        "checkpoint": ckpt_dir,
        "teacher": TE.config_card(),
        "subsample": {k: v for k, v in sub.items() if k != "rows"},
        "subsample_file": os.path.relpath(SUBSAMPLE_PATH, ROOT),
        "predictions_file": os.path.relpath(PREDS_PATH, ROOT),
        "cost": theirs.get("budget"),
        "errors": theirs.get("errors", 0),
        "table": table,
        "pass": bool(all_row.get("n")),
        "verdict": verdict(table),
        "honesty": [
            "the teacher's accuracy is not a threshold we chose; the "
            "release gate's 0.50 is",
            "both sides answered the SAME rows with the SAME option sets; "
            "our side was re-scored on the subsample, not read off the "
            "full-cut gate",
            "`pass` here means the measurement exists, not that the "
            "distance is acceptable — that call is the operator's",
        ],
    }


# -- entry points ---------------------------------------------------------

def plan(ckpt_dir: str, n: int, device: str, write: bool = True) -> dict:
    """Build + version the subsample and estimate the spend. No API calls."""
    picked = stratify(unseen_samples(), n)
    sub = subsample_card(picked, n)
    client = TE.TeacherClient(TE.Budget(max_calls=0, max_usd=0.0))
    est = sum(TE.estimate_tokens(
        client.build_payload(s.state, s.question, s.options))
        for _, s in picked)
    sub["estimated_input_tokens"] = est
    sub["estimated_usd"] = round(est * TE.input_price() / 1_000_000, 6)
    cached = sum(1 for _, s in picked if client._cached(
        client.cache_key(s.state, s.question, s.options)) is not None)
    sub["already_cached"] = cached
    if write:
        GATE_DIR.mkdir(parents=True, exist_ok=True)
        SUBSAMPLE_PATH.write_text(json.dumps(sub, indent=1,
                                             ensure_ascii=False) + "\n")
    return sub


def run(ckpt_dir: str, n: int = DEFAULT_N, device: str = "auto",
        budget_usd: float = 0.5, max_calls: int = 2000,
        write: bool = True, log=print) -> dict:
    picked = stratify(unseen_samples(), n)
    sub = subsample_card(picked, n)
    ours = score_ours(ckpt_dir, picked, device)
    client = TE.TeacherClient(TE.Budget(max_calls=max_calls,
                                        max_usd=budget_usd), log=log)
    theirs = score_teacher(picked, client, log=log)
    table = distance_table(ours, theirs)
    gate = compose_gate(os.path.relpath(ckpt_dir, ROOT), ours, theirs, sub,
                        table)
    if write:
        GATE_DIR.mkdir(parents=True, exist_ok=True)
        SUBSAMPLE_PATH.write_text(json.dumps(sub, indent=1,
                                             ensure_ascii=False) + "\n")
        PREDS_PATH.write_text(json.dumps(
            {"ours": ours["rows"], "teacher": theirs["rows"]},
            indent=1, ensure_ascii=False) + "\n")
        GATE_PATH.write_text(json.dumps(gate, indent=2,
                                        ensure_ascii=False) + "\n")
    return gate


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(prog="eval.teacher_probe")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "gate"):
        p = sub.add_parser(name)
        p.add_argument("--checkpoint", required=(name == "gate"), default="")
        p.add_argument("--n", type=int, default=DEFAULT_N)
        p.add_argument("--device", default="auto")
        p.add_argument("--budget-usd", type=float, default=0.5)
        p.add_argument("--max-calls", type=int, default=2000)
    args = ap.parse_args(argv)
    if args.cmd == "plan":
        card = plan(args.checkpoint, args.n, args.device)
        print(json.dumps(card, indent=2, ensure_ascii=False))
        return 0
    gate = run(args.checkpoint, args.n, args.device, args.budget_usd,
               args.max_calls)
    print(json.dumps(gate["table"].get("ALL", {}), indent=2))
    print(gate["verdict"])
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
