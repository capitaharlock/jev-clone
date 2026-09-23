"""Our model on the FULL label space, against published teacher numbers.

`eval.teacher_probe` is the right measurement — same rows, same options,
teacher and us. It needs a working API key. This module is the measurement
that needs nothing but our own checkpoint, and it answers the same question
from the other side: the public Jev results are reported on the *whole*
benchmark label space (all 77 BANKING77 intents), so scoring our head the
same way puts our number on the same axis as a number somebody else already
published.

What this is NOT
----------------
It is not a same-rows comparison. The teacher's number here is a citation,
not a measurement we made: different rows may have been sampled, the teacher
saw labelled examples per prediction, and BANKING77 is public so nothing
guarantees it was unseen for it. Every reference carries its source and its
caveat in the artifact, and `same_rows: false` is stamped on the whole file
so no reader can mistake it for the probe.

What it IS
----------
The only honest reply to "are we close?" that costs nothing: our accuracy on
the exact benchmark, the exact split and the exact cardinality the published
number was reported on. Our option sets everywhere else cap K at 8; here K
is the whole space, which is the regime the product actually promises.

CLI:
    CKPT=artifacts/checkpoints/decision/lever-stack-d512-prior/stage-001000000
    .venv-train/bin/python -m eval.fullspace gate --checkpoint $CKPT
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

from data.optset import (OptionSetSampler, Sample,  # noqa: E402
                          SamplerConfig, question_text)
from eval import unseen as U  # noqa: E402
from eval.calib import wilson_interval  # noqa: E402

TASK = "T-teacher-probe"
GATE_DIR = ROOT / "artifacts" / "gates" / TASK
GATE_PATH = GATE_DIR / "fullspace.json"

#: Published third-party numbers, each with the claim it actually makes.
#: A reference without a source and a caveat is a rumour; these are quoted
#: so a reader can go and check them instead of trusting this file.
REFERENCES = {
    "banking77": [
        {
            "who": "Jev (teacher)",
            "accuracy": 0.9240,
            "setup": "all 3 080 official test messages, 77 intents, "
                     "category definitions + 24 labelled examples per "
                     "prediction, no weight updates",
            "source": "github.com/simonmesmith/jev-banking77-experiment",
            "caveat": "BANKING77 is public; nobody can inspect Jev's "
                      "training data, so this is not a held-out claim",
        },
        {
            "who": "fine-tuned BERT (BANKING77 paper)",
            "accuracy": 0.9366,
            "setup": "supervised fine-tune on the 10 003 training messages",
            "source": "Casanueva et al. 2020, BANKING77",
            "caveat": "trained ON this label space; our head never was",
        },
    ],
}

SPLIT = {"banking77": "test"}
DEFAULT_ROWS = 3080
#: the control that tells "no signal" apart from "K it never saw in training"
K_SWEEP = (5, 8, 20, 40, 77)
K_SWEEP_ROWS = 1000
K_SWEEP_SEED = 20260923


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def full_samples(dataset: str, limit: int = DEFAULT_ROWS,
                 cfg: SamplerConfig | None = None) -> list:
    """One `Sample` per row with EVERY label of the space as an option.

    The option order is the sorted label space, identical on every row, so
    the gold's position carries no signal: a head that learned "the answer
    is last" — which our sampler's layout could teach — scores at chance
    here, and that is the point.
    """
    from dataclasses import asdict
    cfg = U.eval_config(cfg)
    cfg = SamplerConfig(**{**asdict(cfg), "split": SPLIT.get(dataset, "test")})
    # deliberately NOT `make_sampler`: that one `restrict`s the pool to a
    # label subset, and the whole point here is that nothing is restricted.
    sampler = OptionSetSampler(datasets=(dataset,), config=cfg,
                               rows_per_dataset=U.ROWS_CAP,
                               root=U.T.PREFETCH_DIR)
    out: list = []
    for n, row in enumerate(sampler.rows[dataset]):
        for question in row.get("questions", []):
            gold = question.get("answer")
            pool, index = sampler.space_of(dataset, question)
            ids = sorted(o["id"] for o in pool)
            if gold not in ids or len(ids) < 2:
                continue
            options = [{"id": i, "text": index.text.get(i, i)} for i in ids]
            out.append(Sample(
                dataset=dataset, row_id=f"{dataset}-{n}",
                question_id=question.get("id", ""),
                state=row["state"], question=question_text(question),
                options=options, answer=gold,
                gold_index=ids.index(gold)))
            if len(out) >= limit:
                return out
    return out


def score(engine, samples: list, batch_size: int = 16) -> dict:
    from eval import calib as C

    entries = C.entries_from_samples(engine, samples, "test", "full-space",
                                     samples[0].dataset if samples else "",
                                     batch_size)
    n = len(entries)
    hits = forced = 0
    abstain = 0
    for sample, entry in zip(samples, entries):
        k = len(sample.options)
        probs = entry["probs"]
        best = max(range(k), key=lambda j: probs[j])
        if entry["pred"] >= k:
            abstain += 1
        elif entry["pred"] == sample.gold_index:
            hits += 1
        if best == sample.gold_index:
            forced += 1
    lo, hi = wilson_interval(hits, n) if n else (0.0, 1.0)
    flo, fhi = wilson_interval(forced, n) if n else (0.0, 1.0)
    k = len(samples[0].options) if samples else 0
    return {
        "n": n,
        "cardinality": k,
        "chance": round(1.0 / k, 6) if k else None,
        "accuracy": round(hits / n, 6) if n else None,
        "accuracy_ci95": [round(lo, 6), round(hi, 6)],
        "accuracy_options_only": round(forced / n, 6) if n else None,
        "accuracy_options_only_ci95": [round(flo, 6), round(fhi, 6)],
        "abstain_rate": round(abstain / n, 6) if n else None,
        "beats_chance": bool(n and flo > 1.0 / k),
    }


def resize(samples: list, k: int, seed: int = K_SWEEP_SEED) -> list:
    """The same rows with `k` options: gold kept, distractors drawn down."""
    out = []
    for i, s in enumerate(samples):
        rng = random.Random(f"{seed}\x00{k}\x00{i}")
        gold = s.options[s.gold_index]
        others = [o for o in s.options if o["id"] != gold["id"]]
        opts = rng.sample(others, min(k - 1, len(others))) + [gold]
        rng.shuffle(opts)
        out.append(Sample(
            dataset=s.dataset, row_id=s.row_id, question_id=s.question_id,
            state=s.state, question=s.question, options=opts,
            answer=gold["id"],
            gold_index=[o["id"] for o in opts].index(gold["id"])))
    return out


def cardinality_sweep(engine, samples: list, ks: tuple = K_SWEEP,
                      rows: int = K_SWEEP_ROWS, log=print) -> dict:
    """Accuracy as a function of K, on ONE fixed set of rows.

    Without this control a bad full-space number has two readings and no way
    to choose between them: a head with no real signal, or a head whose
    signal is fine but which never saw a K this large in training (our
    sampler caps at 8). A lift over chance that holds its RATIO as K grows
    is the second; a lift that decays to 1.0 is the first.
    """
    base = samples[:rows]
    out = {}
    for k in ks:
        rep = score(engine, resize(base, k))
        rep["lift_over_chance"] = (round(rep["accuracy"] / rep["chance"], 4)
                                   if rep.get("accuracy") and rep["chance"]
                                   else None)
        out[str(k)] = rep
        log(f"[fullspace] K={k}: acc {rep['accuracy']} "
            f"(chance {rep['chance']}, lift {rep['lift_over_chance']})")
    return out


def sweep_reading(sweep: dict) -> str:
    """Read the sweep off the CIs, not off the point estimates.

    The lift ratio is noisy at n=1 000 (one extra hit at K=40 moves it by
    0.04), so the statement this returns is built from `beats_chance` —
    whether the 95 % lower bound still clears 1/K — which is the only claim
    the sample size supports.
    """
    ks = sorted(int(k) for k in sweep)
    if not ks:
        return "not enough points to read"
    clears = [k for k in ks if sweep[str(k)].get("beats_chance")]
    lifts = {k: sweep[str(k)].get("lift_over_chance") for k in ks}
    if not clears:
        return ("the head does not clear chance at ANY cardinality on this "
                "cut: there is no signal to transfer")
    top = max(clears)
    if top == max(ks):
        return (f"the head still clears chance at K={top} "
                f"(lift {lifts[top]}x): the full-space number is a "
                "cardinality-transfer problem, not an absence of signal")
    return (f"the head clears chance up to K={top} (lift {lifts[top]}x) and "
            f"stops clearing it by K={min(k for k in ks if k > top)}: what "
            "it has at small K is a weak preference, not a ranking of the "
            "real label space, and it dissolves as options are added")


def compose_gate(ckpt_dir: str, manifest: dict, results: dict,
                 sweep: dict | None = None) -> dict:
    table = {}
    for dataset, ours in results.items():
        refs = REFERENCES.get(dataset, [])
        table[dataset] = {
            "ours": ours,
            "references": refs,
            "distance_to_teacher": (
                round(refs[0]["accuracy"] - (ours.get("accuracy") or 0.0), 6)
                if refs else None),
            "ratio_to_teacher": (
                round((ours.get("accuracy") or 0.0) / refs[0]["accuracy"], 4)
                if refs else None),
        }
    return {
        "format": "jev.gate.v1",
        "task": TASK,
        "artifact": "fullspace",
        "generated_utc": utcnow(),
        "model_version": manifest.get("model_version"),
        "run_id": manifest.get("run_id"),
        "checkpoint": ckpt_dir,
        "same_rows": False,
        "table": table,
        "cardinality_sweep": sweep or {},
        "cardinality_reading": sweep_reading(sweep) if sweep else None,
        "honesty": [
            "the teacher's number here is CITED, not measured by us: "
            "different rows, and it was given labelled examples per "
            "prediction while our head gets none",
            "the same-rows measurement is #T-teacher-probe and it needs a "
            "working API key; this artifact does not replace it",
            "K is the whole label space, not our sampler's 3-8: this is the "
            "regime the product promises and the hardest one we publish",
        ],
    }


def run(ckpt_dir: str, datasets: tuple = ("banking77",),
        limit: int = DEFAULT_ROWS, device: str = "auto",
        write: bool = True, sweep: bool = True, log=print) -> dict:
    engine, manifest = U.T.load_checkpoint(ckpt_dir, device)
    results = {}
    first = None
    for dataset in datasets:
        samples = full_samples(dataset, limit)
        first = first or samples
        log(f"[fullspace] {dataset}: {len(samples)} rows, "
            f"K={len(samples[0].options) if samples else 0}")
        results[dataset] = score(engine, samples)
        log(f"[fullspace] {dataset}: {json.dumps(results[dataset])}")
    curve = cardinality_sweep(engine, first, log=log) if sweep and first \
        else {}
    gate = compose_gate(os.path.relpath(ckpt_dir, ROOT), manifest, results,
                        curve)
    if write:
        GATE_DIR.mkdir(parents=True, exist_ok=True)
        GATE_PATH.write_text(json.dumps(gate, indent=2,
                                        ensure_ascii=False) + "\n")
    return gate


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(prog="eval.fullspace")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("gate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--datasets", default="banking77")
    p.add_argument("--limit", type=int, default=DEFAULT_ROWS)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-sweep", action="store_true")
    args = ap.parse_args(argv)
    gate = run(args.checkpoint, tuple(args.datasets.split(",")), args.limit,
               args.device, sweep=not args.no_sweep)
    print(json.dumps(gate["table"], indent=2, ensure_ascii=False))
    print(gate.get("cardinality_reading") or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
