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

This module reads the RESERVED cut (`eval.cuts`, rule R7): every run is
logged in `artifacts/gates/T-eval-cardinality/test-queries.json`. Choosing
between arms is done on the development cut — `eval.scoreboard --cut dev` —
and never here.

CLI:
    CKPT=artifacts/checkpoints/decision/lever-stack-d512-prior/stage-001000000
    .venv-train/bin/python -m eval.fullspace gate --checkpoint $CKPT \
        --reason "why the reserved cut is being read"
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
            "citation": True,
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
            "citation": True,
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
                 cfg: SamplerConfig | None = None, split: str | None = None,
                 keep: set | None = None) -> list:
    """One `Sample` per row with EVERY label of the space as an option.

    The option order is the sorted label space, identical on every row, so
    the gold's position carries no signal: a head that learned "the answer
    is last" — which our sampler's layout could teach — scores at chance
    here, and that is the point.

    `split` and `keep` are what `eval.cuts` injects: the sealed development
    cut is a seeded subset of another split of the same dataset, and it is
    named by sample id (`dataset:row:question`) so the seal says exactly
    which rows were scored.
    """
    from dataclasses import asdict
    cfg = U.eval_config(cfg)
    cfg = SamplerConfig(**{**asdict(cfg),
                           "split": split or SPLIT.get(dataset, "test")})
    # deliberately NOT `make_sampler`: that one `restrict`s the pool to a
    # label subset, and the whole point here is that nothing is restricted.
    sampler = OptionSetSampler(datasets=(dataset,), config=cfg,
                               rows_per_dataset=U.ROWS_CAP,
                               root=U.T.PREFETCH_DIR)
    out: list = []
    for n, row in enumerate(sampler.rows[dataset]):
        for question in row.get("questions", []):
            sample_id = (f"{dataset}:{dataset}-{n}:"
                         f"{question.get('id', '')}")
            if keep is not None and sample_id not in keep:
                continue
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
    return tally(samples, entries)


def tally(samples: list, entries: list) -> dict:
    """The published numbers of one cut, off rows already scored.

    Split out of `score` so a caller that needs the entries for something
    else — `eval.scoreboard` wants the confusion matrix and the prediction
    histogram off the same forward pass — publishes the SAME arithmetic
    instead of a second implementation of it.
    """
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
        "hits": hits,
        "cardinality": k,
        "chance": round(1.0 / k, 6) if k else None,
        "accuracy": round(hits / n, 6) if n else None,
        "accuracy_ci95": [round(lo, 6), round(hi, 6)],
        "hits_options_only": forced,
        "accuracy_options_only": round(forced / n, 6) if n else None,
        "accuracy_options_only_ci95": [round(flo, 6), round(fhi, 6)],
        "abstain_rate": round(abstain / n, 6) if n else None,
        # `beats_chance` is about the number this artifact publishes, which
        # is `accuracy` — the [K + 1] argmax, abstentions included. Deciding
        # it with `flo` (the forced ranking) would let a head that abstains
        # on every row publish `accuracy: 0` beside `beats_chance: true`.
        "beats_chance": bool(n and lo > 1.0 / k),
        "ranking_beats_chance": bool(n and flo > 1.0 / k),
        "beats_chance_decided_by": "accuracy_ci95[0] > chance",
        "ranking_beats_chance_decided_by":
            "accuracy_options_only_ci95[0] > chance — the ranking with "
            "`unknown` taken out of the race, a diagnostic",
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
    """The sweep, stated as counts and intervals and nothing else.

    An earlier version of this function wrote "the head clears chance up to
    K=40 and stops by K=77", which reads as a progression — accuracy holding
    and then breaking at some cardinality. The intervals do not carry that:
    on the published cut K=5 (0.215, CI [0.191, 0.242] against chance 0.200)
    and K=20 do NOT clear their chance rate while K=8 and K=40 do, which is
    not a monotone anything. So this returns which K clear their chance by
    the 95 % lower bound and which do not, says the points are out of order,
    and leaves the explanation to whoever can measure one (#T-option-text,
    #T-bigk-optsets). Rule 2-bis of #T-eval-cardinality.
    """
    ks = sorted(int(k) for k in sweep)
    if not ks:
        return "not enough points to read"

    def cell(k: int) -> str:
        rep = sweep[str(k)]
        return (f"K={k}: {rep.get('hits')}/{rep.get('n')} = "
                f"{rep.get('accuracy')} CI {rep.get('accuracy_ci95')} vs "
                f"chance {rep.get('chance')}")

    clears = [k for k in ks if sweep[str(k)].get("beats_chance")]
    flat = "; ".join(cell(k) for k in ks)
    if not clears:
        return (f"no cardinality on this cut clears its chance rate by the "
                f"95 % lower bound — {flat}")
    misses = [k for k in ks if k not in clears]
    monotone = clears == [k for k in ks if k <= max(clears)]
    tail = ("" if monotone else
            " — these are OUT OF ORDER: a K that clears sits above one that "
            "does not, so there is no progression to read off this sweep and "
            "no cardinality at which the advantage can be said to break")
    return (f"clears chance by the 95 % lower bound at K={clears}, does not "
            f"at K={misses}{tail}. Counts: {flat}")


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
        write: bool = True, sweep: bool = True, reason: str = "",
        log=print) -> dict:
    """Score the RESERVED cut. Rule R7: the query goes in the ledger.

    These are the 3 080 official test rows — the cut `eval.cuts` keeps
    reserved — so running this is a read of the test set and is written down
    with its date, its checkpoint and its reason. Arms are chosen on the dev
    cut (`eval.scoreboard`, `--cut dev`), never here.
    """
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
        from eval import cuts as K  # here: `eval.cuts` imports this module
        K.record_query(
            os.path.relpath(ckpt_dir, ROOT),
            reason or "eval.fullspace gate, no reason given on the command "
                      "line — the read happened anyway and is logged as "
                      "unexplained",
            os.path.relpath(GATE_PATH, ROOT), rows=limit, by="eval.fullspace")
        gate["test_cut_query_logged"] = os.path.relpath(K.LEDGER_PATH, ROOT)
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
    p.add_argument("--reason", default="",
                   help="why the reserved test cut is being read (R7); it "
                        "goes into artifacts/gates/T-eval-cardinality/"
                        "test-queries.json")
    args = ap.parse_args(argv)
    gate = run(args.checkpoint, tuple(args.datasets.split(",")), args.limit,
               args.device, sweep=not args.no_sweep, reason=args.reason)
    print(json.dumps(gate["table"], indent=2, ensure_ascii=False))
    print(gate.get("cardinality_reading") or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
