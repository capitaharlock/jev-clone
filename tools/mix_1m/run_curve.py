"""Run the §62 curve and the §128 ablation on `decision-mix-clean-1m`.

    .venv-train/bin/python -m tools.mix_1m.run_curve [--device mps] [--dry-run]

Four real training runs, in the order that makes partial results useful,
each one `training/python/train_decision.py` over the corpus published by
`tools.mix_1m.run_mix build` (same seed, same members_sha256):

1. `curve-<top1>`     — 1 M decisions: Stage 0 (250 k) and Stage 1 (1 M)
2. `synth-<top1>`     — 250 k, the §128 arm WITH grounded synthetic
3. `nosynth-<top1>`   — 250 k, the §128 arm WITHOUT it
4. `curve-<top2>`     — 1 M decisions, the second backbone

The gate is rewritten after every run, so `artifacts/gates/T-mix-1m/
gate.json` always states what has actually been measured and what is
still running. Nothing here invents a number: a run that has not reached
a stage leaves that stage `pending`.

The §128 arms used to carry `--dataset-cap 0.18` over a 39 980-row
corpus, because without `synth-v1` the seven-source registry covered only
90 % of a mixture at the §65 15 % cap. The #T-mix-1m widening ended that:
dropping `synth-v1` now leaves 1.65 cap units and a 1 150 382-row
ceiling, so both arms are built at the §§65-66 caps themselves, at
250 000 rows each — ONE pass over the corpus for a 250 000-decision
budget. The ablation is still a substitution at fixed budget, which is
the decision the operator actually faces, but it is now made under one
rule instead of two.

Every run here states `--mix-target` and asserts the mixture covers its
`--max-samples`: the trainer aborts with `MixShortfallError` rather than
looping a short corpus, which is how a 1 M-decision curve came to be
measured on 39 981 rows.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.mix_1m import backbones, run_mix  # noqa: E402

#: §128 at the §§65-66 caps: the widening bought the baseline arm its
#: standard cap, so there is no cap exception left to state.
ABLATION_SAMPLES = backbones.STAGES["stage0"]
#: one epoch, not six: the arms' corpus is as large as their budget
ABLATION_ROWS = ABLATION_SAMPLES

LOG = os.path.join(ROOT, "artifacts", "mix-1m", "curve.jsonl")


def plan(seed: int, device: str, batch_size: int) -> list:
    """The four runs, in the order that publishes value earliest."""
    top = backbones.top2()
    common = {"seed": seed, "device": device, "batch_size": batch_size}
    return [
        {"name": "curve", "backbone": top[0],
         "run_id": run_mix.run_id("curve", top[0], seed),
         "cmd": backbones.train_command(
             top[0], run_mix.run_id("curve", top[0], seed),
             backbones.STAGES["stage1"], **common),
         "why": "§62 Stage 0 + Stage 1 for the top-1 backbone"},
        {"name": "synth", "backbone": top[0],
         "run_id": run_mix.run_id("synth", top[0], seed),
         "cmd": backbones.train_command(
             top[0], run_mix.run_id("synth", top[0], seed),
             ABLATION_SAMPLES, mix_target=ABLATION_ROWS, **common),
         "why": "§128 arm WITH grounded synthetic"},
        {"name": "nosynth", "backbone": top[0],
         "run_id": run_mix.run_id("nosynth", top[0], seed),
         "cmd": backbones.train_command(
             top[0], run_mix.run_id("nosynth", top[0], seed),
             ABLATION_SAMPLES, drop=("synth-v1",),
             mix_target=ABLATION_ROWS, **common),
         "why": "§128 arm WITHOUT it — the baseline"},
        {"name": "curve", "backbone": top[1],
         "run_id": run_mix.run_id("curve", top[1], seed),
         "cmd": backbones.train_command(
             top[1], run_mix.run_id("curve", top[1], seed),
             backbones.STAGES["stage1"], **common),
         "why": "§62 Stage 0 + Stage 1 for the second backbone"},
    ]


def log(record: dict) -> None:
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def refresh_gate(seed: int) -> dict:
    """Rewrite gate.json from the corpus on disk plus whatever has run."""
    stage = run_mix.build(None, seed, write=True)
    gate = run_mix.compose(stage, run_mix.training_block(seed), seed)
    run_mix.write_gate(gate)
    return gate


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(prog="run_curve")
    ap.add_argument("--seed", type=int, default=run_mix.DEFAULT_SEED)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None,
                    help="run only the step with this name (curve/synth/nosynth)")
    args = ap.parse_args(argv[1:])

    steps = [s for s in plan(args.seed, args.device, args.batch_size)
             if args.only is None or s["name"] == args.only]
    if args.dry_run:
        for step in steps:
            print(f"# {step['why']}\n{' '.join(step['cmd'])}\n")
        return 0

    log({"t": "start", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "steps": [s["run_id"] for s in steps], "device": args.device})
    failed = []
    for step in steps:
        t0 = time.time()
        log({"t": "run-start", "run_id": step["run_id"], "why": step["why"],
             "cmd": step["cmd"]})
        proc = subprocess.run(step["cmd"], cwd=ROOT,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True)
        tail = "\n".join(proc.stdout.splitlines()[-25:]) if proc.stdout else ""
        log({"t": "run-end", "run_id": step["run_id"], "rc": proc.returncode,
             "elapsed_s": round(time.time() - t0, 1), "tail": tail})
        if proc.returncode != 0:
            failed.append(step["run_id"])
            print(f"[curve] {step['run_id']} FAILED rc={proc.returncode}\n{tail}",
                  file=sys.stderr)
        gate = refresh_gate(args.seed)
        log({"t": "gate", "run_id": step["run_id"], "pass": gate["pass"],
             "failed_checks": gate["failed_checks"]})
        print(f"[curve] {step['run_id']} rc={proc.returncode} "
              f"gate pass={gate['pass']} failed={gate['failed_checks']}")

    gate = refresh_gate(args.seed)
    log({"t": "done", "failed_runs": failed, "pass": gate["pass"]})
    print(json.dumps({"failed_runs": failed, "gate_pass": gate["pass"],
                      "failed_checks": gate["failed_checks"]}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
