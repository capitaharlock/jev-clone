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

Why the §128 arms carry `--dataset-cap 0.18`: without `synth-v1` the
fenced registry covers only 90 % of a mixture at the §65 15 % cap, so the
baseline arm cannot be built at all. Both arms therefore use the SAME
stated cap and the SAME corpus size, and the ablation is a substitution
at fixed budget — which is the decision the operator actually faces,
since the corpus budget and the caps are both fixed.
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

#: the §128 arms need a cap the baseline can actually be built at
ABLATION_DATASET_CAP = 0.18
ABLATION_SAMPLES = backbones.STAGES["stage0"]
ABLATION_ROWS = 39_980

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
             ABLATION_SAMPLES, **common)
         + ["--dataset-cap", str(ABLATION_DATASET_CAP),
            "--mix-target", str(ABLATION_ROWS)],
         "why": "§128 arm WITH grounded synthetic"},
        {"name": "nosynth", "backbone": top[0],
         "run_id": run_mix.run_id("nosynth", top[0], seed),
         "cmd": backbones.train_command(
             top[0], run_mix.run_id("nosynth", top[0], seed),
             ABLATION_SAMPLES, drop=("synth-v1",), **common)
         + ["--dataset-cap", str(ABLATION_DATASET_CAP),
            "--mix-target", str(ABLATION_ROWS)],
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
