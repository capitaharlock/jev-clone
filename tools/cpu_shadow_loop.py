#!/usr/bin/env python3
"""Run independent MiniLM experiments while the main model uses MPS."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "artifacts" / "experiments" / "cpu-shadow-battery"
LOG = ROOT / "artifacts" / "logs" / "cpu-shadow-loop.log"
FIRST = BASE / "model" / "train_manifest.json"
BEST = BASE / "best.json"
PYTHON = ROOT / ".venv-train" / "bin" / "python"


def wait_for_first_run() -> None:
    while not FIRST.exists():
        time.sleep(30)


def score_of(curve: Path) -> float | None:
    if not curve.exists():
        return None
    rows = [json.loads(line) for line in curve.read_text().splitlines()
            if line.strip()]
    return float(rows[-1]["forced"]) if rows else None


def record_best(run: Path, seed: int) -> None:
    score = score_of(run / "model" / "curve.jsonl")
    if score is None:
        return
    previous = json.loads(BEST.read_text()) if BEST.exists() else {}
    if score >= float(previous.get("forced", -1)):
        BEST.write_text(json.dumps({
            "forced": score,
            "seed": seed,
            "run": str(run.relative_to(ROOT)),
        }, indent=2) + "\n")


def generate(run: Path, seed: int) -> None:
    from data import episode_gen as EG
    from data import rule_variety as RV

    EG.run(
        n=10_000,
        seed=seed,
        prose="local",
        teacher="stub",
        out_dir=str(run / "data"),
        families=list(RV.FAMILIES),
        variety=16,
        group_prefix=f"cpu{seed}-",
        layout="battery",
        command=f"cpu-shadow battery-layout seed={seed}",
    )


def train(run: Path) -> int:
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(ROOT),
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "16",
        "VECLIB_MAXIMUM_THREADS": "16",
    })
    command = [
        str(PYTHON), "-u", "-m", "training.python.ce_finetune", "train",
        "--episodes", str(run / "data"),
        "--weights", "minilmv2-l6-mnli-xnli",
        "--init", "none",
        "--out", str(run / "model"),
        "--device", "cpu",
        "--budget", "10000",
        "--eval-every", "2500",
        "--holdout", "0.2",
        "--split-seed", "20260927",
    ]
    with LOG.open("a") as log:
        return subprocess.run(command, cwd=ROOT, env=env,
                              stdout=log, stderr=subprocess.STDOUT).returncode


def main() -> int:
    BASE.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    wait_for_first_run()
    record_best(BASE, 20260929)
    iteration = 1
    while True:
        seed = 20260929 + iteration
        run = BASE / f"run-{iteration:04d}"
        if not (run / "model" / "train_manifest.json").exists():
            generate(run, seed)
            if train(run) != 0:
                time.sleep(60)
                continue
        record_best(run, seed)
        iteration += 1


if __name__ == "__main__":
    sys.exit(main())
