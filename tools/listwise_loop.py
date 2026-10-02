#!/usr/bin/env python3
"""Keep the listwise scorer training and retain its strongest checkpoint."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = ROOT / "artifacts" / "listwise-loop"
CHECKPOINT_ROOT = ROOT / "artifacts" / "checkpoints" / "listwise"
STATE = RUN_ROOT / "state.json"
LOG = ROOT / "artifacts" / "logs" / "listwise-loop.log"
TRAINING = ROOT / "artifacts" / "loop" / "TRAINING"
PYTHON = ROOT / ".venv-train" / "bin" / "python"
# Middle arm of the 0.1 / 0.3 / 1.0 ablation in T-aligned-ranking-v2;
# state.json "pairwise_weight" overrides it per arm.
PAIRWISE_WEIGHT = 0.3


def load_state() -> dict:
    if STATE.exists():
        state = json.loads(STATE.read_text())
        if not state.get("active_checkpoint") and state.get("best_checkpoint"):
            state["active_checkpoint"] = state["best_checkpoint"]
            state["active_forced"] = state.get("best_forced")
            state["protocol_baseline"] = (
                "grandfathered at audit 2026-09-30; earlier listwise "
                "promotions did not run the paired gate")
        return state
    return {
        "cycle": 0,
        "best_forced": None,
        "best_checkpoint": None,
        "active_scorer": "listwise",
    }


def save_state(state: dict) -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2) + "\n")


def result_of(checkpoint: Path) -> dict:
    """Read the eval that matches the weights actually left on disk."""
    path = checkpoint / "eval-dev.json"
    if not path.exists():
        return {}
    doc = json.loads(path.read_text())
    figures = doc["figures"]
    manifest = checkpoint / "train_manifest.json"
    retained = {}
    if manifest.exists():
        trained = json.loads(manifest.read_text())
        retained = trained.get("retained") or {}
        pairwise = (trained.get("hyperparams") or {}).get(
            "pairwise_weight", 0.0)
    else:
        pairwise = None
    return {
        "pairwise_weight": pairwise,
        "forced": figures["forced"]["accuracy"],
        "counterfactual_joint":
            figures["counterfactual_joint"]["accuracy"],
        "by_family": {
            family: node["accuracy"]
            for family, node in
            figures["macro_by_family"]["by_family"].items()
        },
        "by_k": {
            key: node["accuracy"]
            for key, node in figures["by_k"].items()
        },
        "retained_at": retained.get("decisions_consumed"),
    }


def register(state: dict, cycle: int, checkpoint: Path) -> None:
    from training.python import ce_finetune as CF

    result = result_of(checkpoint)
    if not result:
        return
    score = float(result["forced"])
    best = state.get("best_forced")
    research_best = best is None or score > float(best)
    if research_best:
        state["best_forced"] = score
        state["best_checkpoint"] = str(checkpoint)
    active = state.get("active_checkpoint")
    gate = None
    promoted = active is None
    if active:
        control = json.loads(
            (Path(active) / "eval-dev.json").read_text())
        tuned = json.loads((checkpoint / "eval-dev.json").read_text())
        gate = CF.compare(control, tuned)
        promoted = bool(gate["pass"])
    if promoted:
        state["active_checkpoint"] = str(checkpoint)
        state["active_forced"] = score
    state.update({
        "cycle": cycle,
        "last_checkpoint": str(checkpoint),
        "last_result": result,
        "last_promoted": promoted,
        "last_research_best": research_best,
        "last_gate": gate,
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                     time.gmtime()),
    })
    state.pop("last_error", None)
    save_state(state)
    prune(state)


def prune(state: dict) -> None:
    """Keep the promoted weights and the last candidate. A rejected
    checkpoint is 1.1 GB of weights nothing will ever load again."""
    keep = {state.get("active_checkpoint"), state.get("best_checkpoint"),
            state.get("last_checkpoint")}
    for path in sorted(CHECKPOINT_ROOT.glob("loop-*")):
        model = path / "model"
        if str(path) in keep or not model.is_dir():
            continue
        shutil.rmtree(model, ignore_errors=True)
    cycles = sorted(RUN_ROOT.glob("cycle-*"))
    for run in cycles[:-2]:
        shutil.rmtree(run / "mix", ignore_errors=True)


def command_env() -> dict:
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(ROOT),
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    })
    return env


def build_mix(cycle: int, run: Path) -> None:
    command = [
        str(PYTHON), "-m", "training.python.listwise_mix",
        "--cycle", str(cycle),
        "--out", str(run / "mix"),
    ]
    subprocess.run(command, cwd=ROOT, env=command_env(), check=True,
                   stdout=subprocess.DEVNULL)


def train(cycle: int, run: Path, checkpoint: Path,
          initial: str | None, pairwise_weight: float) -> None:
    command = [
        str(PYTHON), "-u", "-m",
        "training.python.listwise_finetune", "train",
        "--episodes", str(run / "mix"),
        "--weights", "mdeberta-v3-base-xnli",
        "--init", initial or "none",
        "--out", str(checkpoint),
        "--device", "mps",
        "--budget", "12000",
        "--eval-every", "3000",
        "--holdout", "0.2",
        "--decisions-per-batch", "2",
        "--eval-batch-size", "4",
        "--patience", "4",
        "--seed", str(20260929 + cycle),
        "--pairwise-weight", str(pairwise_weight),
    ]
    with LOG.open("a") as log:
        subprocess.run(command, cwd=ROOT, env=command_env(), check=True,
                       stdout=log, stderr=subprocess.STDOUT)


def main() -> int:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    state = load_state()
    failures = 0
    while True:
        cycle = int(state.get("cycle") or 0) + 1
        run = RUN_ROOT / f"cycle-{cycle:04d}"
        checkpoint = CHECKPOINT_ROOT / f"loop-{cycle:04d}"
        TRAINING.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime()) + "\n")
        try:
            if not (run / "mix" / "manifest.json").exists():
                build_mix(cycle, run)
            if not (checkpoint / "train_manifest.json").exists():
                train(cycle, run, checkpoint,
                      state.get("active_checkpoint")
                      or state.get("best_checkpoint"),
                      float(state.get("pairwise_weight",
                                      PAIRWISE_WEIGHT)))
            register(state, cycle, checkpoint)
            failures = 0
        except Exception as error:
            failures += 1
            state["last_error"] = (f"{type(error).__name__}: {error} "
                                   f"(cycle {cycle}, attempt {failures})")
            save_state(state)
            if failures >= 3:
                # A cycle that cannot train three times running is not
                # transient. Skip it rather than spin on the same seed.
                state["cycle"] = cycle
                save_state(state)
                failures = 0
            time.sleep(30)
            continue
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
