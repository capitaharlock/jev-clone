"""Continuous training of the publishable decision model.

Each cycle starts from the best own checkpoint, trains on the next local
shard, and keeps that checkpoint only when unseen texts clear chance by a
wider margin. A STOP file ends the loop at the next cycle boundary.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOOP = ROOT / "artifacts" / "own-loop"
STOP = LOOP / "STOP"
STATE = LOOP / "state.json"
HISTORY = LOOP / "history.jsonl"
CHAMPION = LOOP / "champion"
EPISODES = Path(os.environ.get(
    "OWN_EPISODES_DIR",
    str(ROOT.parent / "jev.clone" / "artifacts" / "episodes-rule"),
))
PYTHON = os.environ.get("OWN_PYTHON", sys.executable)
SEED_CHECKPOINT = (ROOT / "artifacts" / "checkpoints" / "own"
                   / "unseen-15m" / "model.safetensors")
STEPS = int(os.environ.get("OWN_LOOP_STEPS", "1000"))
LIMIT = int(os.environ.get("OWN_LOOP_LIMIT", "1200"))


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def episode_files(root: Path = EPISODES) -> list[Path]:
    return sorted(path for path in root.glob("batch-*/episodes.jsonl")
                  if path.is_file())


def gain(result: dict) -> float:
    return float(result["holdout_wilson_lower"]) - float(result["holdout_chance"])


def promotable(result: dict, best_gain: float) -> bool:
    checks = result.get("checks") or {}
    return bool(checks.get("holdout_clean")
                and checks.get("unseen_beats_chance")
                and checks.get("own_weights")
                and gain(result) > best_gain)


def seed_champion(state: dict) -> None:
    if (CHAMPION / "model.safetensors").is_file() or not SEED_CHECKPOINT.is_file():
        return
    CHAMPION.mkdir(parents=True, exist_ok=True)
    for name in ("model.safetensors", "config.json", "train.json"):
        source = SEED_CHECKPOINT.parent / name
        if source.is_file():
            shutil.copy2(source, CHAMPION / name)
    seeded = read_json(CHAMPION / "train.json")
    if seeded:
        state["best_gain"] = gain(seeded)
        state["champion_cycle"] = "seed-unseen-15m"


def run_cycle(state: dict) -> dict:
    files = episode_files()
    if not files:
        raise FileNotFoundError(f"no episodes under {EPISODES}")
    index = int(state.get("next_index") or 0)
    source = files[index % len(files)]
    out = LOOP / "cycles" / f"cycle-{index:06d}"
    command = [
        PYTHON, "-u", "-m", "training.python.own_train",
        "--episodes", str(source),
        "--out", str(out),
        "--gate", str(out / "gate.json"),
        "--steps", str(STEPS),
        "--limit", str(LIMIT),
        "--unseen-fraction", "0.5",
        "--batch-size", "8",
        "--max-length", "128",
        "--log-every", "200",
        "--seed", str(1789 + index),
    ]
    champion = CHAMPION / "model.safetensors"
    if champion.is_file():
        command.extend(["--init", str(champion)])
    state.update({"status": "training", "cycle": index,
                  "source": str(source), "pid": os.getpid(),
                  "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                               time.gmtime())})
    write_json(STATE, state)
    subprocess.run(command, cwd=ROOT, check=True)
    result = read_json(out / "gate.json")
    result["cycle"] = index
    result["gain"] = gain(result)
    best = float(state.get("best_gain") if state.get("best_gain") is not None
                 else -1)
    if promotable(result, best) or not champion.is_file():
        CHAMPION.mkdir(parents=True, exist_ok=True)
        for name in ("model.safetensors", "config.json", "train.json"):
            shutil.copy2(out / name, CHAMPION / name)
        state["best_gain"] = result["gain"]
        state["champion_cycle"] = index
        result["promoted"] = True
    else:
        result["promoted"] = False
    with HISTORY.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "cycle": index,
            "promoted": result["promoted"],
            "gain": result["gain"],
            "holdout_accuracy": result.get("holdout_accuracy"),
            "holdout_chance": result.get("holdout_chance"),
            "holdout_rows": result.get("holdout_rows"),
            "train_rows": result.get("train_rows"),
            "source": source.name,
            "pass": result.get("pass"),
        }, sort_keys=True) + "\n")
    state.update({"status": "idle", "next_index": index + 1,
                  "last_gain": result["gain"], "last_promoted": result["promoted"],
                  "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                               time.gmtime())})
    write_json(STATE, state)
    print(json.dumps({"cycle": index, "promoted": result["promoted"],
                      "gain": round(result["gain"], 4)}), flush=True)
    return result


def main() -> int:
    once = "--once" in sys.argv
    LOOP.mkdir(parents=True, exist_ok=True)
    state = read_json(STATE)
    seed_champion(state)
    while not STOP.exists():
        try:
            run_cycle(state)
        except Exception as exc:
            state["status"] = "error"
            state["last_error"] = f"{type(exc).__name__}: {exc}"
            state["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                 time.gmtime())
            write_json(STATE, state)
            print(state["last_error"], file=sys.stderr, flush=True)
            if once:
                return 1
            time.sleep(60)
            continue
        if once:
            return 0
        time.sleep(15)
    state["status"] = "stopped"
    write_json(STATE, state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
