"""The curriculum loop that produced the published checkpoint.

Each cycle continues the latest weights on a fresh batch of the current
rung and scores that rung's fixed dev file. A rung is cleared at 90% forced
accuracy; the next cycle moves up and keeps 20% of earlier rungs as replay.
The last rung trains on the real Open Food Facts episodes and is scored on
the frozen benchmark, and it never clears: it runs until stopped, and the
best cycle on the benchmark is kept as the champion.

Every cycle appends one line to `history.jsonl`. Touching `STOP` in the
run folder ends the loop at the next cycle boundary.

    python -m asimovia.training.loop --run runs/curriculum
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ..data import curriculum
from ..data.io import read_jsonl, write_json, write_jsonl
from ..model.backbone import DecisionModel

REAL_TRAIN = Path("work/off/train.jsonl")
BENCHMARK = Path("benchmark/product_specs.dev.jsonl")
MAX_LENGTH = 1536
CLEAR_AT = 0.90
REPLAY = 0.2
DEV_ROWS = 1000
HARD_COPIES = 1

RUNGS = [
    {"id": "r0-most", "steps": 600, "batch": 16, "rows": 9600},
    {"id": "r1-pair", "steps": 600, "batch": 16, "rows": 9600},
    {"id": "r2-panel", "steps": 800, "batch": 16, "rows": 12800},
    {"id": "r3-units", "steps": 800, "batch": 8, "rows": 6400},
    {"id": "r4-constraints", "steps": 800, "batch": 8, "rows": 6400},
    {"id": "r5-real", "steps": 1000, "batch": 8, "rows": 0},
]
CAPPED = ("at most", "must stay", "como máximo", "no puede pasar")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def hard(row: dict) -> bool:
    """The families where the model trailed: caps, K=5 and sodium in mg."""
    question = row["question"].lower()
    return (
        any(word in question for word in CAPPED)
        or len(row["candidates"]) >= 5
        or any("sodium" in c["text"] or "sodio" in c["text"] for c in row["candidates"])
    )


def fits(rows: list[dict], limit: int = MAX_LENGTH) -> float:
    """Share of rows whose question and every option survive packing whole."""
    whole = 0
    for row in rows:
        question, _, option = DecisionModel.budgets(limit, len(row["candidates"]))
        if len(row["question"].encode()) + 2 <= question and all(
            len(c["text"].encode()) + 2 <= option for c in row["candidates"]
        ):
            whole += 1
    return whole / len(rows) if rows else 0.0


def dev_file(run: Path, rung: str, benchmark: Path) -> Path:
    if rung == "r5-real":
        return benchmark
    path = run / "dev" / f"{rung}.jsonl"
    if not path.is_file():
        write_jsonl(path, curriculum.batch(rung, DEV_ROWS, seed=900 + curriculum.RUNGS.index(rung)))
    return path


def train_rows(rung: dict, cycle: int, real_train: Path) -> list[dict]:
    if rung["id"] != "r5-real":
        return curriculum.batch(rung["id"], rung["rows"], seed=10_000 + cycle, replay=REPLAY)
    rows = read_jsonl(real_train)
    rows += [row for row in rows if hard(row)] * HARD_COPIES
    # Synthetic replay keeps the constraint skill sharp while the real
    # cards, with their noisier wording, are learned.
    rows += curriculum.batch("r4-constraints", len(rows) // 8, seed=50_000 + cycle, replay=0.5)
    return rows


def update(state: dict, rung: dict, result: dict, cycle: int, seconds: float, rows: int) -> dict:
    """Bookkeeping after one cycle; returns the history line."""
    accuracy = result["holdout_forced_accuracy"]
    best = state.setdefault("best", {})
    cleared = rung["id"] != "r5-real" and accuracy >= CLEAR_AT
    best[rung["id"]] = max(best.get(rung["id"], 0.0), accuracy)
    if cleared:
        state["rung_index"] = min(state.get("rung_index", 0) + 1, len(RUNGS) - 1)
        state.setdefault("cleared", {})[rung["id"]] = cycle
    return {
        "cycle": cycle,
        "rung": rung["id"],
        "accuracy": accuracy,
        "chance": result["holdout_forced_chance"],
        "wilson_lower": result["holdout_forced_wilson_lower"],
        "rows": result["holdout_rows"],
        "loss": result["last_loss"],
        "train_rows": rows,
        "cleared": cleared,
        "seconds": round(seconds, 1),
        "at": now(),
    }


def run_cycle(run: Path, state: dict, python: str, real_train: Path, benchmark: Path) -> dict:
    cycle = int(state.get("next_cycle", 0))
    rung = RUNGS[int(state.get("rung_index", 0))]
    dev = dev_file(run, rung["id"], benchmark)
    fit = fits(read_jsonl(dev)[:500])
    if fit < 0.97:
        raise RuntimeError(f"{rung['id']}: only {fit:.1%} of dev rows fit")
    folder = run / "cycles" / f"cycle-{cycle:06d}"
    train = folder / "train.jsonl"
    rows = write_jsonl(train, train_rows(rung, cycle, real_train))
    command = [
        python,
        "-u",
        "-m",
        "asimovia.training.train",
        "--episodes",
        str(train),
        "--eval",
        str(dev),
        "--out",
        str(folder),
        "--steps",
        str(rung["steps"]),
        "--batch-size",
        str(rung["batch"]),
        "--max-length",
        str(MAX_LENGTH),
        "--lr",
        "3e-4",
        "--warmup",
        "100",
        "--log-every",
        "100",
        "--eval-every",
        "200",
        "--seed",
        str(20261020 + cycle),
    ]
    latest = run / "latest"
    if (latest / "model.safetensors").is_file():
        command += ["--init", str(latest / "model.safetensors")]
    state.update({"status": "training", "cycle": cycle, "rung": rung["id"], "updated_utc": now()})
    write_json(run / "state.json", state)
    started = time.time()
    with (run / "train.log").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"cycle": cycle, "rung": rung["id"], "at": now()}) + "\n")
        handle.flush()
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)
    seconds = time.time() - started
    result = json.loads((folder / "train.json").read_text(encoding="utf-8"))
    train.unlink(missing_ok=True)
    names = ("model.safetensors", "config.json", "train.json")
    latest.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy2(folder / name, latest / name)
    champion = state.get("champion") or {}
    if rung["id"] == "r5-real" and result["holdout_forced_accuracy"] > champion.get(
        "accuracy", 0.0
    ):
        (run / "champion").mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copy2(folder / name, run / "champion" / name)
        state["champion"] = {"cycle": cycle, "accuracy": result["holdout_forced_accuracy"]}
    # Only the latest and the champion are kept; a cycle folder is 68 MB.
    (folder / "model.safetensors").unlink(missing_ok=True)
    line = update(state, rung, result, cycle, seconds, rows)
    with (run / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, sort_keys=True) + "\n")
    state.update({"status": "idle", "next_cycle": cycle + 1, "last": line, "updated_utc": now()})
    write_json(run / "state.json", state)
    print(json.dumps(line), flush=True)
    return line


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    cli.add_argument("--run", type=Path, default=Path("runs/curriculum"))
    cli.add_argument("--real-train", type=Path, default=REAL_TRAIN)
    cli.add_argument("--benchmark", type=Path, default=BENCHMARK)
    cli.add_argument(
        "--cycles", type=int, default=0, help="stop after this many cycles; 0 runs until STOP"
    )
    cli.add_argument("--python", default=os.environ.get("PYTHON", sys.executable))
    args = cli.parse_args(argv)
    args.run.mkdir(parents=True, exist_ok=True)
    state_path = args.run / "state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    done = 0
    while not (args.run / "STOP").exists():
        run_cycle(args.run, state, args.python, args.real_train, args.benchmark)
        done += 1
        if args.cycles and done >= args.cycles:
            break
    state.update({"status": "stopped", "updated_utc": now()})
    write_json(state_path, state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
