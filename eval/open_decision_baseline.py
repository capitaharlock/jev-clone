"""Evaluate a System One-compatible open model on the local dev battery.

The model implementation stays under ``tmp/jev-research``. This runner turns
its output into the same row contract used by the current promotion gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from data import battery_dev as BD
from eval import metrics_suite as MS
from eval import preflight_refs as PR


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "tmp" / "jev-research" / "tinyjev"
TASK = "T-open-decision-backbone"


def load_tinyjev(source: Path):
    if not source.is_dir():
        raise FileNotFoundError(
            f"{source} missing; clone https://github.com/ankit-aglawe/"
            "tinyjev there")
    sys.path.insert(0, str(source))
    import tinyjev
    return tinyjev


def evaluate(model: str, source: Path, backend: str, quantize: int) -> dict:
    tinyjev = load_tinyjev(source)
    episodes = BD.load_battery()
    agent = tinyjev.load(model, backend=backend, quantize=quantize)
    rows = []
    started = time.perf_counter()
    for episode in episodes:
        candidates = episode["candidates"]
        criteria = {item["id"]: item["text"] for item in candidates}
        payload = {
            "state": episode["state"],
            "questions": {
                "decision": {
                    "type": "choice",
                    "instructions": episode["question"],
                    "criteria": criteria,
                }
            },
        }
        scored = agent.logits(payload)["request:decision"]
        probabilities = dict(zip(scored["candidate_ids"],
                                 scored["probabilities"]))
        probs = [probabilities[item["id"]] for item in candidates]
        rows.append({
            "row_id": episode["id"],
            "family": episode["family"],
            "lang": episode["lang"],
            "variant_group": episode["variant_group"],
            "k": len(candidates),
            "gold_index": next(
                index for index, item in enumerate(candidates)
                if item["id"] == episode["answer"]),
            "pred": max(range(len(probs)), key=probs.__getitem__),
            "probs": probs,
        })
    report = MS.report(
        rows,
        cut={"id": "battery-dev", "split": "dev", "sealed": False},
        model_version=f"{model}:mlx-int{quantize}",
        task=TASK,
        calibration={
            "status": "checkpoint",
            "method": "scalar temperature stored in the pointer head",
        },
        notes=["No battery row was used to train or select this checkpoint."],
    )
    return {
        "task": TASK,
        "status": "measured",
        "battery": {
            "file": "data/battery_dev.jsonl",
            "n": len(episodes),
            "rows_sha256": PR.rows_sha(episodes),
            "sha256": BD.battery_sha(episodes),
            "sealed_cut_read": False,
        },
        "checkpoint": {
            "model_version": f"{model}:mlx-int{quantize}",
            "source": str(source.relative_to(ROOT)),
            "backend": backend,
            "quantize": quantize,
        },
        "rows": rows,
        "report": report,
        "figures": {
            "forced": report["ranking"],
            "counterfactual_joint": report["counterfactual"],
            "macro_by_family": report["macro"],
        },
        "seconds": round(time.perf_counter() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyJev-4B-v1")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--backend", default="mlx")
    parser.add_argument("--quantize", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    doc = evaluate(args.model, args.source, args.backend, args.quantize)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({
        "model": doc["checkpoint"]["model_version"],
        "forced": doc["figures"]["forced"]["accuracy"],
        "joint": doc["figures"]["counterfactual_joint"]["accuracy"],
        "seconds": doc["seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
