"""Verification gate for #T-torch-stack — a real forward pass, measured.

Proves, on this machine and with no imputed numbers, that:

1. torch + transformers are installed in `.venv-train` and MPS is live;
2. the pinned backbones under `artifacts/weights/` match their SHA-256
   manifests (a mismatch fails the gate, it does not warn);
3. `encode_state` runs a real forward pass over a REAL V1 schema row on
   both MPS and CPU, with the same output up to tolerance;
4. the latency of that forward is measured on each device.

Writes `artifacts/gates/T-torch-stack/gate.json`.

Usage:
    .venv-train/bin/python tools/smoke_torch.py [--runs 20] [--tol 0.001]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.schema import Example, Option, Question, validate  # noqa: E402
from model import weights  # noqa: E402
from model.encoder import (  # noqa: E402
    device_report,
    encode_state,
    first_row,
    load_backbone,
    prefetch_path,
    state_text,
)

GATE_DIR = os.path.join(ROOT, "artifacts", "gates", "T-torch-stack")
DATASET = "banking77"


def _as_example(row: dict) -> Example:
    return Example(
        state=row["state"],
        split=row.get("split", "train"),
        questions=[Question(
            id=q["id"], kind=q["kind"],
            options=[Option(id=o["id"], text=o["text"])
                     for o in q.get("options", [])],
            answer=q.get("answer"), teacher_conf=q.get("teacher_conf"),
            weights=q.get("weights")) for q in row.get("questions", [])],
    )


def _latency(backbone, state: str, runs: int) -> dict:
    for _ in range(3):  # warm-up: first forward pays kernel compilation
        encode_state(backbone, state)
    samples = [encode_state(backbone, state)["ms"] for _ in range(runs)]
    samples.sort()
    return {
        "runs": runs,
        "p50_ms": round(statistics.median(samples), 3),
        "p95_ms": round(samples[min(len(samples) - 1,
                                    int(0.95 * len(samples)))], 3),
        "mean_ms": round(statistics.fmean(samples), 3),
    }


def run(runs: int = 20, tol: float = 1e-3) -> dict:
    import torch
    import transformers

    row = first_row(prefetch_path(DATASET))
    schema_errors = validate(_as_example(row))
    state = state_text(row)

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.insert(0, "mps")

    results, ok = [], not schema_errors
    for backbone_id in weights.BACKBONES:
        report = weights.verify(backbone_id)
        entry = {"id": backbone_id, "hash_check": report["ok"],
                 "reason": report["reason"],
                 "license": weights.BACKBONES[backbone_id]["license"],
                 "revision": weights.BACKBONES[backbone_id]["revision"],
                 "sha256": {f: v.get("actual")
                            for f, v in report["files"].items()},
                 "devices": {}}
        if not report["ok"]:
            ok = False
            results.append(entry)
            continue
        pooled = {}
        for dev in devices:
            b = load_backbone(backbone_id, dev)
            out = encode_state(b, state)
            entry["hidden_size"] = b.hidden_size
            entry["params"] = b.params
            entry["n_tokens"] = out["n_tokens"]
            shape = tuple(out["tokens"].shape)
            expected = (1, out["n_tokens"], b.hidden_size)
            entry["devices"][dev] = {
                "tokens_shape": list(shape),
                "pooled_shape": list(out["pooled"].shape),
                "shape_ok": shape == expected,
                "latency": _latency(b, state, runs),
            }
            ok = ok and shape == expected
            pooled[dev] = out["pooled"].detach().to("cpu").float()
        if len(pooled) == 2:
            a, c = pooled["mps"], pooled["cpu"]
            diff = float((a - c).abs().max())
            cos = float(torch.nn.functional.cosine_similarity(a, c).mean())
            entry["mps_vs_cpu"] = {
                "max_abs_diff": diff, "cosine": cos, "tol": tol,
                "within_tol": diff <= tol,
            }
            ok = ok and diff <= tol
        else:
            entry["mps_vs_cpu"] = {"skipped": "MPS not available"}
        results.append(entry)

    gate = {
        "task": "T-torch-stack",
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pass": bool(ok),
        "versions": {"python": sys.version.split()[0],
                     "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "device": device_report(),
        "sample": {"dataset": DATASET, "state": state,
                   "question_id": row["questions"][0]["id"],
                   "n_options": len(row["questions"][0]["options"]),
                   "schema_v1_errors": schema_errors},
        "backbones": results,
    }
    os.makedirs(GATE_DIR, exist_ok=True)
    with open(os.path.join(GATE_DIR, "gate.json"), "w") as fh:
        json.dump(gate, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return gate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()
    gate = run(runs=args.runs, tol=args.tol)
    for b in gate["backbones"]:
        line = [f"{b['id']}: hash={'ok' if b['hash_check'] else 'FAIL'}"]
        for dev, d in b.get("devices", {}).items():
            line.append(f"{dev} {d['tokens_shape']} "
                        f"p50={d['latency']['p50_ms']}ms")
        cmp_ = b.get("mps_vs_cpu", {})
        if "max_abs_diff" in cmp_:
            line.append(f"|mps-cpu|max={cmp_['max_abs_diff']:.2e} "
                        f"cos={cmp_['cosine']:.6f}")
        print(" · ".join(line))
    print(f"gate pass={gate['pass']} -> {GATE_DIR}/gate.json")
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
