"""Productor sin fin (#T-episode-scale): lotes por regla + Qwen opcional.

Proceso A de `docs/bucle-infinito.md`. CPU para la regla (miles/s). Qwen
solo si no existe `artifacts/loop/TRAINING` y `QWEN_STREAM=1`.

    PYTHONPATH=. .venv-train/bin/python -u -m data.stream run --forever
    PYTHONPATH=. python3 -m data.stream status
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULE_ROOT = ROOT / "artifacts" / "episodes-rule"
QWEN_ROOT = ROOT / "artifacts" / "episodes-qwen" / "stream"
TRAINING = ROOT / "artifacts" / "loop" / "TRAINING"
HEARTBEAT = ROOT / "artifacts" / "loop" / "producer.json"
PILOT = ROOT / "artifacts" / "episodes-qwen" / "pilot-2k" / "verified.jsonl"

DEFAULT_RULE_N = 5000
DEFAULT_QWEN_N = 10
MAX_RULE_BATCHES = 48


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def list_rule_batches() -> list[int]:
    if not RULE_ROOT.exists():
        return []
    out = []
    for p in RULE_ROOT.iterdir():
        if p.is_dir() and p.name.startswith("batch-") and (p / "episodes.jsonl").exists():
            try:
                out.append(int(p.name.split("-", 1)[1]))
            except ValueError:
                continue
    return sorted(out)


def next_rule_batch() -> int:
    got = list_rule_batches()
    return (got[-1] + 1) if got else 1


def training_paused() -> bool:
    return TRAINING.exists()


def write_heartbeat(doc: dict) -> None:
    HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
    tmp = HEARTBEAT.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(HEARTBEAT)


def publish_rule(batch: int, n: int, variety: int) -> dict:
    from data import rule_variety as RV
    return RV.publish(batch, n, variety=variety)


def publish_qwen(batch: int, n: int) -> dict | None:
    """Un lote chico. None si Qwen no responde o hay TRAINING."""
    if training_paused():
        return None
    if os.environ.get("QWEN_STREAM", "1") != "1":
        return None
    os.environ.setdefault("EPISODE_GEN_BULK", "1")
    from data import episode_gen as EG
    out = QWEN_ROOT / f"batch-{batch:04d}"
    try:
        stats = EG.run(
            n=n, seed=10_000 + batch, prose="qwen", teacher="qwen",
            out_dir=str(out), concurrency=1, batch=1)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return {"ok": True, "out": str(out), **stats}


def status() -> dict:
    batches = list_rule_batches()
    return {
        "measured_utc": utcnow(),
        "rule_batches": batches,
        "n_rule_batches": len(batches),
        "next_rule_batch": next_rule_batch(),
        "training_paused": training_paused(),
        "qwen_stream": os.environ.get("QWEN_STREAM", "1"),
        "pilot_present": PILOT.exists(),
        "heartbeat": (json.loads(HEARTBEAT.read_text())
                      if HEARTBEAT.exists() else None),
    }


def run_forever(rule_n: int, variety: int, qwen_n: int,
                sleep_s: int, max_batches: int) -> None:
    RULE_ROOT.mkdir(parents=True, exist_ok=True)
    QWEN_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[stream] start rule_n={rule_n} qwen_n={qwen_n} "
          f"max_batches={max_batches}", flush=True)
    while True:
        paused = training_paused()
        n_have = len(list_rule_batches())
        action = "idle"
        detail: dict = {}
        if paused:
            action = "pause-training"
        elif n_have >= max_batches:
            action = "cap"
        else:
            bid = next_rule_batch()
            print(f"[stream] rule batch {bid} n={rule_n}", flush=True)
            detail = publish_rule(bid, rule_n, variety)
            action = "rule"
            qwen = publish_qwen(bid, qwen_n)
            if qwen:
                detail["qwen"] = qwen
        beat = {
            "measured_utc": utcnow(),
            "action": action,
            "paused": paused,
            "n_rule_batches": len(list_rule_batches()),
            "detail_keys": sorted(detail.keys()) if isinstance(detail, dict) else [],
        }
        if action == "rule":
            beat["last_rule_out"] = detail.get("out")
            beat["last_rule_n"] = detail.get("n_episodes")
        write_heartbeat(beat)
        print(f"[stream] {action} batches={beat['n_rule_batches']}", flush=True)
        time.sleep(sleep_s if action != "rule" else max(5, sleep_s // 6))


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m data.stream")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    r = sub.add_parser("run")
    r.add_argument("--forever", action="store_true")
    r.add_argument("--rule-batch", type=int, default=DEFAULT_RULE_N)
    r.add_argument("--qwen-batch", type=int, default=DEFAULT_QWEN_N)
    r.add_argument("--variety", type=int, default=16)
    r.add_argument("--sleep", type=int, default=120)
    r.add_argument("--max-batches", type=int, default=MAX_RULE_BATCHES)
    args = ap.parse_args(argv)
    if args.cmd == "status":
        print(json.dumps(status(), ensure_ascii=False, indent=2))
        return 0
    if not args.forever:
        print("refusing: the producer is a 24h process; pass --forever",
              file=sys.stderr)
        return 2
    run_forever(args.rule_batch, args.variety, args.qwen_batch,
                args.sleep, args.max_batches)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
