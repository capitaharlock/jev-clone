"""Qwen episode producer, separate from the rule stream.

The rule producer stays on CPU. This process keeps calling the local
Qwen server and does not stop because a trainer is alive. One batch
that died mid-write is resumed; a published batch is never rewritten.

    PYTHONPATH=. .venv-train/bin/python -u -m data.qwen_stream run --forever
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QWEN_ROOT = ROOT / "artifacts" / "episodes-qwen" / "stream"
HEARTBEAT = ROOT / "artifacts" / "loop" / "qwen.json"


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _manifest(path: Path) -> dict:
    file = path / "manifest.json"
    if not file.is_file():
        return {}
    return json.loads(file.read_text(encoding="utf-8"))


def list_batches() -> list[int]:
    if not QWEN_ROOT.exists():
        return []
    out = []
    for path in QWEN_ROOT.iterdir():
        if path.is_dir() and path.name.startswith("batch-"):
            try:
                out.append(int(path.name.split("-", 1)[1]))
            except ValueError:
                continue
    return sorted(out)


def next_batch() -> tuple[int, bool]:
    """Return `(batch id, resume)`. An unfinished batch is continued."""
    pending = []
    done = []
    for bid in list_batches():
        status = _manifest(QWEN_ROOT / f"batch-{bid:04d}").get("status")
        if status == "published":
            done.append(bid)
        else:
            pending.append(bid)
    if pending:
        return pending[0], True
    return (done[-1] + 1) if done else 1, False


def write_heartbeat(doc: dict) -> None:
    HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
    tmp = HEARTBEAT.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(HEARTBEAT)


def run_forever(n: int, concurrency: int, variety: int, sleep_s: int) -> None:
    from data import episode_gen as EG

    QWEN_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[qwen] start n={n} concurrency={concurrency} variety={variety}",
          flush=True)
    while True:
        bid, resume = next_batch()
        out = QWEN_ROOT / f"batch-{bid:04d}"
        print(f"[qwen] batch {bid} resume={resume}", flush=True)
        try:
            stats = EG.run(
                n=n, seed=20_000 + bid, prose="qwen", teacher="qwen",
                out_dir=str(out), concurrency=concurrency, batch=1,
                variety=variety, resume=resume,
                command=(f"python3 -m data.qwen_stream run --n {n} "
                         f"--concurrency {concurrency} --variety {variety}"),
            )
            action = "qwen"
            error = ""
        except Exception as exc:
            stats = {}
            action = "error"
            error = f"{type(exc).__name__}: {exc}"
            print(f"[qwen] {error}", flush=True)
        write_heartbeat({
            "measured_utc": utcnow(),
            "action": action,
            "batch": bid,
            "n": n,
            "concurrency": concurrency,
            "model": os.environ.get("QWEN_MODEL", EG.MODEL),
            "n_episodes": stats.get("n_episodes"),
            "n_rejects": stats.get("n_rejects"),
            "error": error,
        })
        time.sleep(sleep_s if action == "qwen" else max(sleep_s, 30))


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m data.qwen_stream")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    run = sub.add_parser("run")
    run.add_argument("--forever", action="store_true")
    run.add_argument("--n", type=int, default=200)
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--variety", type=int, default=16)
    run.add_argument("--sleep", type=int, default=5)
    args = parser.parse_args(argv)
    if args.cmd == "status":
        bid, resume = next_batch()
        print(json.dumps({
            "measured_utc": utcnow(),
            "next_batch": bid,
            "resume": resume,
            "batches": list_batches(),
        }, indent=2))
        return 0
    if not args.forever:
        print("refusing: pass --forever", flush=True)
        return 2
    run_forever(args.n, args.concurrency, args.variety, args.sleep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
