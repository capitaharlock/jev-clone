"""Detach dashboard, producer and loop from this shell (24 h)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv-train" / "bin" / "python"
LOG = ROOT / "artifacts" / "logs" / "24h"
ENV = os.environ.copy()
ENV["PYTHONPATH"] = str(ROOT)
ENV.setdefault("TOKENIZERS_PARALLELISM", "false")
ENV.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
ENV.setdefault("QWEN_MODEL", "qwen3.6:27b-q8_0")
ENV.setdefault("QWEN_VOLUME_MODEL", ENV["QWEN_MODEL"])
ENV.setdefault("QWEN_STREAM", "1")
ENV.setdefault("EPISODE_GEN_BULK", "1")
ENV.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
ENV.setdefault("HF_HUB_DISABLE_XET", "1")


def spawn(name: str, argv: list[str]) -> int:
    LOG.mkdir(parents=True, exist_ok=True)
    logf = open(LOG / f"{name}.log", "ab")
    proc = subprocess.Popen(
        argv, cwd=str(ROOT), env=ENV,
        stdout=logf, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    (LOG / f"{name}.pid").write_text(str(proc.pid) + "\n")
    print(f"{name} pid {proc.pid} log {LOG / (name + '.log')}")
    return proc.pid


def main() -> int:
    if not PY.exists():
        print(f"missing {PY}", file=sys.stderr)
        return 1
    spawn("dashboard", [str(PY), str(ROOT / "tools" / "training_monitor.py"),
                        "--interval", "60", "--tick", "20", "--port", "8794"])
    spawn("stream", [str(PY), "-u", "-m", "data.stream", "run", "--forever",
                     "--rule-batch", "5000", "--qwen-batch", "10",
                     "--variety", "16", "--sleep", "180", "--max-batches", "48"])
    spawn("loop", [str(PY), "-u", "-m", "training.python.loop", "run",
                   "--forever", "--device", "mps", "--sleep", "20"])
    print("dashboard http://127.0.0.1:8794/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
