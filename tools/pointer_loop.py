"""Keep the pointer LoRA trainer running and restart stalled children.

Only a STOP file next to the run ends the loop. A child that exits or
publishes no state for 30 minutes is restarted from its last checkpoint.
The dev battery and the sealed cut are not arguments of this process.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "checkpoints" / "pointer" / "loop"
PYTHON = ROOT / ".venv-train" / "bin" / "python"
FLAG = ROOT / "artifacts" / "loop" / "TRAINING"
STATE = OUT / "state.json"
POLL_SECONDS = 20
STALE_SECONDS = 30 * 60
STOP_GRACE_SECONDS = 120
TRAIN_EPISODE_LIMIT = 660_000


def stop_child(proc: subprocess.Popen, reason: str) -> None:
    print(f"{reason}; stopping trainer pid={proc.pid}", flush=True)
    proc.terminate()
    try:
        proc.wait(timeout=STOP_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        print(f"trainer pid={proc.pid} ignored SIGTERM; killing", flush=True)
        proc.kill()
        proc.wait()


def main() -> None:
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    OUT.mkdir(parents=True, exist_ok=True)
    FLAG.parent.mkdir(parents=True, exist_ok=True)
    FLAG.write_text("pointer\n", encoding="utf-8")
    cmd = [
        str(PYTHON), "-u", "-m", "training.python.pointer_finetune",
        "train", "--live", "--out", str(OUT),
        "--limit", str(TRAIN_EPISODE_LIMIT),
    ]
    while True:
        if (OUT / "STOP").exists():
            print("STOP file present", flush=True)
            return
        print("start " + " ".join(cmd), flush=True)
        proc = subprocess.Popen(cmd, cwd=ROOT)
        last_state_mtime = STATE.stat().st_mtime if STATE.exists() else None
        last_progress = time.time()
        while proc.poll() is None:
            if (OUT / "STOP").exists():
                stop_child(proc, "STOP file present")
                return
            state_mtime = STATE.stat().st_mtime if STATE.exists() else None
            if state_mtime is not None and state_mtime != last_state_mtime:
                last_state_mtime = state_mtime
                last_progress = time.time()
            stale_for = time.time() - last_progress
            if stale_for >= STALE_SECONDS:
                stop_child(
                    proc,
                    f"no state progress for {int(stale_for)} seconds",
                )
                break
            time.sleep(POLL_SECONDS)
        code = proc.returncode
        print(f"trainer exit {code}", flush=True)
        if (OUT / "STOP").exists():
            return
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
