"""Development loop: keeps the training loop healthy and changes its recipe
when the recipe stops paying.

Every few minutes it checks the services the autoloop depends on and
restarts the ones launchd supervises when they stop answering. After every
new decision in ``artifacts/autoloop/history.jsonl`` it scores the arm that
produced it and writes the arm for the next cycle to ``strategy.json``; the
autoloop reads that file only when a cycle starts.

Arm choice is a bandit over measured results: every arm runs once, then the
one with the best upper bound on its gain over the champion of its time.
The gain is measured on battery_dev, the same battery the gate uses, so a
plateau report recommends a sealed-battery check before any release claim.
The loop never touches the sealed battery, the gate rule or the data.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOOP = ROOT / "artifacts" / "autoloop"
DEV = ROOT / "artifacts" / "devloop"
STATE = DEV / "state.json"
REPORTS = DEV / "reports"
HISTORY = LOOP / "history.jsonl"
STRATEGY_FILE = LOOP / "strategy.json"
LOOP_STATE = LOOP / "state.json"
STOP = DEV / "STOP"

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DASHBOARD = "http://127.0.0.1:8794/"
POLL_SECONDS = 600
HEARTBEAT_STALE_S = 15 * 60
DISK_MIN_GB = 40
REPORT_HOUR = 18
PLATEAU_CYCLES = 8
UCB_C = 0.01
LABELS = {
    "autoloop": "com.jevclone.autoloop",
    "ollama": "com.asimovia.ollama-qwen",
    "dashboard": "com.jevclone.dashboard",
}

BASE = {"steps": 200, "lr": 2e-5, "warmup": 100,
        "shares": {"qwen": 0.45, "public": 0.30, "rule": 0.25},
        "rule_weak_weight": 2, "families": []}
ARMS = {
    "base": {"why": "recipe of cycles 1 to 9"},
    "short-warmup": {"warmup": 20,
                     "why": "base spends half of its 200 steps warming up"},
    "gentle": {"lr": 1e-5, "warmup": 20,
               "why": "rejected candidates mostly lose ground on families "
                      "the champion already solves"},
    "priority-focus": {"families": ["priority_decision",
                                    "textual_inference_negation"],
                       "rule_weak_weight": 4,
                       "why": "priority_decision holds most of the errors"},
    "rules-heavy": {"shares": {"qwen": 0.35, "public": 0.20, "rule": 0.45},
                    "rule_weak_weight": 3,
                    "why": "rule gold is exact by construction"},
    "longer": {"steps": 400, "warmup": 40,
               "why": "200 steps may be too few to move a 4B model"},
}


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def read_history() -> list[dict]:
    try:
        lines = HISTORY.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


# -- strategy -----------------------------------------------------------------

def arm_strategy(name: str) -> dict:
    spec = {**BASE, **{k: v for k, v in ARMS[name].items() if k != "why"}}
    spec["arm"] = name
    return spec


def arm_results(history: list[dict]) -> dict[str, list[float]]:
    """Gain of each scored candidate over the champion it was measured
    against. Cycles from before the arms existed count as ``base``."""
    results: dict[str, list[float]] = {name: [] for name in ARMS}
    for row in history:
        cand = row.get("candidate") or {}
        before = row.get("champion_before") or {}
        if cand.get("mean") is None or before.get("mean") is None:
            continue
        arm = (row.get("strategy") or {}).get("arm") or "base"
        if arm in results:
            results[arm].append(cand["mean"] - before["mean"])
    return results


def choose_arm(results: dict[str, list[float]]) -> tuple[str, str]:
    untried = [name for name in ARMS if not results.get(name)]
    if untried:
        return untried[0], "not measured yet"
    total = sum(len(v) for v in results.values())

    def bound(name: str) -> float:
        gains = results[name]
        return (sum(gains) / len(gains)
                + UCB_C * math.sqrt(math.log(total) / len(gains)))

    best = max(ARMS, key=bound)
    gains = results[best]
    return best, (f"best upper bound {bound(best):+.4f} over "
                  f"{len(gains)} cycle(s), mean gain "
                  f"{sum(gains) / len(gains):+.4f}")


def cycles_since_promotion(history: list[dict]) -> int:
    count = 0
    for row in reversed(history):
        if row.get("promoted"):
            break
        if row.get("candidate"):
            count += 1
    return count


# -- health -------------------------------------------------------------------

def http_ok(url: str, timeout: int = 5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except (OSError, ValueError):
        return False


def pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (TypeError, ValueError, OSError):
        return False


def age_s(stamp: str | None) -> int | None:
    if not stamp:
        return None
    try:
        then = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc)
    except ValueError:
        return None
    return int(time.time() - then.timestamp())


def on_ac_power() -> bool | None:
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return "AC Power" in out


def kickstart(label: str) -> str:
    target = f"gui/{os.getuid()}/{label}"
    try:
        proc = subprocess.run(["launchctl", "kickstart", target],
                              capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"kickstart failed: {exc}"
    return "kickstarted" if proc.returncode == 0 else (
        f"kickstart exit {proc.returncode}: {proc.stderr.strip()[:120]}")


def health(repair: bool = True) -> dict:
    loop = read_json(LOOP_STATE)
    stopped = (LOOP / "STOP").exists()
    heartbeat = age_s(loop.get("updated_utc"))
    checks = {
        "autoloop": (loop.get("status") == "running"
                     and pid_alive(loop.get("pid"))
                     and heartbeat is not None
                     and heartbeat < HEARTBEAT_STALE_S),
        "ollama": http_ok(OLLAMA + "/api/tags"),
        "dashboard": http_ok(DASHBOARD),
    }
    free_gb = shutil.disk_usage(ROOT).free / 2**30
    power = on_ac_power()
    actions, alerts = [], []
    for name, ok in checks.items():
        if ok:
            continue
        if name == "autoloop" and stopped:
            alerts.append("autoloop stopped by its STOP file")
            continue
        if repair:
            actions.append(f"{name}: {kickstart(LABELS[name])}")
        else:
            alerts.append(f"{name} down")
    if free_gb < DISK_MIN_GB:
        alerts.append(f"disk {free_gb:.0f} GB free, under {DISK_MIN_GB} GB")
    if power is False:
        alerts.append("running on battery; a drained battery stops the loop")
    return {
        "at": utcnow(),
        "checks": checks,
        "heartbeat_age_s": heartbeat,
        "phase": loop.get("phase"),
        "cycle": loop.get("cycle"),
        "disk_free_gb": round(free_gb, 1),
        "ac_power": power,
        "actions": actions,
        "alerts": alerts,
    }


# -- report -------------------------------------------------------------------

def daily_report(history: list[dict], results: dict, chosen: str,
                 why: str, check: dict) -> str:
    now = time.time()
    recent = [r for r in history
              if (age_s(r.get("decided_utc") or r.get("at")) or 1e12) < 86400]
    scored = [r for r in recent if r.get("candidate")]
    failed = [r for r in recent if r.get("failed_phase")]
    promoted = [r for r in scored if r.get("promoted")]
    champion = read_json(LOOP / "champion" / "champion.json")
    summary = champion.get("summary") or {}
    lines = [
        f"# Daily report {time.strftime('%Y-%m-%d', time.localtime(now))}",
        "",
        f"Champion: cycle {champion.get('cycle')}, mean of 9 orders "
        f"{summary.get('mean')}, worst order {summary.get('min')}.",
        f"Last 24 h: {len(scored)} cycle(s) scored, {len(promoted)} "
        f"promoted, {len(failed)} failed.",
        f"Cycles since the last promotion: {cycles_since_promotion(history)}.",
        f"Next arm: {chosen} ({why}).",
        "",
        "| arm | cycles | mean gain | best gain |",
        "|---|---|---|---|",
    ]
    for name, gains in results.items():
        if gains:
            lines.append(f"| {name} | {len(gains)} | "
                         f"{sum(gains) / len(gains):+.4f} | "
                         f"{max(gains):+.4f} |")
        else:
            lines.append(f"| {name} | 0 | - | - |")
    lines += ["", "Health: " + ", ".join(
        f"{k} {'ok' if v else 'DOWN'}" for k, v in check["checks"].items())
        + f", disk {check['disk_free_gb']} GB free, "
        + ("AC power" if check["ac_power"] else "battery") + "."]
    for alert in check["alerts"]:
        lines.append(f"- alert: {alert}")
    if cycles_since_promotion(history) >= PLATEAU_CYCLES:
        lines += ["", "Plateau: every arm has had its turn without moving "
                  "the champion. The recipe space here is spent; the next "
                  "gain needs new data or a new idea from the research "
                  "review, and a sealed-battery check of the champion."]
    return "\n".join(lines) + "\n"


# -- loop ---------------------------------------------------------------------

def refresh_strategy() -> tuple[str, str, bool]:
    """Write the arm for the next cycle. The autoloop calls this at every
    cycle start, so the decision it just appended is already counted."""
    results = arm_results(read_history())
    chosen, why = choose_arm(results)
    changed = read_json(STRATEGY_FILE).get("arm") != chosen
    if changed:
        write_json(STRATEGY_FILE, {**arm_strategy(chosen), "why": why,
                                   "chosen_utc": utcnow()})
    return chosen, why, changed


def tick(state: dict, repair: bool = True) -> dict:
    chosen, why, changed = refresh_strategy()
    history = read_history()
    results = arm_results(history)
    if changed:
        state.setdefault("events", []).append(
            {"at": utcnow(), "event": f"next arm {chosen}: {why}"})
    check = health(repair)
    for action in check["actions"]:
        state.setdefault("events", []).append(
            {"at": utcnow(), "event": action})
    state["events"] = state.get("events", [])[-40:]
    since = cycles_since_promotion(history)
    today = time.strftime("%Y-%m-%d")
    if (time.localtime().tm_hour >= REPORT_HOUR
            and state.get("last_report") != today):
        REPORTS.mkdir(parents=True, exist_ok=True)
        (REPORTS / f"{today}.md").write_text(
            daily_report(history, results, chosen, why, check),
            encoding="utf-8")
        state["last_report"] = today
    state.update({
        "updated_utc": utcnow(),
        "pid": os.getpid(),
        "next_arm": chosen,
        "next_arm_why": why,
        "arms": {name: {"why": ARMS[name]["why"],
                        "cycles": len(gains),
                        "mean_gain": (sum(gains) / len(gains)
                                      if gains else None),
                        "best_gain": max(gains) if gains else None}
                 for name, gains in results.items()},
        "cycles_since_promotion": since,
        "plateau": since >= PLATEAU_CYCLES,
        "health": check,
    })
    write_json(STATE, state)
    return state


def main() -> int:
    once = "--once" in sys.argv
    state = read_json(STATE)
    while not STOP.exists():
        try:
            state = tick(state, repair=not once)
        except Exception as exc:
            state["last_error"] = f"{type(exc).__name__}: {exc}"
            write_json(STATE, state)
        if once:
            break
        time.sleep(POLL_SECONDS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
