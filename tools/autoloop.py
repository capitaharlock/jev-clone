"""Closed improvement loop for the pointer model, meant to run day and night.

One cycle, always in this order, each phase resumable after a crash:

    generate  Qwen writes episodes for the weakest battery families, inside
              a fixed time slot; chunks published before the cut are kept
    verify    a separate Qwen auditor checks them; disagreement is dropped
    mix       verified Qwen + public train rows + a sample of every rule batch
    train     a short LoRA run that starts from the champion adapter
    eval      battery_dev, written order plus shuffled orders
    decide    promote the candidate only when it beats the champion

The GPU runs one heavy job at a time: Qwen and the trainer do not fit in
memory together, so Qwen is unloaded before training starts. When Qwen is
down the cycle trains on the verified pool it already has. Only a STOP file
in ``artifacts/autoloop`` ends the process. The sealed battery is never an
input; ``battery_dev`` only scores and never enters a mix.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOOP = ROOT / "artifacts" / "autoloop"
CHAMPION = LOOP / "champion"
CYCLES = LOOP / "cycles"
QWEN_OUT = LOOP / "qwen"
STATE = LOOP / "state.json"
HISTORY = LOOP / "history.jsonl"
STOP = LOOP / "STOP"
MODELS_FILE = LOOP / "models.json"
PYTHON = ROOT / ".venv-train" / "bin" / "python"
SEED_RUN = ROOT / "artifacts" / "checkpoints" / "pointer" / "loop"
RULE_ROOT = ROOT / "artifacts" / "episodes-rule"
QWEN_ROOT = ROOT / "artifacts" / "episodes-qwen"
PUBLIC_MANIFEST = (ROOT / "artifacts" / "episodes-external" / "public-mix"
                   / "manifest.json")
OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen3.6:27b-q8_0")

PHASES = ("generate", "verify", "mix", "train", "eval", "decide")
FAMILIES = (
    "attribute_comparison",
    "description_classification",
    "extraction_paraphrase",
    "priority_decision",
    "textual_inference_negation",
)
RULE_FAMILIES = ("attribute_comparison", "priority_decision")
WEAK_BELOW = 0.99
GEN_N = int(os.environ.get("AUTOLOOP_GEN_N", "800"))
GEN_SECONDS = int(os.environ.get("AUTOLOOP_GEN_SECONDS", str(2 * 3600)))
GEN_CONCURRENCY = int(os.environ.get("AUTOLOOP_GEN_CONCURRENCY", "4"))
GEN_VARIETY = 16
TRAIN_STEPS = int(os.environ.get("AUTOLOOP_TRAIN_STEPS", "200"))
MIX_SIZE = int(os.environ.get("AUTOLOOP_MIX_SIZE", "12000"))
SHARES = {"qwen": 0.45, "public": 0.30, "rule": 0.25}
QWEN_REPEAT_MAX = 4
RULE_BATCHES_PER_MIX = 8
OPTION_PAD = 9
SUBSAMPLE_KS = (2, 3, 8)
ORDERS = 8
FAMILY_DROP = 0.02
MAX_ATTEMPTS = 3
POLL_SECONDS = 20
STALE_SECONDS = 30 * 60
STOP_GRACE_SECONDS = 120
TIMEOUTS = {
    "generate": GEN_SECONDS,
    "verify": 4 * 3600,
    "train": 10 * 3600,
    "eval": 2 * 3600,
}

_TERM = {"flag": False}


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
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def drop_torn_tail(path: Path) -> None:
    """A child killed mid-write can leave half a line at the end."""
    if not path.is_file():
        return
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    while lines:
        try:
            json.loads(lines[-1])
            break
        except ValueError:
            lines.pop()
    path.write_text("".join(lines), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# -- scoring and promotion ----------------------------------------------------

def summarize(report: dict) -> dict:
    """Written-order score, mean and floor over every order, per family."""
    orders = [o["forced"] for o in report.get("orders") or []]
    if not orders and report.get("forced") is not None:
        orders = [report["forced"]]
    return {
        "step": report.get("adapter_step"),
        "written": report.get("forced"),
        "mean": round(sum(orders) / len(orders), 6) if orders else None,
        "min": min(orders) if orders else None,
        "n_orders": len(orders),
        "by_family": {
            name: cell.get("forced")
            for name, cell in (report.get("by_family") or {}).items()
            if isinstance(cell, dict)
        },
    }


def promotion(champion: dict, candidate: dict) -> tuple[bool, list[str]]:
    """The candidate replaces the champion only if no reason blocks it."""
    reasons = []
    if candidate.get("mean") is None:
        return False, ["candidate has no battery score"]
    if champion.get("mean") is None:
        return True, []
    if candidate["mean"] <= champion["mean"]:
        reasons.append(
            f"mean {candidate['mean']:.4f} <= champion {champion['mean']:.4f}")
    if candidate["min"] < champion["min"]:
        reasons.append(
            f"worst order {candidate['min']:.4f} < champion "
            f"{champion['min']:.4f}")
    for name, before in (champion.get("by_family") or {}).items():
        after = (candidate.get("by_family") or {}).get(name)
        if after is not None and before is not None \
                and after < before - FAMILY_DROP:
            reasons.append(f"{name} {after:.4f} < {before:.4f} - {FAMILY_DROP}")
    return not reasons, reasons


def weak_families(summary: dict, cycle: int) -> list[str]:
    """Families under the bar, worst first, plus one strong family in turn
    so the generated pool keeps covering what already works."""
    scores = summary.get("by_family") or {}
    weak = sorted((f for f in FAMILIES
                   if scores.get(f) is not None and scores[f] < WEAK_BELOW),
                  key=lambda f: scores[f])
    strong = [f for f in FAMILIES if f not in weak]
    if not weak:
        return list(FAMILIES)
    if strong:
        weak.append(strong[cycle % len(strong)])
    return weak


# -- mix ----------------------------------------------------------------------

def positives_of(ep: dict) -> list[str]:
    if ep.get("answer") is not None:
        return [str(ep["answer"])]
    if ep.get("acceptable_answers"):
        return [str(a) for a in ep["acceptable_answers"]]
    return [str(ep["preference"]["order"][0])]


def subsample_options(ep: dict, k: int, rng: random.Random) -> dict:
    """Keep every gold option and enough distractors to reach ``k``.

    Wide label spaces (60 intents, 28 emotions) cannot enter a 9-slot
    pointer head as they are. Dropping distractors keeps the gold right.
    """
    cands = ep["candidates"]
    if len(cands) <= k:
        return ep
    gold = set(positives_of(ep))
    golds = [c for c in cands if c["id"] in gold][:max(1, k - 1)]
    keep_gold = {c["id"] for c in golds}
    others = [c for c in cands if c["id"] not in gold]
    picked = keep_gold | {c["id"] for c in rng.sample(others, k - len(golds))}
    out = dict(ep)
    out["candidates"] = [c for c in cands if c["id"] in picked]
    if ep.get("acceptable_answers"):
        out["acceptable_answers"] = [a for a in ep["acceptable_answers"]
                                     if str(a) in picked]
    if ep.get("preference"):
        pref = dict(ep["preference"])
        pref["order"] = [i for i in pref["order"] if str(i) in picked]
        out["preference"] = pref
    out["id"] = f"{ep['id']}-k{k}"
    out["options_subsampled_from"] = len(cands)
    return out


def qwen_pool() -> list[dict]:
    """Every verified Qwen episode on disk, older runs included."""
    rows = []
    for path in sorted(QWEN_ROOT.glob("**/verified.jsonl")):
        rows.extend(read_jsonl(path))
    for path in sorted(QWEN_OUT.glob("cycle-*/verified.jsonl")):
        rows.extend(read_jsonl(path))
    seen, unique = set(), []
    for ep in rows:
        if ep["id"] not in seen:
            seen.add(ep["id"])
            unique.append(ep)
    return unique


def public_sample(n: int, rng: random.Random) -> list[dict]:
    manifest = read_json(PUBLIC_MANIFEST)
    members = [m for m in manifest.get("members") or []
               if m.get("split") == "train"]
    total = sum(int(m["n_under_cap"]) for m in members)
    if not total or n <= 0:
        return []
    rows = []
    for member in members:
        quota = round(n * int(member["n_under_cap"]) / total)
        pool = read_jsonl(ROOT / member["path"])
        for ep in rng.sample(pool, min(quota, len(pool))):
            k = len(ep["candidates"])
            if k > OPTION_PAD:
                ep = subsample_options(ep, rng.choice(SUBSAMPLE_KS), rng)
            rows.append(ep)
    return rows


def rule_sample(n: int, weak: list[str], rng: random.Random) -> list[dict]:
    """Draw from batches across the whole stream, not only the first ones."""
    batches = sorted(RULE_ROOT.glob("batch-*/episodes.jsonl"))
    if not batches or n <= 0:
        return []
    picked = rng.sample(batches, min(RULE_BATCHES_PER_MIX, len(batches)))
    weight = {f: (2 if f in weak else 1) for f in RULE_FAMILIES}
    total = sum(weight.values())
    want = {f: round(n * weight[f] / total) for f in RULE_FAMILIES}
    by_family: dict[str, list[dict]] = {f: [] for f in RULE_FAMILIES}
    for path in picked:
        for ep in read_jsonl(path):
            if ep.get("family") in by_family:
                by_family[ep["family"]].append(ep)
    rows = []
    for family, pool in by_family.items():
        rows.extend(rng.sample(pool, min(want[family], len(pool))))
    return rows


def leak_free(episodes: list[dict]) -> tuple[list[dict], int]:
    from data import episode_bridge as EB

    index = EB.TrigramIndex(EB.blocked_texts())
    verdict: dict[str, bool] = {}
    kept = []
    for ep in episodes:
        state = ep["state"]
        if state not in verdict:
            verdict[state] = index.scan(state) is None
        if verdict[state]:
            kept.append(ep)
    return kept, len(episodes) - len(kept)


def build_mix(cycle: int, weak: list[str], out: Path) -> dict:
    rng = random.Random(1_000_003 * cycle + 17)
    qwen = qwen_pool()
    qwen_target = int(MIX_SIZE * SHARES["qwen"])
    repeat = (min(QWEN_REPEAT_MAX, max(1, math.ceil(qwen_target / len(qwen))))
              if qwen else 0)
    qwen_rows = [ep for _ in range(repeat) for ep in qwen][:qwen_target]
    short = qwen_target - len(qwen_rows)
    public_n = int(MIX_SIZE * SHARES["public"]) + short // 2
    public_rows = public_sample(public_n, rng)
    rule_n = MIX_SIZE - len(qwen_rows) - len(public_rows)
    rule_rows = rule_sample(rule_n, weak, rng)
    rows = qwen_rows + public_rows + rule_rows
    rows = [ep for ep in rows if len(ep["candidates"]) <= OPTION_PAD]
    rows, leaks = leak_free(rows)
    rng.shuffle(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for ep in rows:
            handle.write(json.dumps(ep, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    return {
        "cycle": cycle,
        "path": str(out.relative_to(ROOT)),
        "sha256": digest,
        "n": len(rows),
        "by_source": {"qwen": len(qwen_rows), "public": len(public_rows),
                      "rule": len(rule_rows)},
        "qwen_pool": len(qwen),
        "qwen_repeat": repeat,
        "leak_removed": leaks,
        "weak_families": weak,
        "by_family": dict(Counter(ep["family"] for ep in rows)),
        "by_k": dict(sorted(Counter(len(ep["candidates"])
                                    for ep in rows).items())),
        "built_utc": utcnow(),
    }


# -- child processes ----------------------------------------------------------

def qwen_models() -> dict:
    """Generator and verifier, overridable per cycle through models.json.

    A verifier from another model than the generator is the closer thing
    to an independent check that this machine can run.
    """
    chosen = {"generate": QWEN_MODEL, "verify": QWEN_MODEL}
    for role, model in read_json(MODELS_FILE).items():
        if role in chosen and model:
            chosen[role] = model
    return chosen


def ollama_call(path: str, payload: dict | None, timeout: int) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        f"{OLLAMA}{path}", data=data,
        headers={"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read() or b"{}")


def qwen_answers(model: str) -> bool:
    try:
        reply = ollama_call("/api/chat", {
            "model": model, "stream": False, "think": False,
            "messages": [{"role": "user",
                          "content": "Reply with exactly: ok"}],
            "options": {"num_predict": 8},
        }, timeout=600)
    except (OSError, ValueError):
        return False
    return "ok" in ((reply.get("message") or {}).get("content") or "").lower()


def child_env(model: str = QWEN_MODEL) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["QWEN_MODEL"] = model
    env["EPISODE_GEN_BULK"] = "1"
    env.setdefault("TOKENIZERS_PARALLELISM", "true")
    return env


def _mtime(path: Path | None) -> float | None:
    if path is None:
        return None
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def stop_child(proc: subprocess.Popen, reason: str) -> None:
    print(f"{reason}; stopping pid={proc.pid}", flush=True)
    proc.terminate()
    try:
        proc.wait(timeout=STOP_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def supervise(cmd: list[str], phase: str, log: Path,
              progress: Path | None, state: dict,
              model: str = QWEN_MODEL) -> str:
    """``ok``, ``stop``, ``stale``, ``timeout`` or ``exit <code>``."""
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{phase}] " + " ".join(cmd), flush=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n# {utcnow()} model={model} {' '.join(cmd)}\n")
        handle.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, env=child_env(model),
                                stdout=handle, stderr=subprocess.STDOUT)
        started = last_progress = time.time()
        seen = _mtime(progress)
        while proc.poll() is None:
            if STOP.exists() or _TERM["flag"]:
                stop_child(proc, "stop requested")
                return "stop"
            now = time.time()
            current = _mtime(progress)
            if current != seen:
                seen, last_progress = current, now
            if progress is not None and now - last_progress >= STALE_SECONDS:
                stop_child(proc, f"no progress for {int(now - last_progress)}s")
                return "stale"
            if now - started >= TIMEOUTS[phase]:
                stop_child(proc, f"{phase} over {TIMEOUTS[phase]}s")
                return "timeout"
            heartbeat(state, child_pid=proc.pid)
            time.sleep(POLL_SECONDS)
    return "ok" if proc.returncode == 0 else f"exit {proc.returncode}"


def unload_qwen() -> None:
    try:
        loaded = ollama_call("/api/ps", None, timeout=30).get("models") or []
        for entry in loaded:
            ollama_call("/api/generate",
                        {"model": entry["name"], "keep_alive": 0}, timeout=60)
    except (OSError, ValueError) as exc:
        print(f"qwen unload skipped: {exc}", flush=True)


def heartbeat(state: dict, **extra) -> None:
    state.update(extra)
    state["pid"] = os.getpid()
    state["updated_utc"] = utcnow()
    write_json(STATE, state)


# -- phases -------------------------------------------------------------------

def bootstrap_champion() -> dict:
    """The first champion is the adapter the long rule run left behind."""
    meta = read_json(CHAMPION / "champion.json")
    if meta and (CHAMPION / "adapters.safetensors").is_file():
        return meta
    CHAMPION.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SEED_RUN / "adapters.safetensors",
                 CHAMPION / "adapters.safetensors")
    report = read_json(SEED_RUN / "battery_regression.json")
    write_json(CHAMPION / "battery_regression.json", report)
    meta = {"source": str(SEED_RUN.relative_to(ROOT)), "cycle": 0,
            "summary": summarize(report), "promoted_utc": utcnow()}
    write_json(CHAMPION / "champion.json", meta)
    return meta


def cycle_dir(cycle: int) -> Path:
    return CYCLES / f"cycle-{cycle:04d}"


def phase_generate(cycle: int, state: dict) -> str:
    out = QWEN_OUT / f"cycle-{cycle:04d}"
    manifest = read_json(out / "manifest.json")
    if manifest.get("status") == "published":
        return "ok"
    cmd = [str(PYTHON), "-u", "-m", "data.episode_gen", "run",
           "--n", str(GEN_N), "--seed", str(30_000 + cycle),
           "--prose", "qwen", "--teacher", "qwen",
           "--concurrency", str(GEN_CONCURRENCY),
           "--variety", str(GEN_VARIETY),
           "--families", ",".join(state["weak_families"]),
           "--group-prefix", f"al{cycle:04d}-",
           "--out", str(out)]
    if (out / "episodes.jsonl").is_file():
        cmd.append("--resume")
    model = qwen_models()["generate"]
    if model != QWEN_MODEL and not qwen_answers(model):
        print(f"[generate] {model} does not answer; using {QWEN_MODEL}",
              flush=True)
        model = QWEN_MODEL
    state["generate_model"] = model
    log = LOOP / "logs" / "generate.log"
    outcome = supervise(cmd, "generate", log, log, state, model)
    for name in ("episodes.jsonl", "rejects.jsonl"):
        drop_torn_tail(out / name)
    published = len(read_jsonl(out / "episodes.jsonl"))
    state["generated"] = published
    if outcome == "timeout" and published:
        # The slot is a GPU time budget, not a count: what was published
        # in 100-row chunks before the cut is complete and goes on.
        return "ok"
    return outcome


def phase_verify(cycle: int, state: dict) -> str:
    out = QWEN_OUT / f"cycle-{cycle:04d}"
    if not (out / "episodes.jsonl").is_file():
        return "ok"
    if read_json(out / "verify_manifest.json").get("status") == "published":
        return "ok"
    cmd = [str(PYTHON), "-u", "-m", "data.episode_verify", "run",
           "--dir", str(out), "--concurrency", str(GEN_CONCURRENCY),
           "--job-id", "autoloop"]
    model = qwen_models()["verify"]
    state["verify_model"] = model
    log = LOOP / "logs" / "verify.log"
    return supervise(cmd, "verify", log, log, state, model)


def phase_mix(cycle: int, state: dict) -> str:
    run = cycle_dir(cycle)
    manifest = build_mix(cycle, state["weak_families"], run / "mix.jsonl")
    write_json(run / "mix_manifest.json", manifest)
    print(f"[mix] {manifest['n']} rows {manifest['by_source']}", flush=True)
    return "ok" if manifest["n"] else "exit empty-mix"


def phase_train(cycle: int, state: dict) -> str:
    unload_qwen()
    run = cycle_dir(cycle)
    if not (run / "checkpoint.json").is_file():
        shutil.copy2(CHAMPION / "adapters.safetensors",
                     run / "adapters.safetensors")
    cmd = [str(PYTHON), "-u", "-m", "training.python.pointer_finetune",
           "train", "--out", str(run),
           "--episodes", str(run / "mix.jsonl"),
           "--steps", str(TRAIN_STEPS),
           "--init", f"champion-cycle-{state['champion_cycle']:04d}"]
    return supervise(cmd, "train", LOOP / "logs" / "train.log",
                     run / "state.json", state)


def phase_eval(cycle: int, state: dict) -> str:
    run = cycle_dir(cycle)
    cmd = [str(PYTHON), "-u", "-m", "training.python.pointer_finetune",
           "eval-battery", "--out", str(run), "--orders", str(ORDERS)]
    return supervise(cmd, "eval", LOOP / "logs" / "eval.log", None, state)


def phase_decide(cycle: int, state: dict) -> str:
    run = cycle_dir(cycle)
    champion = read_json(CHAMPION / "champion.json")
    candidate = summarize(read_json(run / "battery_regression.json"))
    promote, reasons = promotion(champion.get("summary") or {}, candidate)
    decision = {
        "cycle": cycle,
        "decided_utc": utcnow(),
        "promoted": promote,
        "reasons": reasons,
        "candidate": candidate,
        "champion_before": champion.get("summary"),
        "champion_cycle_before": champion.get("cycle"),
        "mix": read_json(run / "mix_manifest.json").get("by_source"),
        "weak_families": state["weak_families"],
    }
    if promote:
        archive = CHAMPION / "previous" / f"cycle-{champion.get('cycle', 0):04d}"
        archive.mkdir(parents=True, exist_ok=True)
        for name in ("adapters.safetensors", "battery_regression.json",
                     "champion.json"):
            if (CHAMPION / name).is_file():
                shutil.copy2(CHAMPION / name, archive / name)
        shutil.copy2(run / "adapters.safetensors",
                     CHAMPION / "adapters.safetensors")
        shutil.copy2(run / "battery_regression.json",
                     CHAMPION / "battery_regression.json")
        write_json(CHAMPION / "champion.json", {
            "source": str(run.relative_to(ROOT)), "cycle": cycle,
            "summary": candidate, "promoted_utc": utcnow()})
        state["champion_cycle"] = cycle
        state["streak_no_promote"] = 0
    else:
        state["streak_no_promote"] = int(state.get("streak_no_promote") or 0) + 1
    write_json(run / "decision.json", decision)
    append_jsonl(HISTORY, decision)
    optimizer = run / "optimizer.safetensors"
    if optimizer.is_file():
        optimizer.unlink()
    verdict = "promoted" if promote else "kept champion: " + "; ".join(reasons)
    print(f"[decide] cycle {cycle} {verdict}", flush=True)
    state["last_decision"] = verdict
    return "ok"


RUNNERS = {
    "generate": phase_generate,
    "verify": phase_verify,
    "mix": phase_mix,
    "train": phase_train,
    "eval": phase_eval,
    "decide": phase_decide,
}


def start_cycle(state: dict, cycle: int) -> None:
    champion = read_json(CHAMPION / "champion.json")
    state.update({
        "cycle": cycle,
        "phase": PHASES[0],
        "attempts": 0,
        "run_dir": str(cycle_dir(cycle).relative_to(ROOT)),
        "weak_families": weak_families(champion.get("summary") or {}, cycle),
        "champion": champion.get("summary"),
        "cycle_started_utc": utcnow(),
    })


def on_term(signum, frame) -> None:
    _TERM["flag"] = True


def main() -> int:
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    LOOP.mkdir(parents=True, exist_ok=True)
    champion = bootstrap_champion()
    state = read_json(STATE)
    state.setdefault("champion_cycle", champion.get("cycle", 0))
    state["status"] = "running"
    state["started_utc"] = utcnow()
    if state.get("phase") not in PHASES:
        start_cycle(state, int(state.get("cycle") or 0) + 1)
    while True:
        if STOP.exists() or _TERM["flag"]:
            heartbeat(state, status="stopped", child_pid=None)
            return 0
        phase, cycle = state["phase"], int(state["cycle"])
        heartbeat(state, status="running", phase_started_utc=utcnow())
        try:
            outcome = RUNNERS[phase](cycle, state)
        except Exception as exc:
            outcome = f"error {type(exc).__name__}: {exc}"
        if outcome == "stop":
            heartbeat(state, status="stopped", child_pid=None)
            return 0
        state.setdefault("phase_log", [])
        state["phase_log"] = (state["phase_log"] + [
            {"cycle": cycle, "phase": phase, "outcome": outcome,
             "at": utcnow()}])[-30:]
        if outcome == "ok":
            index = PHASES.index(phase)
            if index + 1 < len(PHASES):
                state["phase"], state["attempts"] = PHASES[index + 1], 0
            else:
                start_cycle(state, cycle + 1)
            continue
        state["attempts"] = int(state.get("attempts") or 0) + 1
        print(f"[{phase}] {outcome} (attempt {state['attempts']})", flush=True)
        if state["attempts"] < MAX_ATTEMPTS:
            heartbeat(state)
            time.sleep(60)
            continue
        if phase in ("generate", "verify"):
            state["phase"], state["attempts"] = "mix", 0
            print(f"[{phase}] giving up; training on the existing pool",
                  flush=True)
        else:
            append_jsonl(HISTORY, {"cycle": cycle, "failed_phase": phase,
                                   "outcome": outcome, "at": utcnow()})
            start_cycle(state, cycle + 1)


if __name__ == "__main__":
    sys.exit(main())
