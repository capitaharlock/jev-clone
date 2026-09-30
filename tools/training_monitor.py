#!/usr/bin/env python3
"""Live training monitor: terminal totals + realtime dashboard, endless loop.

Stdlib only. Each cycle:
  1. scans artifacts/data-prefetch/manifest.json + file sizes (pending volume)
  2. scans artifacts/runs/*/metrics.json (trained volume, per-task accuracy)
  3. estimates baseline params (TF-IDF 50k feats x classes per task)
  4. checks live processes (ollama/qwen, qwen_convert.py)
  5. optionally retrains (NON-blocking subprocess with .venv-train python)
  6. writes artifacts/runs/training-monitor/state.json + dashboard index.html
     on EVERY heartbeat (default 20 s), so terminal + dashboard always move
  7. prints a structured totals block to stdout + streams [train] log lines

Usage (from repo root):
  python3 tools/training_monitor.py --once                  # one pass, print + state
  python3 tools/training_monitor.py --interval 120           # endless terminal loop
  python3 tools/training_monitor.py --interval 120 --train  # endless + retrain on new data
  python3 tools/training_monitor.py --interval 120 --train --port 8765  # + dashboard

Dashboard: http://127.0.0.1:8765/ (auto-refresh every 15 s)
Unattended: nohup python3 tools/training_monitor.py --interval 300 --train --port 8765 \\
              >> artifacts/logs/monitor/monitor.log 2>&1 &
"""
import argparse
import datetime as dt
import functools
import http.server
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREFETCH = ROOT / "artifacts" / "data-prefetch"
RUNS = ROOT / "artifacts" / "runs"
STATE_DIR = RUNS / "training-monitor"
LOG_DIR = ROOT / "artifacts" / "logs" / "monitor"
FEATS = 50000  # TfidfVectorizer max_features in data/train_baseline.py
START_ACC = 0.6225  # MiniLM without tune, T-ce-finetune eval-none
CHANCE_ACC = 0.3375  # weighted chance for battery_dev K=2/3/8
GOAL_ACC = 0.99
WINDOW_S = 24 * 3600
_GPU_HIST = deque(maxlen=120)
_CPU_HIST = deque(maxlen=120)
_HIST_CACHE = {"sig": None, "data": None}
_RE_FORCED = re.compile(r"forced\s+[0-9.]+\s*->\s*([0-9.]+)")
_RE_BATCH = re.compile(r"batch-(\d+)")
_RE_GPU_UTIL = re.compile(
    r'"(?:Device|Renderer|Tiler) Utili' + r'zation %"\s*=\s*(\d+)')
_RE_GPU_MEM = re.compile(r'"In use system memory"\s*=\s*(\d+)')

# #T-halt-contam: synth-loop is quarantined (contaminated, out of the
# headline); logiqa/reclor are eval-only benchmarks, never train.
QUARANTINE = PREFETCH / "quarantine" / "synth-loop-20260921.jsonl"
CONTAMINATED_TASKS = {"synth-loop"}
EVAL_ONLY_TASKS = {"logiqa", "reclor"}

# #T-split-domain: every number this dashboard published for a dataset whose
# split was assigned by ROW INDEX (i % 10) is invalid as of this date; the
# same template sat in train and in test, so the accuracy measured leakage.
# The replacement is eval/splits.py (group split by skeleton+domain+idioma);
# the evidence is artifacts/gates/T-split-domain/.
INDEX_SPLIT_INVALID_ON = "2026-09-21"
INDEX_SPLIT_TASKS = {"synth-loop", "email-triage"}
INDEX_SPLIT_REASON = ("row-index split (i % 10): the same template appeared "
                      "in both training and test")
SPLIT_EVIDENCE = "artifacts/gates/T-split-domain/"

# #T-unseen-labels: the primary metric. Accuracy + ECE on labels never seen
# in training, always published as a seen/unseen PAIR. This is the headline
# of the dashboard; everything else on this page is context.
TASK_UNSEEN = "T-unseen-labels"


def train_python():
    """Venv first: system python has no sklearn, train would crash instantly."""
    venv = ROOT / ".venv-train" / "bin" / "python"
    if venv.exists():
        return str(venv)
    print("AVISO: .venv-train ausente, uso sys.executable "
          "(probable ModuleNotFoundError sklearn)", flush=True)
    return sys.executable


def utcnow():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0


def scan_datasets():
    """Pending volume: manifest rows/examples + bytes on disk."""
    sets, total_ex, total_bytes = [], 0, 0
    man = PREFETCH / "manifest.json"
    rows = {}
    if man.exists():
        try:
            m = json.loads(man.read_text())
            for j in m.get("jobs", []):
                rows[j.get("job")] = j
        except Exception as e:
            rows = {"_manifest_error": str(e)}
    for f in sorted(PREFETCH.glob("*.jsonl")):
        if f.name.endswith(".raw.jsonl"):
            continue
        st = f.stat()
        total_bytes += st.st_size
        info = rows.get(f.stem, {}) if isinstance(rows, dict) else {}
        ex = info.get("examples") or info.get("rows") or 0
        total_ex += ex or 0
        sets.append({
            "name": f.stem, "file": f.name,
            "bytes": st.st_size, "size": human(st.st_size),
            "examples": ex, "status": info.get("status", "?"),
            "mtime": dt.datetime.fromtimestamp(st.st_mtime,
                     dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
    if QUARANTINE.exists():
        # Contaminated row: shown, never counted (examples=0 keeps it out of
        # the headline totals). No metric paints green off this corpus again.
        st = QUARANTINE.stat()
        sets.append({
            "name": "synth-loop", "file": "quarantine/synth-loop-20260921.jsonl",
            "bytes": st.st_size, "size": human(st.st_size),
            "examples": 0, "frozen_rows": quarantine_rows(),
            "status": "CONTAMINADA · cuarentena #T-halt-contam",
            "mtime": dt.datetime.fromtimestamp(st.st_mtime,
                     dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
    # extra producers (qwen synth, prog gold) — volume only
    for extra in ("data-qwen", "prog_gold", "data-raw"):
        p = ROOT / "artifacts" / extra
        if p.exists():
            b = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            total_bytes += b
            sets.append({"name": f"[{extra}]", "file": extra + "/",
                         "bytes": b, "size": human(b), "examples": 0,
                         "status": "producer", "mtime": ""})
    return sets, total_ex, total_bytes


def scan_runs():
    """Trained volume: latest run with metrics.json wins."""
    best = None
    for m in sorted(RUNS.glob("*/metrics.json")):
        if "training-monitor" in str(m):
            continue
        try:
            d = json.loads(m.read_text())
            ts = m.parent.name
            if best is None or ts > best[0]:
                best = (ts, d)
        except Exception:
            continue
    if not best:
        return None, 0, []
    ts, d = best
    jobs = d.get("jobs", [])
    trained = sum(j.get("n_train", 0) for j in jobs)
    return ts, trained, jobs


def scan_unseen_gate():
    """#T-unseen-labels: the seen/unseen table that OWNS the headline.

    `artifacts/gates/T-unseen-labels/gate.json` is the only artifact in
    this repo that answers the product's actual promise — accuracy and ECE
    on labels whose TEXT never appeared in training — and it answers it in
    PAIRS, because an unseen number without its seen twin cannot tell
    generalisation from an easy cut. Absent, this panel says so; it never
    promotes a seen-label number to the headline.
    """
    gp = ROOT / "artifacts" / "gates" / TASK_UNSEEN / "gate.json"
    if not gp.exists():
        return None
    try:
        gate = json.loads(gp.read_text())
    except Exception:
        return None
    return {
        "pass": gate.get("pass"),
        "failed": gate.get("failed_criteria") or [],
        "verdict": gate.get("verdict"),
        "model_version": gate.get("model_version"),
        "checkpoint": gate.get("checkpoint"),
        "generated_utc": gate.get("generated_utc"),
        "headline": gate.get("headline") or {},
        "table": gate.get("table") or {},
        "eval_only": gate.get("eval_only_results") or {},
        "criteria": {k: bool(v.get("pass"))
                     for k, v in (gate.get("criteria") or {}).items()},
        "cross_lingual_blocker": ((gate.get("criteria") or {})
                                  .get("cross_lingual_holdout") or {})
        .get("blocker"),
        "calibration": gate.get("calibration") or {},
        "n_scored": gate.get("n_scored"),
        "path": str(gp.relative_to(ROOT)),
        "age_s": max(0.0, time.time() - gp.stat().st_mtime),
    }


def scan_decision_run():
    """#T-train-real: the listwise run that OWNS the headline.

    Reads `artifacts/runs/<run_id>/metrics.jsonl` (one json object per
    step / eval / stage) and returns the last step, the last stage and the
    250 k -> 1 M scale curve. The primary metric of #I-honest-eval is the
    UNSEEN-label accuracy against its own chance rate — row accuracy over
    a label space the model was fitted on is not a capability number and
    is not what this panel headlines.
    """
    best = None
    for m in sorted(RUNS.glob("*/metrics.jsonl")):
        if m.stat().st_size == 0:
            continue
        if best is None or m.stat().st_mtime > best.stat().st_mtime:
            best = m
    if best is None:
        return None
    run = last_step = last_stage = None
    curve = []
    try:
        with best.open() as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                kind = rec.get("t")
                if kind == "run":
                    run = rec
                elif kind == "step":
                    last_step = rec
                elif kind == "stage":
                    last_stage = rec
                    u = rec.get("unseen") or {}
                    curve.append({
                        "samples": rec.get("samples"),
                        "tokens": rec.get("tokens"),
                        "unseen": u.get("accuracy"),
                        "chance": u.get("chance"),
                        "unseen_ece": u.get("ece"),
                        "rank": u.get("accuracy_options_only"),
                        "rank_chance": u.get("chance_options_only"),
                        "abstain": u.get("abstain_rate"),
                        "seen": (rec.get("seen") or {}).get("accuracy"),
                        "seen_ece": (rec.get("seen") or {}).get("ece"),
                    })
    except OSError:
        return None
    if run is None and last_step is None:
        return None
    gate = None
    gp = ROOT / "artifacts" / "gates" / "T-train-real" / "gate.json"
    if gp.exists():
        try:
            gate = json.loads(gp.read_text())
        except Exception:
            gate = None
    return {"run_id": best.parent.name, "run": run, "step": last_step,
            "stage": last_stage, "curve": curve, "gate": gate,
            "age_s": max(0.0, time.time() - best.stat().st_mtime),
            "path": str(best.relative_to(ROOT))}


def live_lines(p):
    """Fast live row count (manifest can lag a running generator)."""
    try:
        with open(p, "rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


_QUARANTINE_ROWS = None


def quarantine_rows():
    """Frozen row count of the quarantined corpus (immutable evidence: count
    once per process instead of re-reading ~210 MB every heartbeat)."""
    global _QUARANTINE_ROWS
    if _QUARANTINE_ROWS is None and QUARANTINE.exists():
        _QUARANTINE_ROWS = live_lines(QUARANTINE)
    return _QUARANTINE_ROWS or 0


def scan_gen_train():
    """Latest targeted retrain from the generator loop (runs-gen/*.json).

    The monitor's own full retrain only fires on NEW prefetch data, so when
    it says 'parado' this is the real continuous-training signal: every
    --retrain-every batches the generator trains TF-IDF+LogReg on
    synth-loop and writes one JSON here (train split + held-out eval
    split = forward-test number)."""
    gdir = ROOT / "artifacts" / "runs-gen"
    best = None
    if gdir.exists():
        for f in sorted(gdir.glob("*.json")):
            if f.name == "forward-history.jsonl":
                continue
            try:
                d = json.loads(f.read_text())
            except Exception:
                continue
            try:
                mt = f.stat().st_mtime
            except OSError:
                continue
            if best is None or mt > best[0]:
                best = (mt, f.stem, d)
    if not best:
        return None
    mt, ts, d = best
    return {"ts": ts, "age_s": time.time() - mt, "n_train": d.get("n_train"),
            "n_eval": d.get("n_eval"), "accuracy": d.get("accuracy"),
            "logloss": d.get("logloss"), "seconds": d.get("seconds")}


def scan_forward_history(n=8):
    """Forward-test trend: one entry per targeted retrain, newest last."""
    hist = ROOT / "artifacts" / "runs-gen" / "forward-history.jsonl"
    rows = []
    if hist.exists():
        for line in open(hist):
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows[-n:]


def scan_qwen_sets():
    """Q-W-E-N-built sets: per-file live rows + bytes in artifacts/data-qwen/."""
    qdir = ROOT / "artifacts" / "data-qwen"
    rows, total = [], 0
    if qdir.exists():
        for f in sorted(qdir.glob("*.jsonl")):
            if f.name == "loop-templates.jsonl":
                continue
            try:
                st = f.stat()
            except OSError:
                continue
            n = live_lines(f)
            total += n
            rows.append({
                "name": f.stem, "rows": n,
                "bytes": st.st_size, "size": human(st.st_size),
                "mtime": dt.datetime.fromtimestamp(st.st_mtime,
                         dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            })
    tpl = qdir / "loop-templates.jsonl"
    templates = live_lines(tpl) if tpl.exists() else 0
    return rows, total, templates


def growth_rate(synth_rows, total_ex):
    """Rows/min since monitor start (persisted across cycles)."""
    hist = STATE_DIR / "growth.json"
    now = time.time()
    try:
        h = json.loads(hist.read_text()) if hist.exists() else {}
    except Exception:
        h = {}
    first = h.get("first") or {"ts": now, "synth": synth_rows, "total": total_ex}
    prev = h.get("prev") or first
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    hist.write_text(json.dumps({"first": first,
                                "prev": {"ts": now, "synth": synth_rows,
                                         "total": total_ex}}))
    dt_min = (now - first["ts"]) / 60.0
    if dt_min < 0.5:
        return {"synth_per_min": None, "total_per_min": None,
                "synth_delta": 0, "total_delta": 0,
                "since_min": round(dt_min, 1)}
    return {"synth_per_min": round((synth_rows - first["synth"]) / dt_min, 1),
            "total_per_min": round((total_ex - first["total"]) / dt_min, 1),
            "synth_delta": synth_rows - first["synth"],
            "total_delta": total_ex - first["total"],
            "since_min": round(dt_min, 1)}


def estimate_params(jobs):
    """Baseline params ~= FEATS x classes. Classes unknown -> infer: boolean=2,
    choice>=2 from accuracy/logloss presence; fallback counts distinct via eval.
    We approximate choice classes as 4 when unknown (documented estimate)."""
    total, detail = 0, []
    for j in jobs:
        name = j.get("task", "?")
        boolean = name in ("boolq", "civil-comments")
        classes = 2 if boolean else 4
        p = FEATS * classes
        total += p
        detail.append({"task": name, "classes_assumed": classes, "params": p})
    return total, detail


def proc_alive(needle):
    try:
        out = subprocess.run(["ps", "aux"], capture_output=True, text=True,
                             timeout=10).stdout
        return any(needle in l and "training_monitor" not in l for l in out.splitlines())
    except Exception:
        return False


def ollama_alive():
    try:
        s = socket.create_connection(("127.0.0.1", 11434), timeout=3)
        s.close()
        return True
    except Exception:
        return False


def data_mtime():
    newest = 0.0
    for f in list(PREFETCH.glob("*.jsonl")):
        try:
            newest = max(newest, f.stat().st_mtime)
        except OSError:
            pass
    return newest


class Trainer:
    """Non-blocking training: the monitor heartbeat keeps moving while
    train_baseline.py runs in the background (old code blocked the whole
    loop for hours, terminal + dashboard frozen)."""

    def __init__(self, state_dir):
        self.state_dir = state_dir
        self.proc = None
        self.log = None
        self.start_ts = None
        self.data_ts = 0.0
        self.last_result = "no training in this session yet"
        self.log_off = 0

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def elapsed(self):
        if self.start_ts is None:
            return 0
        return time.time() - self.start_ts

    def maybe_start(self):
        if self.running():
            return None
        marker = self.state_dir / "last_train_data_mtime.txt"
        last = float(marker.read_text().strip()) if marker.exists() else 0.0
        now = data_mtime()
        if now <= last:
            return None
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log = LOG_DIR / f"train-{utcnow().replace(':', '')}.log"
        py = train_python()
        cmd = [py, "data/train_baseline.py"]
        fh = open(self.log, "w")
        fh.write(f"# train started {utcnow()} cmd={' '.join(cmd)}\n")
        fh.flush()
        self.proc = subprocess.Popen(
            cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        fh.close()
        self.start_ts = time.time()
        self.data_ts = now
        self.log_off = 0
        return f"train ARRANCADO pid={self.proc.pid} log={self.log.name}"

    def new_log_lines(self):
        if self.log is None or not self.log.exists():
            return []
        try:
            with open(self.log) as fh:
                fh.seek(self.log_off)
                lines = fh.read().splitlines()
                self.log_off = fh.tell()
        except OSError:
            return []
        return lines

    def tail(self, n=12):
        if self.log is None or not self.log.exists():
            return []
        try:
            lines = self.log.read_text().splitlines()
        except OSError:
            return []
        return lines[-n:]

    def poll(self):
        """Returns a note only when the process just finished."""
        if self.proc is None or self.running():
            return None
        rc = self.proc.poll()
        tail = " | ".join(self.tail(3))[:500]
        if rc == 0:
            marker = self.state_dir / "last_train_data_mtime.txt"
            marker.write_text(str(self.data_ts))
            self.last_result = f"train OK exit=0 log={self.log.name}"
        else:
            self.last_result = (f"train FALLO exit={rc} log={self.log.name} "
                                f"cola: {tail}")
        self.proc = None
        return self.last_result


def build_state():
    sets, total_ex, total_bytes = scan_datasets()
    run_ts, trained, jobs = scan_runs()
    params, pdetail = estimate_params(jobs)
    qwen_sets, qwen_total, qwen_templates = scan_qwen_sets()
    gen_train = scan_gen_train()
    forward_hist = scan_forward_history()
    decision = scan_decision_run()
    unseen_gate = scan_unseen_gate()
    synth_rows = next(
        (s.get("frozen_rows") or s["examples"] for s in sets
         if s["name"] == "synth-loop"), 0)
    synth_trained = next((j.get("n_train", 0) for j in jobs if j.get("task") == "synth-loop"), 0)
    grow = growth_rate(synth_rows, total_ex)
    pending = max(total_ex - trained, 0)
    pct = (100.0 * trained / total_ex) if total_ex else 0.0
    state = {
        "unseen_gate": unseen_gate,
        "decision_run": decision,
        "synth_rows_live": synth_rows,
        "synth_trained": synth_trained,
        "qwen_sets": qwen_sets,
        "qwen_total_rows": qwen_total,
        "qwen_templates": qwen_templates,
        "gen_train": gen_train,
        "forward_hist": forward_hist,
        "growth": grow,
        "ts": utcnow(),
        "datasets": sets,
        "n_datasets": sum(1 for s in sets if not s["name"].startswith("[")),
        "total_examples": total_ex,
        "total_bytes": total_bytes,
        "total_size": human(total_bytes),
        "last_run": run_ts,
        "trained_examples": trained,
        "pending_examples": pending,
        "pct_trained": round(pct, 2),
        "baseline_params_est": params,
        "params_detail": pdetail,
        # #T-split-domain: what this dashboard no longer counts as green.
        "invalidated_numbers": {
            "on": INDEX_SPLIT_INVALID_ON,
            "by": "T-split-domain",
            "reason": INDEX_SPLIT_REASON,
            "tasks": sorted(INDEX_SPLIT_TASKS),
            "evidence": SPLIT_EVIDENCE,
        },
        # Flagged tasks (contaminated / eval-only) keep their history visible
        # but their accuracy is NOT shown as a number: no green without a
        # clean split behind it (#T-halt-contam).
        "per_task": [{"task": j.get("task"), "n_train": j.get("n_train"),
                      "accuracy": (None if j.get("task") in
                                   (CONTAMINATED_TASKS | EVAL_ONLY_TASKS
                                    | INDEX_SPLIT_TASKS)
                                   else (round(j.get("accuracy", 0), 4)
                                         if j.get("accuracy") else None)),
                      "flag": ("CONTAMINADA" if j.get("task") in CONTAMINATED_TASKS
                               else ("eval-only" if j.get("task") in EVAL_ONLY_TASKS
                                     else (f"INVALID {INDEX_SPLIT_INVALID_ON} "
                                           "(split i%10)"
                                           if j.get("task") in INDEX_SPLIT_TASKS
                                           else None))),
                      "invalid_split": j.get("task") in INDEX_SPLIT_TASKS,
                      "seconds": j.get("seconds")} for j in jobs],
        "procs": {
            "ollama_11434": ollama_alive(),
            "ollama_runner_qwen": proc_alive("ollama runner"),
            "qwen_convert": proc_alive("qwen_convert.py"),
            "gen_loop": proc_alive("data_gen_loop.py"),
        },
    }
    return state


def _read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def _last_jsonl(path):
    try:
        lines = Path(path).read_text().splitlines()
        return json.loads(lines[-1]) if lines else {}
    except (OSError, ValueError):
        return {}


def _parse_utc(s):
    if not s:
        return None
    try:
        return (dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
                .replace(tzinfo=dt.timezone.utc).timestamp())
    except ValueError:
        return None


def _gpu_stats():
    """Apple GPU occupancy from IOKit. Zero is a real idle sample, not missing."""
    util, mem = 0, 0
    try:
        raw = subprocess.check_output(
            ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
            stderr=subprocess.DEVNULL, timeout=0.8)
        text = raw.decode("utf-8", "replace")
        nums = [int(x) for x in _RE_GPU_UTIL.findall(text)]
        util = max(nums) if nums else 0
        mems = [int(x) for x in _RE_GPU_MEM.findall(text)]
        mem = max(mems) if mems else 0
    except (OSError, subprocess.SubprocessError):
        pass
    now = time.time()
    _GPU_HIST.append({"t": now, "y": util})
    return {
        "util": util,
        "mem_gb": round(mem / (1024 ** 3), 2) if mem else 0,
        "spark": list(_GPU_HIST),
    }


def _cpu_stats():
    """Machine-wide CPU occupancy, normalized across all logical cores."""
    util = None
    try:
        raw = subprocess.check_output(
            ["ps", "-A", "-o", "%cpu="],
            stderr=subprocess.DEVNULL, timeout=0.8)
        used = sum(float(v) for v in raw.decode().split())
        util = round(min(100.0, used / max(1, os.cpu_count() or 1)), 1)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    now = time.time()
    if util is not None:
        _CPU_HIST.append({"t": now, "y": util})
    return {"util": util, "spark": list(_CPU_HIST)}


def _research_benchmarks():
    """Read comparison figures from their gate artifacts, never from UI constants."""
    qwen = _read_json(
        ROOT / "artifacts" / "gates" / "T-qwen38-ref" / "gate.json")
    laya = _read_json(
        ROOT / "artifacts" / "gates" / "T-laya-baseline" /
        "refs-laya.json")
    base = _read_json(
        ROOT / "artifacts" / "gates" / "T-preflight-refs" /
        "refs-nli-nograd.json")
    jev_gate = _read_json(
        ROOT / "artifacts" / "gates" / "T-teacher-probe" / "battery.json")
    bank_laya = _read_json(
        ROOT / "artifacts" / "gates" / "T-laya-baseline" /
        "fullspace.json")
    parity = _read_json(
        ROOT / "artifacts" / "gates" / "T-jev-parity" / "parity.json")
    qwen_col = ((qwen.get("columns") or {}).get("qwen3.8") or {})
    jev_report = jev_gate.get("report") or {}
    bank_table = bank_laya.get("table") or {}
    bank = bank_table.get("banking77") or {}
    refs = bank.get("references") or parity.get("references") or []
    jev = next((r for r in refs if str(r.get("who", "")).lower()
                .startswith("jev")), {})
    return {
        "battery": {
            "n": ((laya.get("cut") or {}).get("n") or 400),
            "rows_sha256": ((laya.get("cut") or {}).get("split_sha256")),
            "qwen": ((qwen_col.get("forced") or {}).get("accuracy")),
            "qwen_cf": ((qwen_col.get("counterfactual_joint") or {})
                        .get("accuracy")),
            "laya": ((laya.get("ranking") or {}).get("accuracy")),
            "laya_cf": ((laya.get("counterfactual") or {}).get("accuracy")),
            "baseline": ((base.get("ranking") or {}).get("accuracy")),
            "baseline_cf": ((base.get("counterfactual") or {})
                            .get("accuracy")),
            "jev": ((jev_report.get("ranking") or {}).get("accuracy")
                    if jev_gate.get("n_scored") == 400 else None),
            "jev_n": jev_gate.get("n_scored"),
        },
        "banking77": {
            "n": ((bank_laya.get("table") or {}).get("banking77") or {})
            .get("ours", {}).get("n"),
            "laya": (bank.get("ours") or {}).get("accuracy"),
            "jev_cited": jev.get("accuracy"),
            "pointer": (parity.get("ours") or {}).get("accuracy"),
            "jev_is_citation": bool(jev.get("citation")),
        },
    }


def _forced_from_cycle(cycle_n, decision):
    curve = _last_jsonl(
        ROOT / "artifacts" / "checkpoints" / "ce" /
        f"loop-{int(cycle_n):04d}" / "curve.jsonl")
    forced = curve.get("forced")
    joint = curve.get("counterfactual_joint")
    if forced is None:
        m = _RE_FORCED.search(str(decision.get("reason") or ""))
        if m:
            forced = float(m.group(1))
    return (forced, joint, curve.get("decisions_consumed") or 0,
            curve.get("by_family") or {})


def _episodes_from_rule_dir(rule_dir, fallback_batches):
    m = _RE_BATCH.search(str(rule_dir or ""))
    if m:
        return int(m.group(1)) * 5000
    return int(fallback_batches or 0) * 5000


def _listwise_history(producer):
    """Every listwise evaluation in the last 24 h, plus the running best.

    Timestamps come from each cycle's curve mtime walked backwards by the
    measured train time, so the points sit where the work happened.
    """
    root = ROOT / "artifacts" / "checkpoints" / "listwise"
    runs = sorted(p for p in root.glob("loop-*") if (p / "curve.jsonl").exists())
    now = time.time()
    cutoff = now - WINDOW_S
    acc, best_line, vol = [], [], []
    best = None
    last_eval = None
    trained_12h = 0
    fallback_batches = producer.get("n_rule_batches") or 0
    for run in runs:
        curve_file = run / "curve.jsonl"
        try:
            rows = [json.loads(line) for line in
                    curve_file.read_text().splitlines() if line.strip()]
        except (OSError, ValueError):
            continue
        if not rows:
            continue
        manifest = _read_json(run / "train_manifest.json")
        ended = curve_file.stat().st_mtime
        span = float(manifest.get("train_seconds") or 0) + \
            float(manifest.get("eval_seconds") or 0)
        started = ended - span if span else ended
        cycle_n = int(run.name.split("-")[-1])
        # A peak whose weights were overwritten is not a result we hold.
        # Runs before the retain fix only ever kept their last eval.
        marked = [r for r in rows if r.get("retained")]
        if marked:
            kept = max(r.get("forced") or 0 for r in marked)
        else:
            kept = rows[-1].get("forced")
        for i, row in enumerate(rows):
            forced = row.get("forced")
            if forced is None:
                continue
            share = (i + 1) / len(rows)
            ts = started + (ended - started) * share
            if kept is not None and (best is None or kept > best):
                best = kept
            if ts < cutoff:
                continue
            acc.append({"t": ts, "y": round(float(forced), 4),
                        "cycle": cycle_n})
            best_line.append({"t": ts, "y": round(float(best), 4),
                              "cycle": cycle_n})
            last_eval = {
                "forced": round(float(forced), 4),
                "counterfactual": row.get("counterfactual_joint"),
                "cycle": cycle_n,
                "ts": dt.datetime.fromtimestamp(
                    ts, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "promote": bool(row.get("retained")),
                "by_family": row.get("by_family") or {},
            }
        if ended >= now - 12 * 3600:
            trained_12h += int(rows[-1].get("decisions_consumed") or 0)
        vol.append({"t": ended, "y": int(fallback_batches or 0) * 5000,
                    "cycle": cycle_n})
    now_ep = int(fallback_batches or 0) * 5000
    if not vol or vol[-1]["y"] != now_ep:
        vol.append({"t": now, "y": now_ep, "cycle": None})
    return {
        "acc": acc,
        "best": best_line,
        "vol": vol,
        "trained_12h": trained_12h,
        "episodes_now": now_ep,
        "episodes_12h": 0,
        "last_eval": last_eval,
        "cycles_24h": len({p["cycle"] for p in acc}),
    }


def _history_24h(producer):
    """One point per completed cycle in the last 24 h. Cached per latest mtime."""
    loop_root = ROOT / "artifacts" / "loop"
    dirs = sorted(p for p in loop_root.glob("cycle-*") if p.is_dir())
    latest = dirs[-1] if dirs else None
    sig = (
        latest.name if latest else None,
        latest.stat().st_mtime if latest else 0,
        producer.get("n_rule_batches"),
        producer.get("measured_utc"),
    )
    if _HIST_CACHE["sig"] == sig:
        return _HIST_CACHE["data"]
    now = time.time()
    cutoff = now - WINDOW_S
    acc, vol = [], []
    trained_12h = 0
    episodes_12h_start = None
    last_eval = None
    fallback_batches = producer.get("n_rule_batches") or 0
    for p in dirs:
        decision = _read_json(p / "decision.json")
        if not decision:
            continue
        ts = _parse_utc(decision.get("measured_utc"))
        if ts is None or ts < cutoff:
            continue
        cycle_n = decision.get("cycle") or int(p.name.split("-")[-1])
        forced, joint, consumed, families = _forced_from_cycle(
            cycle_n, decision)
        budget = int(decision.get("budget") or consumed or 0)
        episodes = _episodes_from_rule_dir(decision.get("rule_dir"),
                                           fallback_batches)
        if ts >= now - 12 * 3600:
            trained_12h += budget
            if episodes_12h_start is None:
                episodes_12h_start = episodes
        if forced is not None:
            acc.append({"t": ts, "y": round(float(forced), 4),
                        "cycle": cycle_n})
            last_eval = {
                "forced": round(float(forced), 4),
                "counterfactual": (round(float(joint), 4)
                                   if joint is not None else None),
                "cycle": cycle_n,
                "ts": dt.datetime.fromtimestamp(
                    ts, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "promote": bool(decision.get("promote")),
                "by_family": families,
            }
        vol.append({"t": ts, "y": episodes, "cycle": cycle_n})
    now_ep = int(fallback_batches or 0) * 5000
    if not vol or vol[-1]["y"] != now_ep:
        vol.append({"t": now, "y": now_ep, "cycle": None})
    payload = {
        "acc": acc,
        "vol": vol,
        "trained_12h": trained_12h,
        "episodes_now": now_ep,
        "episodes_12h": max(0, now_ep - (episodes_12h_start or now_ep)),
        "last_eval": last_eval,
        "cycles_24h": len(vol) - (1 if vol and vol[-1]["cycle"] is None else 0),
    }
    _HIST_CACHE["sig"] = sig
    _HIST_CACHE["data"] = payload
    return payload


def live_snapshot():
    """One-second payload: live occupancy + 24 h history that survives a cycle reset."""
    loop_root = ROOT / "artifacts" / "loop"
    checkpoint_root = ROOT / "artifacts" / "checkpoints" / "ce"
    state = _read_json(loop_root / "state.json")
    listwise_state = _read_json(
        ROOT / "artifacts" / "listwise-loop" / "state.json")
    listwise_root = ROOT / "artifacts" / "checkpoints" / "listwise"
    if listwise_state.get("active_scorer") == "listwise":
        state = {
            **state,
            "cycle": listwise_state.get("cycle"),
            "active_weights": "mdeberta-v3-base-xnli",
            "active_device": "mps",
            "current": listwise_state.get("best_checkpoint"),
            "last_decision": (
                f"listwise best {listwise_state.get('best_forced')}"
                if listwise_state.get("best_forced") is not None
                else "listwise first cycle"),
            "streak_no_promote": 0,
            "step": 1,
        }
        checkpoint_root = listwise_root
    producer = _read_json(loop_root / "producer.json")
    training_flag = loop_root / "TRAINING"
    active_runs = sorted(
        (p for p in checkpoint_root.glob("loop-*") if p.is_dir()),
        key=lambda p: p.name)
    active = active_runs[-1] if active_runs else None
    curve = _last_jsonl(active / "curve.jsonl") if active else {}
    manifest = _read_json(active / "train_manifest.json") if active else {}
    weight_id = state.get("active_weights") or "minilmv2-l6-mnli-xnli"
    weights = _read_json(
        ROOT / "artifacts" / "weights" / weight_id / "manifest.json")
    elapsed = (max(0, int(time.time() - training_flag.stat().st_mtime))
               if training_flag.exists() else 0)
    hist = (_listwise_history(producer)
            if checkpoint_root == listwise_root
            else _history_24h(producer))
    last = hist.get("last_eval") or {}
    live_forced = curve.get("forced")
    headline_forced = (live_forced if live_forced is not None
                       else last.get("forced"))
    live_joint = curve.get("counterfactual_joint")
    headline_joint = (live_joint if live_joint is not None
                      else last.get("counterfactual"))
    budget = int((manifest.get("budget")
                  or (manifest.get("hyperparams") or {}).get("budget")
                  or 5000))
    consumed = int(curve.get("decisions_consumed")
                   or manifest.get("decisions_consumed") or 0)
    gpu = _gpu_stats()
    active_forced = (listwise_state.get("active_forced")
                     if listwise_state.get("active_forced") is not None
                     else listwise_state.get("best_forced"))
    active_run = (listwise_state.get("active_checkpoint")
                  or listwise_state.get("best_checkpoint"))
    active_run = Path(active_run).name if active_run else None
    if active_forced is None:
        active_forced = headline_forced
    span = GOAL_ACC - START_ACC
    toward = 0.0
    if active_forced is not None and span > 0:
        toward = max(0.0, min(1.0, (active_forced - START_ACC) / span))
    return {
        "ts": utcnow(),
        "now": time.time(),
        "training": training_flag.exists(),
        "cycle": (int(state.get("cycle") or 0) +
                  (1 if training_flag.exists() else 0)),
        "completed_cycle": state.get("cycle"),
        "step": state.get("step"),
        "streak_no_promote": state.get("streak_no_promote"),
        "current": state.get("current"),
        "last_decision": state.get("last_decision"),
        "run": active.name if active else None,
        "elapsed_s": elapsed,
        "decisions_consumed": consumed,
        "budget": budget,
        "steps": curve.get("steps") or 0,
        "loss": curve.get("train_loss"),
        "forced": headline_forced,
        "forced_live": live_forced,
        "active_forced": active_forced,
        "active_run": active_run,
        "candidate_forced": live_forced,
        "last_promoted": listwise_state.get("last_promoted"),
        "counterfactual": headline_joint,
        "holdout": curve.get("holdout_forced"),
        "by_family": (curve.get("by_family")
                      or last.get("by_family") or {}),
        "producer": producer,
        "rule_batches": producer.get("n_rule_batches") or 0,
        "model": ("mDeBERTa listwise"
                  if checkpoint_root == listwise_root
                  else (weights.get("id") or weight_id)),
        "params_m": weights.get("params_m") or 107,
        "device": state.get("active_device") or "mps",
        "gpu": gpu,
        "cpu": _cpu_stats(),
        "chance_acc": CHANCE_ACC,
        "start_acc": START_ACC,
        "goal_acc": GOAL_ACC,
        "toward_goal": round(toward, 4),
        "history": hist,
        "benchmarks": _research_benchmarks(),
        "dashboard_age_s": 0,
    }


DASH_TMPL = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>JEv training monitor</title>
<style>
:root{{--bg:#070a10;--panel:rgba(16,23,34,.88);--panel2:rgba(10,16,25,.9);--line:#26344a;--txt:#f2f6fb;--dim:#8491a5;--acc:#39e6a3;--blue:#58a6ff;--violet:#9b7cff;--warn:#f2bd4a;--bad:#ff637d}}
*{{box-sizing:border-box}}body{{font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;background:radial-gradient(circle at 18% -15%,rgba(88,166,255,.16),transparent 34%),radial-gradient(circle at 82% 0,rgba(155,124,255,.11),transparent 28%),var(--bg);color:var(--txt);margin:0;padding:0 14px 20px;min-height:100vh}}
header.top{{display:grid;grid-template-columns:minmax(220px,1fr) auto minmax(220px,1fr);align-items:center;gap:16px;padding:9px 4px 9px;position:sticky;top:0;background:rgba(9,13,18,.94);backdrop-filter:blur(12px);z-index:5;border-bottom:1px solid var(--line)}}
header.top h1{{margin:0;font-size:19px;letter-spacing:-.02em;white-space:nowrap}}header.top small{{color:var(--dim);font-size:13px;font-weight:400;font-variant-numeric:tabular-nums}}
.brand small{{display:block;margin:2px 0 0 17px;text-transform:uppercase;letter-spacing:.16em;font-size:10px}}
.header-right{{display:flex;align-items:center;justify-content:flex-end;gap:12px}}.live-badge{{border:1px solid rgba(57,230,163,.35);background:rgba(57,230,163,.08);color:var(--acc);border-radius:999px;padding:4px 9px;font-size:11px;font-weight:700;letter-spacing:.08em;white-space:nowrap}}
.live-dot{{display:inline-block;width:9px;height:9px;border-radius:50%;background:var(--acc);margin-right:7px;box-shadow:0 0 10px var(--acc);animation:pulse 1.5s infinite}}
@keyframes pulse{{50%{{opacity:.38}}}}
nav{{display:flex;gap:6px;justify-self:center;justify-content:center;flex-wrap:nowrap}}
nav button{{border:1px solid var(--line);background:var(--panel2);color:var(--dim);padding:6px 14px;border-radius:999px;cursor:pointer;white-space:nowrap;font-size:13px;transition:color .18s,border-color .18s,background .18s}}
nav button:hover{{color:var(--txt)}}
nav button.active{{color:var(--txt);border-color:var(--blue);background:#12233a;box-shadow:0 0 12px rgba(88,166,255,.18)}}
.tab-page{{display:none}}.tab-page.active{{display:block}}
section.panel{{background:linear-gradient(145deg,rgba(18,27,40,.94),rgba(10,16,25,.92));border:1px solid var(--line);border-radius:10px;padding:11px 13px;margin-top:10px;box-shadow:0 12px 35px rgba(0,0,0,.15),inset 0 1px 0 rgba(255,255,255,.025)}}
section.panel h2{{margin:0 0 9px;font-size:12px;text-transform:uppercase;letter-spacing:.12em;color:var(--dim)}}
main.tab-page>.cards:first-child{{margin-top:10px}}
p.warn{{color:var(--warn);border-left:3px solid var(--warn);padding-left:10px;margin:0 0 12px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:11px}}
.cards.activity-cards{{grid-template-columns:repeat(8,minmax(0,1fr));gap:10px}}
.card{{background:linear-gradient(150deg,rgba(15,23,35,.96),rgba(8,13,21,.96));border:1px solid var(--line);border-radius:8px;padding:10px 12px;min-height:78px;box-shadow:inset 0 1px 0 rgba(255,255,255,.025)}}
.activity-cards .card{{min-height:76px;padding:9px 10px}}
.card .k{{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.card .v{{font-size:21px;font-weight:680;margin-top:3px;font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.card .s{{font-size:11px;color:var(--dim);margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.hero{{display:grid;grid-template-columns:1fr;gap:14px}}
.charts-main{{display:grid;grid-template-columns:minmax(0,1.7fr) minmax(280px,.8fr);gap:10px}}
.mid-grid{{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,1fr) minmax(250px,.85fr);gap:10px}}
@media(max-width:1200px){{.mid-grid{{grid-template-columns:1fr}}}}
.chart{{height:198px;background:linear-gradient(180deg,rgba(90,169,255,.09),transparent);border:1px solid var(--line);border-radius:8px;padding:6px 8px 4px;position:relative}}
.chart.tall{{height:238px}}
.chart svg{{width:100%;height:calc(100% - 27px);display:block}}
.chart line.grid{{stroke:#253142;stroke-width:1}}
.chart line.mark{{stroke-dasharray:4 5;stroke-width:1}}
.chart polyline{{fill:none;stroke:var(--blue);stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round;filter:drop-shadow(0 0 4px rgba(88,166,255,.3))}}
.chart .gpu{{stroke:var(--acc)}}.chart .cpu{{stroke:var(--blue)}}
.chart.short{{height:130px}}
.chart-labels{{display:flex;justify-content:space-between;align-items:center;color:var(--dim);font-size:13px;margin-top:4px;white-space:nowrap;gap:12px}}
.chart-labels b{{font-weight:650}}
.axis-y{{position:absolute;left:6px;top:7px;bottom:30px;width:34px;color:var(--dim);font-size:11px;font-variant-numeric:tabular-nums;pointer-events:none}}
.axis-y span{{position:absolute;transform:translateY(-50%)}}
.axis-y .y100{{top:0;transform:none}}.axis-y .y75{{top:25%}}.axis-y .y50{{top:50%}}.axis-y .y25{{top:75%}}.axis-y .y0{{bottom:0;transform:none}}
.chart.with-axis svg{{margin-left:36px;width:calc(100% - 36px)}}
.chart.named{{padding-left:32px}}
.chart .axis-name{{position:absolute;left:4px;top:8px;bottom:29px;width:24px;display:flex;align-items:center;justify-content:center;font-size:17px;font-weight:750;letter-spacing:.12em;writing-mode:vertical-rl;transform:rotate(180deg);line-height:1;pointer-events:none}}
.chart .axis-name.gpu{{color:var(--acc)}}.chart .axis-name.cpu{{color:var(--blue)}}
.ticker{{font:14px ui-monospace,SFMono-Regular,monospace;color:var(--dim);line-height:1.7}}.ticker b{{color:var(--txt)}}
.lower-grid{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr) minmax(280px,1.1fr);gap:10px}}
.ref-list{{display:grid;gap:9px}}.ref-row{{display:grid;grid-template-columns:96px 58px 1fr;align-items:center;gap:9px;font-size:12px}}.ref-name{{font-weight:750;color:var(--txt);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.ref-score{{font-variant-numeric:tabular-nums;font-weight:700;text-align:right}}.ref-track{{height:14px;background:#243044;border-radius:7px;overflow:hidden;position:relative}}.ref-fill{{display:block;height:14px;border-radius:7px;background:#6d7b90;transition:width .6s}}.ref-note{{font-size:10px;color:var(--dim);margin-top:9px;line-height:1.4}}.tag{{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:1px 6px;font-size:9px;letter-spacing:.06em;color:var(--dim);margin-left:5px}}
.ref-fill.winner{{background:#2ec9a0}}
.ref-fill.ours{{background:#5b9dff}}
.ref-group{{margin-bottom:11px}}.ref-group:last-child{{margin-bottom:0}}
.ref-group h3{{margin:0 0 8px;font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:var(--dim);font-weight:650}}
.fam-list{{display:grid;gap:8px}}
.fam-row{{display:grid;grid-template-columns:1fr 52px;align-items:center;gap:9px;font-size:12px}}
.fam-name{{color:var(--dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.fam-val{{text-align:right;font-variant-numeric:tabular-nums;font-weight:700}}
.fam-track{{grid-column:1/-1;height:6px;background:#0b111b;border-radius:5px;overflow:hidden;margin-top:-3px}}
.fam-fill{{height:100%;border-radius:5px;transition:width .5s}}
.spec{{display:grid;gap:7px;font-size:12px}}
.spec-row{{display:flex;justify-content:space-between;gap:10px;padding-bottom:6px;border-bottom:1px solid rgba(38,52,74,.55)}}
.spec-row:last-child{{border-bottom:none;padding-bottom:0}}
.spec-k{{color:var(--dim)}}.spec-v{{font-weight:700;font-variant-numeric:tabular-nums;text-align:right}}
.goalbar{{background:var(--panel2);border:1px solid var(--line);height:12px;border-radius:6px;overflow:hidden;margin-top:8px;position:relative}}
.goalbar .fill{{background:var(--blue);height:100%}}
.goalbar .tick{{position:absolute;top:0;bottom:0;width:1px;background:var(--warn)}}
.status-running{{color:var(--acc)}}.status-idle{{color:var(--warn)}}.delta-up{{color:var(--acc)}}.delta-down{{color:var(--bad)}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}@media(max-width:1100px){{.grid2{{grid-template-columns:1fr}}}}
table{{border-collapse:collapse;width:100%;font-size:15px;font-variant-numeric:tabular-nums}}
th,td{{border-bottom:1px solid var(--line);padding:7px 10px;text-align:left}}
th{{color:var(--dim);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.06em}}
td.num,th.num{{text-align:right}}tr:last-child td{{border-bottom:none}}
.bar{{background:var(--panel2);border:1px solid var(--line);height:10px;border-radius:6px;overflow:hidden;margin-top:10px}}
.fill{{background:var(--acc);height:100%;transition:width .5s}}
pre{{background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:12px;overflow:auto;max-height:300px;font-size:14px;line-height:1.5}}
.pill{{display:inline-block;padding:2px 10px;border-radius:999px;font-size:14px;border:1px solid var(--line)}}
.on{{color:var(--acc);border-color:var(--acc)}}.off{{color:var(--dim)}}footer{{color:var(--dim);font-size:11px;margin-top:7px}}
@media(max-width:850px){{.cards.activity-cards{{grid-template-columns:repeat(4,1fr)}}.charts-main,.lower-grid{{grid-template-columns:1fr}}}}
</style></head><body>
<header class="top"><div class="brand"><h1><span class="live-dot"></span>JEv · Decision Intelligence Lab</h1><small>Continuous learning research system</small></div>
<nav>
<button class="active" data-tab="activity">Overview</button><button data-tab="training">Training</button>
<button data-tab="data">Datasets</button><button data-tab="model">Model</button>
<button data-tab="benchmarks">Benchmarks</button>
</nav>
<div class="header-right"><span class="live-badge">LIVE RESEARCH RUN</span><small id="clock">{tick_ts}</small></div></header>
<main id="activity" class="tab-page active">
<div class="cards activity-cards">
<div class="card"><div class="k">Research state</div><div class="v status-running" id="live-state">LOADING</div><div class="s" id="live-elapsed">-</div></div>
<div class="card"><div class="k">Active model accuracy</div><div class="v" id="live-forced">-</div><div class="s" id="live-forced-src">promoted model</div></div>
<div class="card"><div class="k">Progress to 99%</div><div class="v" id="live-toward">-</div><div class="s" id="live-goal-s">baseline 62.25%</div>
<div class="goalbar"><div class="fill" id="live-goal-fill" style="width:0%"></div></div></div>
<div class="card"><div class="k">Research corpus</div><div class="v" id="live-episodes">-</div><div class="s" id="live-ep-rate">verified episodes</div></div>
<div class="card"><div class="k">Training · 12 h</div><div class="v" id="live-12h">-</div><div class="s" id="live-12h-s">decisions processed</div></div>
<div class="card"><div class="k">Active cycle</div><div class="v" id="live-cycle">-</div><div class="s" id="live-consumed">batch progress</div></div>
<div class="card"><div class="k">Candidate in training</div><div class="v" id="live-current">-</div><div class="s" id="live-streak">candidate under test</div></div>
<div class="card"><div class="k">Counterfactual</div><div class="v" id="live-counter">-</div><div class="s">joint consistency</div></div>
</div>
<div class="charts-main">
<section class="panel"><h2>Decision accuracy · 24 hours · full 0-100% scale</h2>
<div class="chart tall with-axis"><div class="axis-y"><span class="y100">100%</span><span class="y75">75%</span><span class="y50">50%</span><span class="y25">25%</span><span class="y0">0%</span></div><svg id="svg-acc" viewBox="0 0 900 300" preserveAspectRatio="none">
<line class="grid" x1="0" y1="75" x2="900" y2="75"/><line class="grid" x1="0" y1="150" x2="900" y2="150"/><line class="grid" x1="0" y1="225" x2="900" y2="225"/>
<line class="mark" id="mark-chance" x1="0" y1="0" x2="900" y2="0" style="stroke:var(--bad)"/>
<line class="mark" id="mark-start" x1="0" y1="0" x2="900" y2="0" style="stroke:var(--dim)"/>
<line class="mark" id="mark-h3" x1="0" y1="0" x2="900" y2="0" style="stroke:var(--warn)"/>
<line class="mark" id="mark-goal" x1="0" y1="0" x2="900" y2="0" style="stroke:var(--acc)"/>
<polyline id="line-acc" points=""/>
<polyline id="line-best" points="" style="stroke:var(--acc);stroke-width:2.5;fill:none"/></svg>
<div class="chart-labels"><span><span style="color:var(--bad)">chance 33.75%</span> · <span style="color:var(--dim)">baseline 62.25%</span> · <b id="legend-current" style="color:var(--blue)">candidate -</b> · <b id="legend-best" style="color:var(--acc)">best -</b> · <span style="color:var(--warn)">H3 70%</span> · <span style="color:var(--acc)">target 99%</span></span><span>24 h</span></div></div>
</section>
<section class="panel"><h2>Verified corpus growth · 24 hours</h2>
<div class="chart tall"><svg viewBox="0 0 900 280" preserveAspectRatio="none">
<line class="grid" x1="0" y1="70" x2="900" y2="70"/><line class="grid" x1="0" y1="140" x2="900" y2="140"/><line class="grid" x1="0" y1="210" x2="900" y2="210"/>
<polyline id="line-vol" points=""/></svg>
<div class="chart-labels"><span><b style="color:var(--blue)">published episodes</b></span><span id="vol-max">-</span></div></div>
</section>
</div>
<div class="mid-grid">
<section class="panel"><h2>Competitive benchmark <span class="tag">MEASURED + CITED</span></h2>
<div class="ref-group"><h3>Decision battery · same 400 rows · K=2/3/8</h3>
<div class="ref-list">
<div class="ref-row"><span class="ref-name">Qwen 3.8</span><span class="ref-score" data-metric="ref-qwen">-</span><div class="ref-track"><div class="ref-fill" data-metric="ref-qwen-bar" style="width:{w_qwen}"></div></div></div>
<div class="ref-row"><span class="ref-name">TypeSafe JEv</span><span class="ref-score" data-metric="ref-jev">-</span><div class="ref-track"><div class="ref-fill winner" data-metric="ref-jev-bar" style="width:{w_jev}"></div></div></div>
<div class="ref-row" data-role="ours"><span class="ref-name">JEvClone</span><span class="ref-score" data-metric="ref-current">-</span><div class="ref-track"><div class="ref-fill ours" data-metric="ref-current-bar" style="width:{w_current}"></div></div></div>
<div class="ref-row"><span class="ref-name">MiniLM base</span><span class="ref-score" data-metric="ref-base">-</span><div class="ref-track"><div class="ref-fill" data-metric="ref-base-bar" style="width:{w_base}"></div></div></div>
<div class="ref-row"><span class="ref-name">Laya</span><span class="ref-score" data-metric="ref-laya">-</span><div class="ref-track"><div class="ref-fill" data-metric="ref-laya-bar" style="width:{w_laya}"></div></div></div>
</div></div>
<div class="ref-group"><h3>BANKING77 · K=77 · 3,080 rows</h3>
<div class="ref-list">
<div class="ref-row"><span class="ref-name">JEv (cited)</span><span class="ref-score" data-metric="ref-bank-jev">-</span><div class="ref-track"><div class="ref-fill winner" data-metric="ref-bank-jev-bar" style="width:{w_bank_jev}"></div></div></div>
<div class="ref-row"><span class="ref-name">Laya</span><span class="ref-score" data-metric="ref-bank-laya">-</span><div class="ref-track"><div class="ref-fill" data-metric="ref-bank-laya-bar" style="width:{w_bank_laya}"></div></div></div>
<div class="ref-row" data-role="ours"><span class="ref-name">Clone · untuned</span><span class="ref-score" data-metric="ref-bank-ours">-</span><div class="ref-track"><div class="ref-fill ours" data-metric="ref-bank-ours-bar" style="width:{w_bank_ours}"></div></div></div>
</div></div>
<div class="ref-note">Green is the leader. Blue is this clone. Every other result is grey. The BANKING77 JEv figure is a published citation under a richer information regime. Chance there is 1.3%, and the clone was never trained on that 77-label space.</div>
</section>
<section class="panel"><h2>Accuracy by decision family · live candidate</h2>
<div class="fam-list" id="fam-list"><div class="fam-row"><span class="fam-name">waiting for evaluation</span><span class="fam-val">-</span></div></div>
</section>
<section class="panel"><h2>Research infrastructure</h2>
<div class="spec">
<div class="spec-row"><span class="spec-k">Compute</span><span class="spec-v">Apple M5 Pro · 20-core GPU</span></div>
<div class="spec-row"><span class="spec-k">Unified memory</span><span class="spec-v" id="spec-mem">-</span></div>
<div class="spec-row"><span class="spec-k">Backbone</span><span class="spec-v" id="spec-model">MiniLMv2-L6</span></div>
<div class="spec-row"><span class="spec-k">Parameters</span><span class="spec-v" id="spec-params">-</span></div>
<div class="spec-row"><span class="spec-k">Teacher</span><span class="spec-v">Qwen 3.8 · 27B MLX</span></div>
<div class="spec-row"><span class="spec-k">Cycles completed</span><span class="spec-v" id="spec-cycles">-</span></div>
<div class="spec-row"><span class="spec-k">Holdout accuracy</span><span class="spec-v" id="spec-holdout">-</span></div>
<div class="spec-row"><span class="spec-k">Batch loss</span><span class="spec-v" id="spec-loss">-</span></div>
<div class="spec-row"><span class="spec-k">Data producer</span><span class="spec-v" id="spec-producer">-</span></div>
</div>
</section>
</div>
<div class="lower-grid">
<section class="panel"><h2>Accelerator load · last 2 min</h2>
<div class="chart short named"><span class="axis-name gpu">GPU</span><svg viewBox="0 0 900 160" preserveAspectRatio="none">
<line class="grid" x1="0" y1="40" x2="900" y2="40"/><line class="grid" x1="0" y1="80" x2="900" y2="80"/><line class="grid" x1="0" y1="120" x2="900" y2="120"/>
<polyline id="line-gpu" class="gpu" points=""/></svg>
<div class="chart-labels"><span>usage</span><span id="live-gpu">-</span></div></div>
</section>
<section class="panel"><h2>Host load · last 2 min</h2>
<div class="chart short named"><span class="axis-name cpu">CPU</span><svg viewBox="0 0 900 160" preserveAspectRatio="none">
<line class="grid" x1="0" y1="40" x2="900" y2="40"/><line class="grid" x1="0" y1="80" x2="900" y2="80"/><line class="grid" x1="0" y1="120" x2="900" y2="120"/>
<polyline id="line-cpu" class="cpu" points=""/></svg>
<div class="chart-labels"><span>usage</span><span id="live-cpu">-</span></div></div>
</section>
<section class="panel"><h2>Same-row research battery <span class="tag">400 ROWS</span></h2>
<div class="ref-list">
<div class="ref-row"><span class="ref-name">Qwen 3.8</span><span class="ref-score" data-metric="ref-qwen">-</span><div class="ref-track"><div class="ref-fill" data-metric="ref-qwen-bar" style="width:{w_qwen}"></div></div></div>
<div class="ref-row" data-role="ours"><span class="ref-name">Candidate</span><span class="ref-score" data-metric="ref-current">-</span><div class="ref-track"><div class="ref-fill ours" data-metric="ref-current-bar" style="width:{w_current}"></div></div></div>
<div class="ref-row"><span class="ref-name">MiniLM base</span><span class="ref-score" data-metric="ref-base">-</span><div class="ref-track"><div class="ref-fill" data-metric="ref-base-bar" style="width:{w_base}"></div></div></div>
<div class="ref-row"><span class="ref-name">Laya</span><span class="ref-score" data-metric="ref-laya">-</span><div class="ref-track"><div class="ref-fill" data-metric="ref-laya-bar" style="width:{w_laya}"></div></div></div>
</div>
<div class="ref-note">Identical battery-dev rows and candidate sets. TypeSafe JEv appears only after the complete 400-row gate has been returned. The candidate remains experimental while the promotion gate says NO-GO.</div>
</section>
</div>
</main>
<main id="training" class="tab-page">
{decision_sec}
<section class="panel"><h2>Execution summary</h2><div class="cards">
<div class="card"><div class="k">historical training</div><div class="v">{trained:,}</div><div class="s">run {run}</div></div>
<div class="card"><div class="k">total examples</div><div class="v">{total:,}</div><div class="s">{size}</div></div>
<div class="card"><div class="k">pending</div><div class="v">{pending:,}</div><div class="s">{pct}%</div></div>
<div class="card"><div class="k">historical baseline</div><div class="v">≈ {params:,}</div><div class="s">TF-IDF · not the product model</div></div>
</div><div class="bar"><div class="fill" style="width:{pct}%"></div></div></section>
<section class="panel"><h2>Training log</h2><pre>{train_tail}</pre><p><span class="pill">{procs}</span></p><p><i>{note}</i></p></section>
</main>
<main id="data" class="tab-page">
<section class="panel"><h2>Producers and corpora</h2><div class="cards">
<div class="card"><div class="k">synth-loop on disk</div><div class="v">{synth_live}</div><div class="s">trained {synth_tr} · {rate}</div></div>
<div class="card"><div class="k">total Qwen</div><div class="v">{qwen_tot}</div><div class="s">templates {qwen_tpl}</div></div>
<div class="card"><div class="k">historical generator</div><div class="v">{gen_state}</div><div class="s">data_gen_loop.py</div></div>
<div class="card"><div class="k">Ollama</div><div class="v">{oll_state}</div><div class="s">runner {run_state} · convert {conv_state}</div></div>
</div>{synth_sec}{fwd_sec}</section>
<section class="panel"><h2>Datasets</h2><table><tr><th>dataset</th><th class="num">examples</th><th class="num">size</th><th>status</th></tr>{rows}</table></section>
</main>
<main id="model" class="tab-page">
<section class="panel"><h2>Decision model</h2><div class="cards">
<div class="card"><div class="k">backbone</div><div class="v">MiniLMv2-L6</div><div class="s">multilingual NLI</div></div>
<div class="card"><div class="k">parameters</div><div class="v">107M</div><div class="s">encoder + classification head</div></div>
<div class="card"><div class="k">device</div><div class="v">Apple MPS</div><div class="s">M5 Pro · 48 GB</div></div>
<div class="card"><div class="k">promotion</div><div class="v">GATE</div><div class="s">paired bootstrap · no regression</div></div>
</div></section>
</main>
<main id="benchmarks" class="tab-page">
<section class="panel"><h2>Battery-dev · same 400 rows · K=2/3/8 · English and Spanish</h2>
<div class="cards">
<div class="card"><div class="k">Qwen 3.8 teacher</div><div class="v">{battery_qwen}</div><div class="s">gate artifact · measured</div></div>
<div class="card"><div class="k">MiniLM untuned</div><div class="v">{battery_base}</div><div class="s">control checkpoint · measured</div></div>
<div class="card"><div class="k">Laya</div><div class="v">{battery_laya}</div><div class="s">gate artifact · measured</div></div>
<div class="card"><div class="k">TypeSafe JEv</div><div class="v">{battery_jev}</div><div class="s">remote gate · same 400 rows</div></div>
</div>
<p>Forced accuracy is paired on the exact same rows and candidate sets.
The active candidate is shown live in Overview and remains experimental when
its gate verdict is NO-GO.</p></section>
<section class="panel"><h2>BANKING77 · K=77 · protocol-labelled comparison</h2>
<div class="cards">
<div class="card"><div class="k">JEv teacher</div><div class="v">{bank_jev}</div><div class="s">citation only · richer information regime</div></div>
<div class="card"><div class="k">Laya</div><div class="v">{bank_laya}</div><div class="s">n=3,080 · measured locally</div></div>
<div class="card"><div class="k">JEvClone pointer</div><div class="v">{bank_pointer}</div><div class="s">n=3,080 · chance 1.30% · measured</div></div>
<div class="card"><div class="k">Same-protocol JEv</div><div class="v">PENDING</div><div class="s">API credentials or local weights required</div></div>
</div>
<p>JEv's 92.40% is cited from the public BANKING77 experiment. It is not a
same-pipeline measurement and is never presented as one.</p></section>
{unseen_sec}
<section class="panel"><h2>Historical comparisons by task</h2>{invalid_sec}<table><tr><th>task</th><th class="num">n_train</th><th class="num">accuracy</th><th class="num">seconds</th></tr>{trows}</table></section>
</main>
<footer>Live telemetry: 1 s · research curves: completed evaluations over 24 h · static evidence: {tick} s</footer>
<script>
const $=id=>document.getElementById(id);
const fmt=v=>v===null||v===undefined||v===''?'-':Number(v).toFixed(4);
const nfmt=v=>Number(v||0).toLocaleString('es');
const dur=s=>`${{Math.floor(s/60)}}m ${{s%60}}s`;
function yOf(v,min,max,h){{return h-((Math.max(min,Math.min(max,v))-min)*h/(max-min));}}
function plotWindow(id,series,ymin,ymax,now,windowS,w,h){{
  const t0=now-windowS;
  const pts=(series||[]).filter(p=>p&&p.t>=t0&&p.y!=null).map(p=>`${{((p.t-t0)/windowS)*w}},${{yOf(p.y,ymin,ymax,h)}}`).join(' ');
  $(id).setAttribute('points',pts);
}}
function hline(id,y,ymin,ymax,h){{const yy=yOf(y,ymin,ymax,h);$(id).setAttribute('y1',yy);$(id).setAttribute('y2',yy);}}
async function live(){{
 try{{
  const r=await fetch('/api/live?'+Date.now());const x=await r.json();
  const h=x.history||{{}}; const last=h.last_eval||{{}};
  $('clock').textContent=x.ts+' · 1 s';
  $('live-state').textContent=x.training?'TRAINING':'EVALUATING';
  $('live-state').className='v '+(x.training?'status-running':'status-idle');
  $('live-elapsed').textContent=x.training?dur(x.elapsed_s):'closing batch and scoring';
  const gpu=(x.gpu&&x.gpu.util!=null)?x.gpu.util:null;
  $('live-gpu').textContent=gpu==null?'-':gpu+'%';
  const cpu=(x.cpu&&x.cpu.util!=null)?x.cpu.util:null;
  $('live-cpu').textContent=cpu==null?'-':cpu+'%';
  $('live-forced').textContent=fmt(x.active_forced);
  $('live-forced-src').textContent=x.active_run?('promoted model · '+x.active_run):'no promoted model yet';
  const pct=Math.round((x.toward_goal||0)*100);
  $('live-toward').textContent=pct+'%';
  $('live-goal-s').textContent=`baseline ${{fmt(x.start_acc)}} / target ${{fmt(x.goal_acc)}}`;
  $('live-goal-fill').style.width=pct+'%';
  $('live-episodes').textContent=nfmt(h.episodes_now);
  $('live-ep-rate').textContent=`+${{nfmt(h.episodes_12h)}} in 12 h`;
  $('live-12h').textContent=nfmt(h.trained_12h);
  $('live-12h-s').textContent=`${{h.cycles_24h||0}} cycles in 24 h`;
  $('live-cycle').textContent=x.cycle||'-';
  $('live-consumed').textContent=`${{nfmt(x.decisions_consumed)}} / ${{nfmt(x.budget)}} decisions`;
  $('live-current').textContent=x.candidate_forced!=null?fmt(x.candidate_forced):'measuring';
  $('live-streak').textContent=x.active_forced!=null?('candidate vs active '+fmt(x.active_forced)):'candidate under test';
  $('live-counter').textContent=fmt(x.counterfactual);
  const now=x.now||(Date.now()/1000);
  const ymin=0, ymax=1.0, ah=300, aw=900;
  hline('mark-chance', x.chance_acc||0.3375, ymin, ymax, ah);
  hline('mark-start', x.start_acc||0.6225, ymin, ymax, ah);
  hline('mark-h3', 0.70, ymin, ymax, ah);
  hline('mark-goal', x.goal_acc||0.99, ymin, ymax, ah);
  plotWindow('line-acc', h.acc, ymin, ymax, now, 86400, aw, ah);
  plotWindow('line-best', h.best||[], ymin, ymax, now, 86400, aw, ah);
  const accPoints=(h.acc||[]).filter(p=>p&&p.t>=now-86400&&p.y!=null);
  const current=x.candidate_forced!=null?{{y:x.candidate_forced}}:(accPoints.length?accPoints[accPoints.length-1]:null);
  $('legend-current').textContent=current?'candidate '+(current.y*100).toFixed(2)+'%':'candidate -';
  $('legend-best').textContent=x.active_forced!=null?'best '+(x.active_forced*100).toFixed(2)+'%':'best -';
  const setMetric=(id,text,width)=>{{
    document.querySelectorAll('[data-metric="'+id+'"]').forEach(node=>{{
      if(text!=null)node.textContent=text;
    }});
    if(width!=null){{
      document.querySelectorAll('[data-metric="'+id+'-bar"]').forEach(node=>{{
        node.style.width=width;
      }});
    }}
  }};
  const shown=x.active_forced!=null?x.active_forced:(current?current.y:null);
  if(shown!=null)setMetric('ref-current',(shown*100).toFixed(1)+'%',(shown*100)+'%');
  const bm=x.benchmarks||{{}};
  const battery=bm.battery||{{}}; const bank=bm.banking77||{{}};
  [['qwen','ref-qwen'],['jev','ref-jev'],['baseline','ref-base'],['laya','ref-laya']].forEach(([key,id])=>{{
    const v=battery[key]; if(v!=null)setMetric(id,(v*100).toFixed(1)+'%',(v*100)+'%');
  }});
  [['jev_cited','ref-bank-jev'],['laya','ref-bank-laya'],['pointer','ref-bank-ours']].forEach(([key,id])=>{{
    const v=bank[key]; if(v!=null)setMetric(id,(v*100).toFixed(1)+'%',(v*100)+'%');
  }});
  document.querySelectorAll('.ref-list').forEach(list=>{{
    const rows=[...list.querySelectorAll('.ref-row')];
    const score=row=>{{const n=parseFloat(row.querySelector('.ref-score').textContent);return Number.isFinite(n)?n:-1;}};
    rows.sort((a,b)=>score(b)-score(a));
    rows.forEach(row=>list.appendChild(row));
    rows.forEach((row,i)=>{{
      const fill=row.querySelector('.ref-fill');
      fill.classList.remove('winner','ours');
      if(i===0&&score(row)>=0)fill.classList.add('winner');
      else if(row.dataset.role==='ours')fill.classList.add('ours');
    }});
  }});
  const fam=x.by_family||{{}}; const famKeys=Object.keys(fam).sort();
  $('fam-list').innerHTML=famKeys.length?famKeys.map(k=>{{
    const v=Number(fam[k])||0; const pctv=(v*100);
    const color=v>=0.70?'var(--acc)':v>=0.50?'var(--warn)':'var(--bad)';
    const label=k.replace(/_/g,' ');
    return `<div class="fam-row"><span class="fam-name">${{label}}</span><span class="fam-val">${{pctv.toFixed(1)}}%</span><div class="fam-track"><div class="fam-fill" style="width:${{pctv}}%;background:${{color}}"></div></div></div>`;
  }}).join(''):'<div class="fam-row"><span class="fam-name">waiting for evaluation</span><span class="fam-val">-</span></div>';
  $('spec-mem').textContent=(x.gpu&&x.gpu.mem_gb)?x.gpu.mem_gb+' GB in use':'48 GB';
  $('spec-model').textContent=x.model||'MiniLMv2-L6';
  $('spec-params').textContent=(x.params_m||107)+'M';
  $('spec-cycles').textContent=nfmt(x.completed_cycle);
  $('spec-holdout').textContent=fmt(x.holdout);
  $('spec-loss').textContent=fmt(x.loss);
  $('spec-producer').textContent=(x.producer&&x.producer.action)||'idle';
  const vols=h.vol||[]; const vmax=Math.max(1, ...vols.map(p=>p.y||0));
  plotWindow('line-vol', vols, 0, vmax*1.05, now, 86400, 900, 280);
  $('vol-max').textContent='total '+nfmt(vmax);
  plotWindow('line-gpu', (x.gpu&&x.gpu.spark)||[], 0, 100, now, 120, 900, 160);
  plotWindow('line-cpu', (x.cpu&&x.cpu.spark)||[], 0, 100, now, 120, 900, 160);
 }}catch(e){{$('live-state').textContent='NO HEARTBEAT';$('live-state').className='v delta-down'}}
}}
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{{document.querySelectorAll('nav button,.tab-page').forEach(x=>x.classList.remove('active'));b.classList.add('active');$(b.dataset.tab).classList.add('active');location.hash=b.dataset.tab}});
const initial=location.hash.slice(1);if(initial&&$(initial))document.querySelector(`[data-tab="${{initial}}"]`).click();
live();setInterval(live,1000);
</script></body></html>"""


def unseen_panel(state):
    """THE headline: the seen/unseen table of #T-unseen-labels.

    One row per cut, both halves side by side, plus the eval-only
    thermometer (LogiQA / ReClor) with its date and its `model_version`. A
    `pass: false` gate renders its table exactly the same way — the
    verdict pill says what failed, the numbers are never hidden or
    rounded in the product's favour.
    """
    u = state.get("unseen_gate")
    if not u:
        return ('<section class="panel"><h2>Primary evidence · UNSEEN labels '
                '(#T-unseen-labels)</h2><p>No <tt>artifacts/gates/'
                'T-unseen-labels/gate.json</tt> yet. Run '
                '<tt>.venv-train/bin/python -m eval.unseen gate '
                '--checkpoint &lt;ckpt&gt;</tt>. Until it exists, seen-label '
                'accuracy cannot support the product claim.</p>'
                '</section>')

    def num(v, nd=4):
        return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "?"

    ok = bool(u.get("pass"))
    pill = (f'<span class="pill {"on" if ok else "off"}">'
            f'{"GATE PASS" if ok else "GATE FAIL: " + ", ".join(u["failed"])}'
            f'</span>')
    overall = (u.get("headline") or {}).get("overall") or {}
    cuts = (u.get("headline") or {}).get("cuts") or {}

    def row(name, cut):
        table_cut = (u.get("table") or {}).get(name) or {}
        comp = ((table_cut.get("pair") or {}).get("comparability") or {})
        acc, ece, brier, n = (cut["accuracy"], cut["ece"], cut["brier"],
                              cut["n"])
        beats = ("yes" if cut.get("unseen_beats_chance") else "no")
        return (f'<tr><td>{name}</td>'
                f'<td class="num">{num(acc[0])}</td>'
                f'<td class="num">{num(acc[1])}</td>'
                f'<td class="num">{num(acc[0] - acc[1])}</td>'
                f'<td class="num">{num(ece[0])}</td>'
                f'<td class="num">{num(ece[1])}</td>'
                f'<td class="num">{num(brier[0], 3)}</td>'
                f'<td class="num">{num(brier[1], 3)}</td>'
                f'<td class="num">{n[0]:,} / {n[1]:,}</td>'
                f'<td class="num">{num(comp.get("mean_k_seen"), 2)} / '
                f'{num(comp.get("mean_k_unseen"), 2)}</td>'
                f'<td class="num">{beats}</td></tr>')

    body = "".join(row(k, v) for k, v in sorted(
        cuts.items(), key=lambda kv: (kv[0] != "ALL", kv[0])))
    erows = "".join(
        f'<tr><td>{name}</td><td class="num">{(r.get("n") or 0):,}</td>'
        f'<td class="num">{num((r.get("raw") or {}).get("accuracy"))}</td>'
        f'<td class="num">{num((r.get("raw") or {}).get("chance"), 3)}</td>'
        f'<td class="num">'
        f'{num((r.get("raw") or {}).get("accuracy_options_only"))}</td>'
        f'<td class="num">'
        f'{num((r.get("raw") or {}).get("chance_options_only"), 3)}</td>'
        f'<td class="num">{num((r.get("raw") or {}).get("ece"))}</td>'
        f'<td>{r.get("split", "?")}</td><td>{r.get("date", "?")}</td></tr>'
        for name, r in sorted((u.get("eval_only") or {}).items()))
    blocker = u.get("cross_lingual_blocker")
    blocker_html = (f'<p class="warn"><b>Cross-lingual arm not met</b>: '
                    f'{blocker}</p>' if blocker else "")
    cal = u.get("calibration") or {}
    return f'''<section class="panel"><h2>Primary evidence · accuracy and ECE on
 UNSEEN training labels (#T-unseen-labels)</h2>
<div class="cards">
<div class="card"><div class="k">unseen accuracy</div><div class="v">{num(overall.get("accuracy_unseen"))}</div><div class="s">seen {num(overall.get("accuracy_seen"))} · drop {num(overall.get("accuracy_drop"))}</div></div>
<div class="card"><div class="k">unseen ECE</div><div class="v">{num(overall.get("ece_unseen"))}</div><div class="s">seen {num(overall.get("ece_seen"))} · rise {num(overall.get("ece_rise"))}</div></div>
<div class="card"><div class="k">unseen Brier</div><div class="v">{num(overall.get("brier_unseen"), 3)}</div><div class="s">seen {num(overall.get("brier_seen"), 3)}</div></div>
<div class="card"><div class="k">beats chance</div><div class="v">{"yes" if overall.get("unseen_beats_chance") else "no"}</div><div class="s">ranking without unknown {"yes" if overall.get("unseen_ranking_beats_chance") else "no"}</div></div>
<div class="card"><div class="k">scored rows</div><div class="v">{(u.get("n_scored") or 0):,}</div><div class="s">{u.get("generated_utc", "?")}</div></div>
<div class="card"><div class="k">temperature</div><div class="v">{num(cal.get("temperature"), 3)}</div><div class="s">NLL {num(cal.get("nll_before"), 3)} to {num(cal.get("nll_after"), 3)} · n={cal.get("n_fit", "?")}</div></div>
</div>
<p>{pill} model_version <tt>{u.get("model_version") or "—"}</tt> ·
checkpoint <tt>{u.get("checkpoint") or "—"}</tt> ·
evidence <tt>{u.get("path")}</tt></p>
{blocker_html}
<table><tr><th>cut</th><th class="num">seen acc</th><th class="num">unseen acc</th>
<th class="num">drop</th><th class="num">seen ECE</th><th class="num">unseen ECE</th>
<th class="num">seen Brier</th><th class="num">unseen Brier</th><th class="num">n seen/unseen</th>
<th class="num">K seen/unseen</th><th class="num">&gt; chance</th></tr>
{body or '<tr><td colspan="11">No published cuts</td></tr>'}</table>
<h3>External thermometer · evaluation only, never training data (#T-halt-contam)</h3>
<table><tr><th>benchmark</th><th class="num">n</th><th class="num">accuracy</th><th class="num">chance</th>
<th class="num">ranking without unknown</th><th class="num">ranking chance</th><th class="num">ECE</th>
<th>split</th><th>date</th></tr>
{erows or '<tr><td colspan="9">No evaluation-only results</td></tr>'}</table>
<p><i>Seen and unseen cuts do not have equal difficulty by construction.
Both K and chance are published so the result remains auditable.</i></p>
</section>'''


def decision_panel(state):
    """The headline: unseen-label accuracy of the listwise run, or nothing.

    #I-honest-eval's primary metric is accuracy + ECE on labels never seen
    in training. Until a run publishes one, this panel says so instead of
    promoting a row-accuracy number to the headline (#T-halt-contam).
    """
    d = state.get("decision_run")
    if not d:
        return ('<section class="panel"><h2>Decision model · #T-train-real'
                '</h2><p>No listwise run yet. Start <tt>train-decision</tt>. '
                'The primary metric is accuracy on <b>unseen labels</b>, not '
                'row accuracy from the TF-IDF baseline.</p></section>')
    step = d.get("step") or {}
    stage = d.get("stage") or {}
    run = d.get("run") or {}
    gate = d.get("gate") or {}
    unseen = (stage.get("unseen") or {})
    seen = (stage.get("seen") or {})
    acc = unseen.get("accuracy")
    chance = unseen.get("chance")
    if acc is None:
        head_v, head_s = "measuring", "first stage not evaluated"
        pill = '<span class="pill off">no verdict</span>'
    else:
        head_v = f"{acc:.4f}"
        delta = (acc - chance) if chance is not None else None
        head_s = (f"chance {chance:.4f}"
                  + (f" · {'+' if delta >= 0 else ''}{delta:.4f}" if delta is not None else ""))
        ok = bool(gate.get("pass"))
        pill = (f'<span class="pill {"on" if ok else "off"}">'
                f'{gate.get("verdict", "no gate")}</span>')
    live = "RUNNING" if d.get("age_s", 1e9) < 300 else "stopped"
    def cell(v):
        return v if v is not None else "?"

    rows = "".join(
        f"<tr><td class=\"num\">{(c['samples'] or 0):,}</td>"
        f"<td class=\"num\">{(c['tokens'] or 0):,}</td>"
        f"<td class=\"num\">{cell(c['unseen'])}</td>"
        f"<td class=\"num\">{cell(c['chance'])}</td>"
        f"<td class=\"num\">{cell(c.get('rank'))}</td>"
        f"<td class=\"num\">{cell(c.get('rank_chance'))}</td>"
        f"<td class=\"num\">{cell(c.get('abstain'))}</td>"
        f"<td class=\"num\">{cell(c['unseen_ece'])}</td>"
        f"<td class=\"num\">{cell(c['seen'])}</td>"
        f"<td class=\"num\">{cell(c['seen_ece'])}</td></tr>"
        for c in d.get("curve", []))
    return f'''<section class="panel"><h2>Listwise scaling curve
 (#T-train-real)</h2>
<div class="cards">
<div class="card"><div class="k">unseen-label accuracy</div><div class="v">{head_v}</div><div class="s">{head_s}</div></div>
<div class="card"><div class="k">ranking without unknown</div><div class="v">{unseen.get("accuracy_options_only", "?")}</div><div class="s">chance {unseen.get("chance_options_only", "?")} · abstention {unseen.get("abstain_rate", "?")}</div></div>
<div class="card"><div class="k">unseen ECE</div><div class="v">{unseen.get("ece", "?")}</div><div class="s">Brier {unseen.get("brier", "?")}</div></div>
<div class="card"><div class="k">seen accuracy</div><div class="v">{seen.get("accuracy", "?")}</div><div class="s">ECE {seen.get("ece", "?")}</div></div>
<div class="card"><div class="k">samples · tokens</div><div class="v">{(step.get("samples") or 0):,}</div><div class="s">{(step.get("tokens") or 0):,} state tokens</div></div>
<div class="card"><div class="k">run listwise</div><div class="v">{live}</div><div class="s">{d.get("run_id", "?")} · {run.get("device", "?")}</div></div>
</div>
<p>{pill} loss <b>{run.get("architecture", {}).get("loss", "listwise CE")}</b> ·
one model over {", ".join(run.get("datasets", [])) or "the clean mixture"} ·
model_version <tt>{gate.get("model_version") or stage.get("model_version") or "—"}</tt> ·
latest step loss <b>{step.get("loss", "?")}</b> batch accuracy <b>{step.get("accuracy", "?")}</b>
ECE <b>{step.get("ece", "?")}</b> · {step.get("samples_per_s", "?")} samples/s</p>
<table><tr><th class="num">samples</th><th class="num">tokens</th><th class="num">unseen acc</th>
<th class="num">chance</th><th class="num">ranking without unknown</th><th class="num">ranking chance</th>
<th class="num">abstention</th><th class="num">unseen ECE</th><th class="num">seen acc</th>
<th class="num">seen ECE</th></tr>{rows or '<tr><td colspan="10">No stages yet</td></tr>'}</table>
</section>'''


def write_outputs(state, train_note="", train_live="", train_tail="",
                  cycle=0, tick=0, tick_ts=""):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / "state.json").write_text(json.dumps(state, indent=1))
    rows = "".join(
        f"<tr><td>{s['name']}</td><td class=\"num\">{s['examples']:,}</td>"
        f"<td class=\"num\">{s['size']}</td><td>{s['status']}</td></tr>"
        for s in state["datasets"])
    trows = "".join(
        f"<tr><td>{t['task']}</td><td class=\"num\">{t['n_train']:,}</td>"
        f"<td class=\"num\">{t['flag'] or t['accuracy']}</td>"
        f"<td class=\"num\">{t['seconds']}</td></tr>"
        for t in state["per_task"])
    procs = ", ".join(f"{k}={'ON' if v else 'off'}" for k, v in state["procs"].items())
    g = state.get("growth", {})
    rate = (f"{g.get('synth_per_min'):,}/min" if g.get("synth_per_min") is not None
            else "measuring (first 30 s)")
    gen = "ON" if state.get("procs", {}).get("gen_loop") else "off"
    qwen_rows = "".join(
        f"<tr><td>{q['name']}</td><td class=\"num\">{q['rows']:,}</td>"
        f"<td class=\"num\">{q['size']}</td><td>{q['mtime']}</td></tr>"
        for q in state.get("qwen_sets", []))
    frozen_banner = ("<p><b>synth-loop frozen in quarantine "
                       "(#T-halt-contam)</b>: contaminated and excluded from primary results.</p>"
                       if QUARANTINE.exists() else "")
    synth_sec = (
        frozen_banner +
        f"<p>synth-loop: <b>{state.get('synth_rows_live', 0):,}</b> rows on disk · "
        f"trained: <b>{state.get('synth_trained', 0):,}</b> · "
        f"rate ≈ <b>{rate}</b> "
        f"(+{g.get('synth_delta', 0):,} in {g.get('since_min', 0)} min) · "
        f"generator: <b>{gen}</b> · fresh Qwen templates: <b>{state.get('qwen_templates', 0)}</b></p>"
        f"<table><tr><th>Qwen set</th><th class=\"num\">rows</th><th class=\"num\">size</th><th>updated</th></tr>"
        f"{qwen_rows}"
        f"<tr><td><b>total Q-W-E-N</b></td><td class=\"num\"><b>{state.get('qwen_total_rows', 0):,}</b></td>"
        f"<td></td><td></td></tr></table>")
    gt = state.get("gen_train")
    fh = state.get("forward_hist") or []
    if QUARANTINE.exists():
        # The "1.000 → 1.000 → 1.000" headline measured leakage, not
        # capacity (#T-halt-contam, hallazgo C). It does not render again.
        fwd_sec = (
            f"<p>Forward test (synth-loop): <b>frozen</b>. "
            f"{quarantine_rows():,} quarantined rows "
            f"(190 skeletons, <tt>i%10</tt> split, 1.0000 leaked accuracy). "
            f"No trend is published until #T-gen-schemas provides a clean "
            f"template/domain split. The corrected split exists in "
            f"<tt>eval/splits.py</tt> (#T-split-domain); honest accuracy is in "
            f"<tt>artifacts/gates/T-split-domain/contrast.json</tt>.</p>")
    elif gt:
        age_m = gt["age_s"] / 60.0
        acc = gt.get("accuracy")
        acc_s = f"{acc:.4f}" if isinstance(acc, (int, float)) else "?"
        trend = " → ".join(
            f"{r.get('accuracy'):.3f}" if isinstance(r.get("accuracy"), (int, float)) else "?"
            for r in fh) or "no history yet"
        fwd_sec = (
            f"<p>Forward test (Qwen/synth, retained split): latest "
            f"<b>accuracy={acc_s}</b> over <b>{(gt.get('n_eval') or 0):,}</b> unseen examples "
            f"({(gt.get('n_train') or 0):,} trained) · {age_m:.0f} min ago · "
            f"run {gt['ts']}</p>"
            f"<p>Trend: <b>{trend}</b> (each retrain evaluates new unseen data)</p>")
    else:
        fwd_sec = ("<p>Forward test (Qwen/synth): no generator retrains yet.</p>")
    inv = state.get("invalidated_numbers") or {}
    invalid_sec = (
        f'<p class="warn"><b>INVALID HISTORICAL RESULTS '
        f'({inv.get("on", INDEX_SPLIT_INVALID_ON)})</b>: '
        f'{", ".join(inv.get("tasks", sorted(INDEX_SPLIT_TASKS)))}: '
        f'{inv.get("reason", INDEX_SPLIT_REASON)}. Those figures measured '
        f'leakage, not capability, and are not published. Correct grouped split: '
        f'<tt>eval/splits.py</tt>. Evidence: <tt>{inv.get("evidence", SPLIT_EVIDENCE)}'
        f'</tt> (#T-split-domain).</p>')
    import html as _html
    pr = state.get("procs", {})
    research = _research_benchmarks()
    battery = research.get("battery") or {}
    banking = research.get("banking77") or {}

    def pct_text(value):
        return (f"{value * 100:.2f}%"
                if isinstance(value, (int, float)) else "N/A")

    def pct_width(value):
        return (f"{float(value) * 100:.1f}%"
                if isinstance(value, (int, float)) else "0%")

    (STATE_DIR / "index.html").write_text(DASH_TMPL.format(
        fwd_sec=fwd_sec, decision_sec=decision_panel(state),
        unseen_sec=unseen_panel(state),
        ts=state["ts"], trained=state["trained_examples"], total=state["total_examples"],
        pct=state["pct_trained"], pending=state["pending_examples"], size=state["total_size"],
        params=state["baseline_params_est"], run=state["last_run"], rows=rows,
        trows=trows, procs=procs, note=train_note, synth_sec=synth_sec,
        invalid_sec=invalid_sec,
        train_live=_html.escape(train_live), train_tail=_html.escape(train_tail),
        cycle=cycle, tick=tick, tick_ts=tick_ts,
        synth_live=f"{state.get('synth_rows_live', 0):,}",
        synth_tr=f"{state.get('synth_trained', 0):,}",
        rate=rate, qwen_tot=f"{state.get('qwen_total_rows', 0):,}",
        qwen_tpl=state.get('qwen_templates', 0),
        gen_state="ON" if pr.get("gen_loop") else "off",
        oll_state="ON" if pr.get("ollama_11434") else "off",
        run_state="ON" if pr.get("ollama_runner_qwen") else "off",
        conv_state="ON" if pr.get("qwen_convert") else "off",
        battery_qwen=pct_text(battery.get("qwen")),
        battery_base=pct_text(battery.get("baseline")),
        battery_laya=pct_text(battery.get("laya")),
        battery_jev=pct_text(battery.get("jev")),
        bank_jev=pct_text(banking.get("jev_cited")),
        bank_laya=pct_text(banking.get("laya")),
        bank_pointer=pct_text(banking.get("pointer")),
        w_qwen=pct_width(battery.get("qwen")),
        w_jev=pct_width(battery.get("jev")),
        w_base=pct_width(battery.get("baseline")),
        w_laya=pct_width(battery.get("laya")),
        w_current=pct_width(state.get("forced")),
        w_bank_jev=pct_width(banking.get("jev_cited")),
        w_bank_laya=pct_width(banking.get("laya")),
        w_bank_ours=pct_width(banking.get("pointer")),
        train_short="RUNNING" if train_live.startswith("RUNNING") else "stopped"))


def train_status(state, trainer):
    """Combined train state for the dashboard card + terminal.

    Two trainers exist: the monitor's FULL retrain (train_baseline.py over
    everything, only fires on new prefetch data — usually 'parado' because
    the last run already covered the disk) and the generator's TARGETED
    retrain (TF-IDF+LogReg on Q-W-E-N/synth-loop every --retrain-every
    batches). The card used to show only the first, hence a confusing
    'parado' while Q-W-E-N data was really being trained every ~2 min."""
    if trainer.running():
        return (f"RUNNING pid={trainer.proc.pid} "
                f"elapsed={trainer.elapsed():.0f}s log={trainer.log.name}")
    gt = state.get("gen_train")
    gen_on = bool(state.get("procs", {}).get("gen_loop"))
    if gt and gen_on and gt["age_s"] < 15 * 60:
        acc = gt.get("accuracy")
        acc_s = f"{acc:.4f}" if isinstance(acc, (int, float)) else "?"
        return (f"RUNNING (Qwen/synth) · latest retrain "
                f"{gt['age_s'] / 60.0:.0f} min ago: n_train={(gt.get('n_train') or 0):,} "
                f"acc={acc_s} · full: {trainer.last_result}")
    if gt and gen_on:
        return (f"generator ON but latest retrain was {gt['age_s'] / 60.0:.0f} min ago "
                f"(generator may be paused) · full: {trainer.last_result}")
    if gen_on:
        return f"generator ON, no retrains yet · full: {trainer.last_result}"
    tail = f"stopped · {trainer.last_result} · generator off"
    if QUARANTINE.exists():
        tail += " · synth-loop quarantined (contaminated, excluded)"
    return tail


def print_block(state, train_note=""):
    print(f"=== training-monitor {state['ts']} ===", flush=True)
    u = state.get("unseen_gate")
    if u:
        o = (u.get("headline") or {}).get("overall") or {}
        print(f"TITULAR #T-unseen-labels: NO VISTAS acc={o.get('accuracy_unseen')} "
              f"ECE={o.get('ece_unseen')} Brier={o.get('brier_unseen')} | "
              f"VISTAS acc={o.get('accuracy_seen')} ECE={o.get('ece_seen')} | "
              f"caida={o.get('accuracy_drop')} supera_azar="
              f"{o.get('unseen_beats_chance')} | gate="
              f"{'PASS' if u.get('pass') else 'FAIL:' + ','.join(u.get('failed') or [])} "
              f"| model_version={u.get('model_version')}", flush=True)
        for name, cut in sorted(((u.get("headline") or {}).get("cuts")
                                 or {}).items()):
            acc, ece = cut["accuracy"], cut["ece"]
            print(f"  · {name:26s} acc {acc[0]} / {acc[1]}   "
                  f"ECE {ece[0]} / {ece[1]}   n {cut['n'][0]}/{cut['n'][1]}"
                  f"   >azar={cut.get('unseen_beats_chance')}", flush=True)
        for name, r in sorted((u.get("eval_only") or {}).items()):
            raw = r.get("raw") or {}
            print(f"  · eval-only {name:14s} n={r.get('n')} "
                  f"acc={raw.get('accuracy')} (azar {raw.get('chance')}) "
                  f"ranking={raw.get('accuracy_options_only')} "
                  f"ECE={raw.get('ece')} split={r.get('split')} "
                  f"fecha={r.get('date')}", flush=True)
    else:
        print("TITULAR #T-unseen-labels: sin gate.json todavía — este "
              "dashboard no tiene titular hasta que exista", flush=True)
    d = state.get("decision_run")
    if d:
        stage = d.get("stage") or {}
        unseen = stage.get("unseen") or {}
        seen = stage.get("seen") or {}
        step = d.get("step") or {}
        gate = d.get("gate") or {}
        print(f"escala #T-train-real ({d['run_id']}): acc etiquetas NO VISTAS="
              f"{unseen.get('accuracy')} (azar {unseen.get('chance')}) "
              f"ECE={unseen.get('ece')} ranking_s/unknown="
              f"{unseen.get('accuracy_options_only')} (azar "
              f"{unseen.get('chance_options_only')}) abstencion="
              f"{unseen.get('abstain_rate')} | vistas acc={seen.get('accuracy')} "
              f"ECE={seen.get('ece')} | muestras={step.get('samples')} "
              f"tokens={step.get('tokens')} | gate={gate.get('verdict', '—')}",
              flush=True)
    else:
        print("escala #T-train-real: sin run listwise todavía "
              "(job train-decision parado)", flush=True)
    print(f"datasets: {state['n_datasets']}  volumen: {state['total_size']} "
          f"({state['total_examples']:,} ejemplos según manifest)", flush=True)
    for s in state["datasets"]:
        print(f"  - {s['name']:20s} {s['examples']:>10,} ej  {s['size']:>9s}  {s['status']}",
              flush=True)
    print(f"ENTRENADO (run {state['last_run']}): {state['trained_examples']:,} ej  "
          f"({state['pct_trained']}%)  | PENDIENTE: {state['pending_examples']:,} ej",
          flush=True)
    g = state.get("growth", {})
    rsp = (f"{g.get('synth_per_min'):,}/min" if g.get("synth_per_min") is not None
           else "calculando…")
    print(f"Q-W-E-N/synth: synth-loop {state.get('synth_rows_live', 0):,} filas "
          f"(entrenadas {state.get('synth_trained', 0):,}) · Q-W-E-N {state.get('qwen_total_rows', 0):,} filas "
          f"en {len(state.get('qwen_sets', []))} sets · velocidad ≈ {rsp} "
          f"· gen_loop={state.get('procs', {}).get('gen_loop')}",
          flush=True)
    gt = state.get("gen_train")
    if gt:
        print(f"forward-test Q-W-E-N/synth: acc={gt.get('accuracy')} "
              f"n_train={gt.get('n_train'):,} n_eval={gt.get('n_eval'):,} "
              f"hace {gt['age_s'] / 60.0:.1f} min (run {gt['ts']})", flush=True)
    print(f"PARAMS baseline HISTORICO (TF-IDF 50k x clases, estimado): "
          f"{state['baseline_params_est']:,} — fuera de la ruta de producto "
          f"(#T-train-real)", flush=True)
    inv = state.get("invalidated_numbers") or {}
    print(f"NUMEROS HISTORICOS INVALIDOS ({inv.get('on')}): "
          f"{', '.join(inv.get('tasks', []))} — {inv.get('reason')} "
          f"(#T-split-domain, evidencia {inv.get('evidence')})", flush=True)
    for t in state["per_task"]:
        acc = t["flag"] or t["accuracy"]
        print(f"  - {t['task']:15s} n={t['n_train']:>7,} acc={acc} s={t['seconds']}",
              flush=True)
    print("procs: " + ", ".join(f"{k}={'ON' if v else 'off'}" for k, v in state["procs"].items()),
          flush=True)
    if train_note:
        print(f"train: {train_note}", flush=True)
    print(f"state: {STATE_DIR / 'state.json'}", flush=True)


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(STATE_DIR), **kw)

    def do_GET(self):
        if self.path.split("?", 1)[0] == "/api/live":
            body = json.dumps(live_snapshot(), ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, *a):
        pass


def serve(port):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--tick", type=int, default=20,
                    help="segundos entre latidos visibles (terminal+dashboard)")
    a = ap.parse_args()
    if a.port:
        serve(a.port)
        print(f"dashboard: http://127.0.0.1:{a.port}/ (sirviendo {STATE_DIR}/)",
              flush=True)
    print(f"train python: {train_python()}", flush=True)
    trainer = Trainer(STATE_DIR)
    cycle = 0
    while True:
        cycle += 1
        state = build_state()
        note = ""
        if a.train:
            try:
                started = trainer.maybe_start()
                if started:
                    print(f">>> {started}", flush=True)
                    note = started
                done = trainer.poll()
                if done:
                    print(f">>> {done}", flush=True)
                    note = done
                    # rescan: el run nuevo cambia los totales ya
                    state = build_state()
            except Exception as e:
                note = f"ERROR train: {e}"
        # stream del log de entrenamiento al terminal (movimiento real)
        for line in trainer.new_log_lines():
            print(f"[train] {line}", flush=True)
        live = train_status(state, trainer)
        tail_txt = "\n".join(trainer.tail(12))
        if not tail_txt and state.get("gen_train"):
            gt0 = state["gen_train"]
            tail_txt = (f"(sin full-train en esta sesión; último reentreno Q-W-E-N/synth: "
                        f"run {gt0['ts']} n_train={gt0.get('n_train')} "
                        f"acc={gt0.get('accuracy')} logloss={gt0.get('logloss')})")
        state["train_note"] = note or trainer.last_result
        state["train_live"] = live
        state["train_tail"] = tail_txt
        state["cycle"] = cycle
        tick_ts = utcnow()
        write_outputs(state, state["train_note"], live, tail_txt,
                      cycle, a.tick, tick_ts)
        print_block(state, state["train_note"])
        print(f"--- latido ciclo={cycle} {tick_ts} · {live} "
              f"(próximo en {a.tick}s) ---", flush=True)
        if a.once or a.interval <= 0:
            break
        # espera fraccionada: el entrenamiento corre en fondo, el latido no
        waited = 0
        step = a.tick if a.tick > 0 else a.interval
        while waited < a.interval:
            time.sleep(min(step, a.interval - waited))
            waited += step
            if waited >= a.interval:
                break
            # latido intermedio: reescanea, reescribe dashboard, sin bloquificar
            state = build_state()
            for line in trainer.new_log_lines():
                print(f"[train] {line}", flush=True)
            done = trainer.poll()
            if done:
                print(f">>> {done}", flush=True)
                state = build_state()
            live = train_status(state, trainer)
            tail_txt = "\n".join(trainer.tail(12))
            if not tail_txt and state.get("gen_train"):
                gt0 = state["gen_train"]
                tail_txt = (f"(sin full-train en esta sesión; último reentreno Q-W-E-N/synth: "
                            f"run {gt0['ts']} n_train={gt0.get('n_train')} "
                            f"acc={gt0.get('accuracy')} logloss={gt0.get('logloss')})")
            state["train_note"] = trainer.last_result
            state["train_live"] = live
            state["train_tail"] = tail_txt
            state["cycle"] = cycle
            tick_ts = utcnow()
            write_outputs(state, state["train_note"], live, tail_txt,
                          cycle, a.tick, tick_ts)
            print(f"--- latido ciclo={cycle} {tick_ts} · {live} ---", flush=True)


if __name__ == "__main__":
    main()
