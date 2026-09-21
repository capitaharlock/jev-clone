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
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREFETCH = ROOT / "artifacts" / "data-prefetch"
RUNS = ROOT / "artifacts" / "runs"
STATE_DIR = RUNS / "training-monitor"
LOG_DIR = ROOT / "artifacts" / "logs" / "monitor"
FEATS = 50000  # TfidfVectorizer max_features in data/train_baseline.py

# #T-halt-contam: synth-loop is quarantined (contaminated, out of the
# headline); logiqa/reclor are eval-only benchmarks, never train.
QUARANTINE = PREFETCH / "quarantine" / "synth-loop-20260921.jsonl"
CONTAMINATED_TASKS = {"synth-loop"}
EVAL_ONLY_TASKS = {"logiqa", "reclor"}


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
        self.last_result = "aún sin entrenar en esta sesión"
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
    synth_rows = next(
        (s.get("frozen_rows") or s["examples"] for s in sets
         if s["name"] == "synth-loop"), 0)
    synth_trained = next((j.get("n_train", 0) for j in jobs if j.get("task") == "synth-loop"), 0)
    grow = growth_rate(synth_rows, total_ex)
    pending = max(total_ex - trained, 0)
    pct = (100.0 * trained / total_ex) if total_ex else 0.0
    state = {
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
        # Flagged tasks (contaminated / eval-only) keep their history visible
        # but their accuracy is NOT shown as a number: no green without a
        # clean split behind it (#T-halt-contam).
        "per_task": [{"task": j.get("task"), "n_train": j.get("n_train"),
                      "accuracy": (None if j.get("task") in
                                   (CONTAMINATED_TASKS | EVAL_ONLY_TASKS)
                                   else (round(j.get("accuracy", 0), 4)
                                         if j.get("accuracy") else None)),
                      "flag": ("CONTAMINADA" if j.get("task") in CONTAMINATED_TASKS
                               else ("eval-only" if j.get("task") in EVAL_ONLY_TASKS
                                     else None)),
                      "seconds": j.get("seconds")} for j in jobs],
        "procs": {
            "ollama_11434": ollama_alive(),
            "ollama_runner_qwen": proc_alive("ollama runner"),
            "qwen_convert": proc_alive("qwen_convert.py"),
            "gen_loop": proc_alive("data_gen_loop.py"),
        },
    }
    return state


DASH_TMPL = """<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="15"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>JEv training monitor</title>
<style>
:root{{--bg:#0d1117;--panel:#161b22;--line:#30363d;--txt:#e6edf3;--dim:#8b949e;--acc:#2da44e;--warn:#d29922}}
*{{box-sizing:border-box}}body{{font-family:system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--txt);margin:0;padding:0 24px 40px}}
header.top{{display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px;padding:20px 4px 4px}}
header.top h1{{margin:0;font-size:22px}}header.top small{{color:var(--dim);font-weight:400}}
.live-dot{{display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--acc);margin-right:6px;box-shadow:0 0 8px var(--acc)}}
section.panel{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin-top:16px}}
section.panel h2{{margin:0 0 12px;font-size:14px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim)}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}}
.card{{background:#0d1117;border:1px solid var(--line);border-radius:8px;padding:10px 12px}}
.card .k{{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--dim)}}
.card .v{{font-size:20px;font-weight:650;margin-top:2px;font-variant-numeric:tabular-nums}}
.card .s{{font-size:12px;color:var(--dim);margin-top:2px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}@media(max-width:1100px){{.grid2{{grid-template-columns:1fr}}}}
table{{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}}
th,td{{border-bottom:1px solid var(--line);padding:7px 10px;text-align:left}}
th{{color:var(--dim);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.06em}}
td.num,th.num{{text-align:right}}tr:last-child td{{border-bottom:none}}
.bar{{background:#0d1117;border:1px solid var(--line);height:22px;border-radius:6px;overflow:hidden;margin-top:10px}}
.fill{{background:var(--acc);height:100%;border-radius:0;transition:width .5s}}
pre{{background:#0d1117;border:1px solid var(--line);border-radius:8px;padding:12px;overflow:auto;max-height:260px;font-size:12px;line-height:1.5}}
.pill{{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;border:1px solid var(--line)}}
.on{{color:#3fb950;border-color:#3fb950}}.off{{color:var(--dim)}}
footer{{color:var(--dim);font-size:12px;margin-top:16px}}
</style>
</head><body>
<header class="top"><h1><span class="live-dot"></span>JEv training monitor</h1><small>{ts} · ciclo {cycle} · latido {tick_ts} (cada {tick}s)</small></header>
<section class="panel"><h2>Q-W-E-N · entrenamiento en vivo (modelo local)</h2>
<div class="cards">
<div class="card"><div class="k">synth-loop en disco</div><div class="v">{synth_live}</div><div class="s">entrenadas {synth_tr} · {rate}/min</div></div>
<div class="card"><div class="k">total Q-W-E-N</div><div class="v">{qwen_tot}</div><div class="s">plantillas frescas {qwen_tpl}</div></div>
<div class="card"><div class="k">generador</div><div class="v">{gen_state}</div><div class="s">data_gen_loop.py</div></div>
<div class="card"><div class="k">ollama :11434</div><div class="v">{oll_state}</div><div class="s">runner {run_state} · convert {conv_state}</div></div>
<div class="card"><div class="k">train</div><div class="v">{train_short}</div><div class="s">{train_live}</div></div>
</div>
{synth_sec}
{fwd_sec}
</section>
<section class="panel"><h2>Totales</h2>
<div class="cards">
<div class="card"><div class="k">entrenados</div><div class="v">{trained:,}</div><div class="s">run {run}</div></div>
<div class="card"><div class="k">total ejemplos</div><div class="v">{total:,}</div><div class="s">volumen {size}</div></div>
<div class="card"><div class="k">pendiente</div><div class="v">{pending:,}</div><div class="s">{pct}% completado</div></div>
<div class="card"><div class="k">params baseline</div><div class="v">≈ {params:,}</div><div class="s">TF-IDF 50k × clases</div></div>
</div>
<div class="bar"><div class="fill" style="width:{pct}%"></div></div>
</section>
<div class="grid2">
<section class="panel"><h2>Datasets</h2><table><tr><th>dataset</th><th class="num">ejemplos</th><th class="num">tamaño</th><th>estado</th></tr>{rows}</table></section>
<section class="panel"><h2>Por tarea · último run</h2><table><tr><th>tarea</th><th class="num">n_train</th><th class="num">acc</th><th class="num">segs</th></tr>{trows}</table></section>
</div>
<section class="panel"><h2>Log de entrenamiento</h2><pre>{train_tail}</pre><p><span class="pill">{procs}</span></p><p><i>{note}</i></p></section>
<footer>este html se regenera en cada latido · sirve desde artifacts/runs/training-monitor/</footer></body></html>"""


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
            else "calculando… (primeros 30s)")
    gen = "ON" if state.get("procs", {}).get("gen_loop") else "off"
    qwen_rows = "".join(
        f"<tr><td>{q['name']}</td><td class=\"num\">{q['rows']:,}</td>"
        f"<td class=\"num\">{q['size']}</td><td>{q['mtime']}</td></tr>"
        for q in state.get("qwen_sets", []))
    frozen_banner = ("<p><b>synth-loop CONGELADA en cuarentena "
                       "(#T-halt-contam)</b> — contaminada, fuera del titular.</p>"
                       if QUARANTINE.exists() else "")
    synth_sec = (
        frozen_banner +
        f"<p>synth-loop (tu generador): <b>{state.get('synth_rows_live', 0):,}</b> filas en disco · "
        f"entrenadas: <b>{state.get('synth_trained', 0):,}</b> · "
        f"velocidad ≈ <b>{rate}</b> "
        f"(+{g.get('synth_delta', 0):,} en {g.get('since_min', 0)} min) · "
        f"generador: <b>{gen}</b> · plantillas frescas Q-W-E-N: <b>{state.get('qwen_templates', 0)}</b></p>"
        f"<table><tr><th>set Q-W-E-N</th><th class=\"num\">filas</th><th class=\"num\">tamaño</th><th>actualizado</th></tr>"
        f"{qwen_rows}"
        f"<tr><td><b>total Q-W-E-N</b></td><td class=\"num\"><b>{state.get('qwen_total_rows', 0):,}</b></td>"
        f"<td></td><td></td></tr></table>")
    gt = state.get("gen_train")
    fh = state.get("forward_hist") or []
    if QUARANTINE.exists():
        # The "1.000 → 1.000 → 1.000" headline measured leakage, not
        # capacity (#T-halt-contam, hallazgo C). It does not render again.
        fwd_sec = (
            f"<p>forward test (synth-loop): <b>congelado</b> — "
            f"{quarantine_rows():,} filas en cuarentena "
            f"(190 esqueletos, split <tt>i%10</tt>, acc 1,0000 por fuga). "
            f"Este panel no publica tendencia hasta que #T-gen-schemas aporte "
            f"un split limpio por plantilla/dominio.</p>")
    elif gt:
        age_m = gt["age_s"] / 60.0
        acc = gt.get("accuracy")
        acc_s = f"{acc:.4f}" if isinstance(acc, (int, float)) else "?"
        trend = " → ".join(
            f"{r.get('accuracy'):.3f}" if isinstance(r.get("accuracy"), (int, float)) else "?"
            for r in fh) or "sin historial aún"
        fwd_sec = (
            f"<p>forward test (Q-W-E-N/synth, split retenido): último "
            f"<b>acc={acc_s}</b> sobre <b>{(gt.get('n_eval') or 0):,}</b> ej no vistos "
            f"(entrenados {(gt.get('n_train') or 0):,}) · hace {age_m:.0f} min · "
            f"run {gt['ts']}</p>"
            f"<p>tendencia: <b>{trend}</b> (cada reentreno evalúa datos nuevos no vistos)</p>")
    else:
        fwd_sec = ("<p>forward test (Q-W-E-N/synth): aún sin reentrenos del generador "
                   "(arranca el generador para ver aquí acc sobre datos no vistos).</p>")
    import html as _html
    pr = state.get("procs", {})
    (STATE_DIR / "index.html").write_text(DASH_TMPL.format(
        fwd_sec=fwd_sec,
        ts=state["ts"], trained=state["trained_examples"], total=state["total_examples"],
        pct=state["pct_trained"], pending=state["pending_examples"], size=state["total_size"],
        params=state["baseline_params_est"], run=state["last_run"], rows=rows,
        trows=trows, procs=procs, note=train_note, synth_sec=synth_sec,
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
        train_short="EN MARCHA" if train_live.startswith("EN MARCHA") else "parado"))


def train_status(state, trainer):
    """Combined train state for the dashboard card + terminal.

    Two trainers exist: the monitor's FULL retrain (train_baseline.py over
    everything, only fires on new prefetch data — usually 'parado' because
    the last run already covered the disk) and the generator's TARGETED
    retrain (TF-IDF+LogReg on Q-W-E-N/synth-loop every --retrain-every
    batches). The card used to show only the first, hence a confusing
    'parado' while Q-W-E-N data was really being trained every ~2 min."""
    if trainer.running():
        return (f"EN MARCHA pid={trainer.proc.pid} "
                f"elapsed={trainer.elapsed():.0f}s log={trainer.log.name}")
    gt = state.get("gen_train")
    gen_on = bool(state.get("procs", {}).get("gen_loop"))
    if gt and gen_on and gt["age_s"] < 15 * 60:
        acc = gt.get("accuracy")
        acc_s = f"{acc:.4f}" if isinstance(acc, (int, float)) else "?"
        return (f"EN MARCHA (Q-W-E-N/synth) · último reentreno hace "
                f"{gt['age_s'] / 60.0:.0f} min: n_train={(gt.get('n_train') or 0):,} "
                f"acc={acc_s} · full: {trainer.last_result}")
    if gt and gen_on:
        return (f"generador ON pero último reentreno hace {gt['age_s'] / 60.0:.0f} min "
                f"(¿generador suspendido?) · full: {trainer.last_result}")
    if gen_on:
        return f"generador ON, aún sin reentrenos · full: {trainer.last_result}"
    tail = f"parado · {trainer.last_result} · generador off"
    if QUARANTINE.exists():
        tail += " · synth-loop en cuarentena (contaminada, fuera del titular)"
    return tail


def print_block(state, train_note=""):
    print(f"=== training-monitor {state['ts']} ===", flush=True)
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
    print(f"PARAMS baseline (TF-IDF 50k x clases, estimado): {state['baseline_params_est']:,}",
          flush=True)
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
