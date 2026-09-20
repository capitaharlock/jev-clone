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
    pending = max(total_ex - trained, 0)
    pct = (100.0 * trained / total_ex) if total_ex else 0.0
    state = {
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
        "per_task": [{"task": j.get("task"), "n_train": j.get("n_train"),
                      "accuracy": round(j.get("accuracy", 0), 4) if j.get("accuracy") else None,
                      "seconds": j.get("seconds")} for j in jobs],
        "procs": {
            "ollama_11434": ollama_alive(),
            "ollama_runner_qwen": proc_alive("ollama runner"),
            "qwen_convert": proc_alive("qwen_convert.py"),
        },
    }
    return state


DASH_TMPL = """<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="15"><title>JEv training monitor</title>
<style>body{{font-family:system-ui,sans-serif;max-width:900px;margin:2em auto;padding:0 1em}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:4px 8px;text-align:left}}
.bar{{background:#eee;height:18px;border-radius:4px}}.fill{{background:#2a7;height:100%;border-radius:4px}}</style>
</head><body><h1>JEv training monitor <small>{ts}Z</small></h1>
<p><b>{trained:,}</b> / {total:,} ejemplos entrenados ({pct}%) · pendiente: <b>{pending:,}</b> ·
volumen: {size} · params baseline ≈ {params:,} · último run: {run}</p>
<div class="bar"><div class="fill" style="width:{pct}%"></div></div>
<h2>Datasets</h2><table><tr><th>dataset</th><th>ejemplos</th><th>tamaño</th><th>estado</th></tr>{rows}</table>
<h2>Por tarea (último run)</h2><table><tr><th>tarea</th><th>n_train</th><th>acc</th><th>segs</th></tr>{trows}</table>
<h2>Entrenamiento en vivo</h2><p>{train_live}</p><pre>{train_tail}</pre>
<h2>Procesos</h2><p>{procs}</p><p><i>{note}</i></p>
<p><small>monitor vivo: ciclo {cycle}, latido {tick_ts} (cada {tick}s) · este html se regenera en cada latido</small></p></body></html>"""


def write_outputs(state, train_note="", train_live="", train_tail="",
                  cycle=0, tick=0, tick_ts=""):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / "state.json").write_text(json.dumps(state, indent=1))
    rows = "".join(
        f"<tr><td>{s['name']}</td><td>{s['examples']:,}</td>"
        f"<td>{s['size']}</td><td>{s['status']}</td></tr>"
        for s in state["datasets"])
    trows = "".join(
        f"<tr><td>{t['task']}</td><td>{t['n_train']:,}</td>"
        f"<td>{t['accuracy']}</td><td>{t['seconds']}</td></tr>"
        for t in state["per_task"])
    procs = ", ".join(f"{k}={'ON' if v else 'off'}" for k, v in state["procs"].items())
    import html as _html
    (STATE_DIR / "index.html").write_text(DASH_TMPL.format(
        ts=state["ts"], trained=state["trained_examples"], total=state["total_examples"],
        pct=state["pct_trained"], pending=state["pending_examples"], size=state["total_size"],
        params=state["baseline_params_est"], run=state["last_run"], rows=rows,
        trows=trows, procs=procs, note=train_note,
        train_live=train_live, train_tail=_html.escape(train_tail),
        cycle=cycle, tick=tick, tick_ts=tick_ts))


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
    print(f"PARAMS baseline (TF-IDF 50k x clases, estimado): {state['baseline_params_est']:,}",
          flush=True)
    for t in state["per_task"]:
        print(f"  - {t['task']:15s} n={t['n_train']:>7,} acc={t['accuracy']} s={t['seconds']}",
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
        if trainer.running():
            live = (f"EN MARCHA pid={trainer.proc.pid} "
                    f"elapsed={trainer.elapsed():.0f}s log={trainer.log.name}")
        else:
            live = f"parado · {trainer.last_result}"
        tail_txt = "\n".join(trainer.tail(12))
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
            if trainer.running():
                live = (f"EN MARCHA pid={trainer.proc.pid} "
                        f"elapsed={trainer.elapsed():.0f}s log={trainer.log.name}")
            else:
                live = f"parado · {trainer.last_result}"
            tail_txt = "\n".join(trainer.tail(12))
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
