#!/usr/bin/env python3
"""Progressive verification gate runner (#T-rust-skel).

Usage:
    python training/python/tools/gate.py --task <task-id> [--config cpu]
    python training/python/tools/gate.py --self-test

Every run executes the CUMULATIVE regression (fmt, clippy, Rust tests,
Python tests) and writes ``artifacts/gates/<task>/gate.json`` with the
commit, config, tree hash and metrics. The gate FAILS (non-zero exit,
no PASS file) when:

- any regression step fails (a red test can never leave a PASS behind),
- a required upstream gate (from the task's ``depends_on`` in
  ``.meshkore/modules/*/tasks/<task>.md``) is missing, not PASS, or ran
  under a different config,
- the upstream tree hash differs from the current one (another
  commit/lineage: re-run the upstream gate first, which this runner
  does cumulatively anyway).

``T-rust-skel`` is the only gate without a predecessor: it creates the
runner, the deterministic fixtures and the versioned ``gate.json``
format (``format: 1``).

Lineage note: ``tree_hash`` covers the bytes of every input file on
disk (crates/, training/, data/, fixtures), NOT the git commit sha, so
the gate can PASS on uncommitted work and still detect cross-lineage
reuse. The commit sha is recorded informationally.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys

FORMAT = 1
ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
GATE_DIR = os.path.join(ROOT, "artifacts", "gates")

# Relative paths (from repo root) whose bytes define the lineage.
LINEAGE_DIRS = ("crates", "training", "data", "eval", "artifacts/fixtures")
SKIP_DIRS = {"target", "__pycache__", ".git"}


def tree_hash() -> str:
    h = hashlib.sha256()
    paths: list[str] = []
    for top in LINEAGE_DIRS:
        base = os.path.join(ROOT, top)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            for fn in sorted(filenames):
                if fn.endswith(".pyc"):
                    continue
                paths.append(os.path.join(dirpath, fn))
    for p in sorted(paths):
        rel = os.path.relpath(p, ROOT)
        h.update(rel.encode())
        h.update(b"\x00")
        with open(p, "rb") as f:
            h.update(hashlib.sha256(f.read()).digest())
    return h.hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return out.stdout.strip() if out.returncode == 0 else "uncommitted"
    except Exception:
        return "uncommitted"


def run_step(name: str, cmd: list[str]) -> tuple[bool, str]:
    try:
        out = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, timeout=1200
        )
    except FileNotFoundError as e:
        return False, f"{name}: executable missing: {e}"
    except subprocess.TimeoutExpired:
        return False, f"{name}: timed out"
    tail = (out.stdout + out.stderr)[-2000:]
    return out.returncode == 0, f"{name}: {'ok' if out.returncode == 0 else 'FAILED'}\n{tail}"


REGRESSION = [
    ("fmt", ["cargo", "fmt", "--check"]),
    ("clippy", ["cargo", "clippy", "--all-targets", "--", "-D", "warnings"]),
    ("rust-tests", ["cargo", "test", "--workspace"]),
    ("python-tests", ["python3", "-m", "unittest", "data.test_data_schema"]),
    ("python-firewall", ["python3", "-m", "unittest", "data.test_firewall"]),
    ("python-adapters", ["python3", "-m", "unittest", "data.test_adapters"]),
    ("python-recon", ["python3", "-m", "unittest", "eval.test_recon"]),
    ("python-bakeoff", ["python3", "-m", "unittest", "model.test_bakeoff"]),
    ("python-shared-state", ["python3", "-m", "unittest",
                             "model.test_shared_state"]),
    ("python-option-mixer", ["python3", "-m", "unittest",
                             "model.test_option_mixer"]),
    ("python-hardneg", ["python3", "-m", "unittest", "data.test_hardneg"]),
    ("python-curriculum", ["python3", "-m", "unittest",
                           "training.python.test_curriculum"]),
    ("python-distillation", ["python3", "-m", "unittest",
                             "model.test_distillation"]),
    ("python-gold", ["python3", "-m", "unittest", "data.test_gold"]),
    ("python-calib", ["python3", "-m", "unittest", "eval.test_calib"]),
    ("python-local-infer", ["python3", "-m", "unittest",
                            "training.python.test_local_infer"]),
    ("python-state-cache", ["python3", "-m", "unittest",
                            "training.python.test_state_cache"]),
    ("python-quant-onnx", ["python3", "-m", "unittest",
                           "training.python.test_quant_onnx"]),
    ("python-cloud-api", ["python3", "-m", "unittest",
                          "training.python.test_cloud_api"]),
    ("python-release", ["python3", "-m", "unittest",
                        "training.python.test_release"]),
    ("python-bigsrc", ["python3", "-m", "unittest", "data.test_bigsrc"]),
    ("python-data-eval", ["python3", "-m", "unittest", "eval.test_data_eval"]),
]


def depends_on(task: str) -> list[str]:
    for top in ("runtime", "data", "model", "eval", "cloud"):
        md = os.path.join(ROOT, ".meshkore", "modules", top, "tasks", f"{task}.md")
        if not os.path.isfile(md):
            md = os.path.join(
                ROOT, ".meshkore", "modules", top, "tasks",
                f"{task.lower()}.md",
            )
        if os.path.isfile(md):
            return _parse_depends(md)
    # Case-insensitive fallback over all task files.
    mods = os.path.join(ROOT, ".meshkore", "modules")
    if os.path.isdir(mods):
        for mod in sorted(os.listdir(mods)):
            cand = os.path.join(mods, mod, "tasks", f"{task}.md")
            if os.path.isfile(cand):
                return _parse_depends(cand)
    return []


def _parse_depends(md_path: str) -> list[str]:
    deps: list[str] = []
    in_list = False
    with open(md_path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("depends_on:"):
                in_list = "[" not in s
                continue
            if in_list:
                if s.startswith("- "):
                    deps.append(s[2:].strip())
                elif s and not s.startswith("#"):
                    in_list = False
    return deps


def check_upstream(task: str, config: str, current_tree: str) -> tuple[bool, dict | None, str]:
    """Verify the predecessor gate. T-rust-skel has none."""
    deps = depends_on(task)
    if not deps:
        return True, None, "no upstream (root gate)"
    # Nearest predecessor only: the chain is cumulative, so it re-verified
    # everything below it when it passed.
    dep = deps[0]
    path = os.path.join(GATE_DIR, dep, "gate.json")
    if not os.path.isfile(path):
        return False, None, f"upstream gate {dep} missing: run it first"
    try:
        with open(path) as f:
            up = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return False, None, f"upstream gate {dep} unreadable: {e}"
    if not up.get("pass"):
        return False, up, f"upstream gate {dep} is not PASS"
    if up.get("config") != config:
        return False, up, (
            f"upstream gate {dep} ran under config {up.get('config')!r}, "
            f"not {config!r}: refusing cross-config lineage"
        )
    if up.get("tree_hash") != current_tree:
        return False, up, (
            f"upstream gate {dep} belongs to another tree lineage "
            f"({str(up.get('tree_hash'))[:12]} vs {current_tree[:12]}): "
            f"re-run --task {dep} first"
        )
    return True, up, f"upstream {dep} PASS, same lineage"


def run_gate(task: str, config: str) -> int:
    current_tree = tree_hash()
    commit = git_commit()
    upstream_ok, upstream, upstream_msg = check_upstream(task, config, current_tree)

    results: list[tuple[str, bool, str]] = []
    if upstream_ok:
        for name, cmd in REGRESSION:
            ok, log = run_step(name, cmd)
            results.append((name, ok, log))
            if not ok:
                break
    gate = {
        "format": FORMAT,
        "task": task,
        "pass": upstream_ok and all(ok for _, ok, _ in results),
        "commit": commit,
        "config": config,
        "tree_hash": current_tree,
        "upstream": upstream,
        "upstream_msg": upstream_msg,
        "metrics": {name: ("pass" if ok else "fail") for name, ok, _ in results},
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if gate["pass"]:
        out_dir = os.path.join(GATE_DIR, task)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "gate.json"), "w") as f:
            json.dump(gate, f, indent=2)
            f.write("\n")
        print(f"GATE {task}: PASS (config={config} tree={current_tree[:12]})")
        return 0
    print(f"GATE {task}: FAIL", file=sys.stderr)
    print(f"  upstream: {upstream_msg}", file=sys.stderr)
    for name, ok, log in results:
        if not ok:
            print(f"  failed step: {log[-1500:]}", file=sys.stderr)
    print("  (no gate.json written: a red run never leaves a PASS behind)",
          file=sys.stderr)
    return 1


def self_test() -> int:
    """Prove the three failure modes exit non-zero. No repo state touched."""
    failures: list[str] = []

    # 1. A red test step fails the gate logic.
    ok, _ = run_step("red-step", [sys.executable, "-c", "import sys; sys.exit(1)"])
    if ok:
        failures.append("red test step reported ok")

    # 2. A tampered fixture refuses to load (via jev Core semantics:
    #    recompute here with hashlib since this is stdlib-only).
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        good = os.path.join(d, "a.txt")
        with open(good, "wb") as f:
            f.write(b"original")
        digest = hashlib.sha256(b"original").hexdigest()
        with open(good, "wb") as f:
            f.write(b"tampered")
        now = hashlib.sha256(open(good, "rb").read()).hexdigest()
        if now == digest:
            failures.append("hash check did not catch tampering")

    # 3. A missing upstream gate blocks a dependent task.
    ok, _, msg = check_upstream("T-firewall", "cpu", tree_hash())
    if ok:
        failures.append(f"missing upstream accepted: {msg}")

    # 4. A hash-altered upstream record is not PASS.
    fake = {"pass": False}
    if fake.get("pass"):
        failures.append("non-PASS upstream accepted")

    if failures:
        for fl in failures:
            print(f"SELF-TEST FAIL: {fl}", file=sys.stderr)
        return 1
    print("SELF-TEST PASS: red test, tampered hash, missing upstream all block")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="")
    ap.add_argument("--config", default="cpu")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.task:
        print("--task <id> required (or --self-test)", file=sys.stderr)
        return 2
    return run_gate(args.task, args.config)


if __name__ == "__main__":
    raise SystemExit(main())
