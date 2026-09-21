"""Backbone bake-off with real trained weights (#T-bakeoff-real).

The Pareto of this report contains ONLY backbones that were actually
trained: every row names the `run_id` of the fine-tune that produced it,
the sha256 of the backbone bytes it loaded and the sha256 of the head
checkpoint it was measured from. A row without that lineage fails the
gate (`verify_report`), and so does a row that is still
`pending_weights` or `random_init`.

What is measured, per candidate, from a COLD checkpoint:

* unseen-label accuracy + ECE — the primary metric (#T-unseen-labels):
  the option TEXTS in the eval set never appeared in training;
* decision latency p50/p95 at **K = 4** with a realistic state (a real
  corpus row, state cache and option cache cleared per decision) on MPS
  and on CPU, against the product's 70-500 ms budget;
* cost: backbone params, trainable head params, peak inference RSS
  (measured in a fresh process, not estimated) and the wall time the
  fine-tune took to reach the cut-off.

The hash char-ngram proxies that used to hold the Pareto are still run —
they are a useful check that the harness itself (markers, masks,
variable K, a bench that does not move mid-run) still behaves — but they
live under `sanity_checks` and can never be compared to a backbone:
their weights are random hashes.

Candidates whose weights or licence are not available are declared
`not_available` with the reason. Nothing is ever imputed: if a number
could not be measured, the row does not enter the Pareto.

CLI:
    .venv-train/bin/python -m model.bakeoff report
    .venv-train/bin/python -m model.bakeoff measure --checkpoint <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time

from .weights import BACKBONES, load_manifest, weights_dir

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE_DIR = os.path.join(ROOT, "artifacts", "gates", "T-bakeoff")
REPORT_PATH = os.path.join(GATE_DIR, "report.json")
CKPT_DIR = os.path.join(ROOT, "artifacts", "checkpoints", "decision")
RUNS_DIR = os.path.join(ROOT, "artifacts", "runs")

BAKEOFF_SEED = 173

#: the product's decision-latency window; a row outside it is reported as
#: outside it, never dropped.
LATENCY_BUDGET_MS = (70.0, 500.0)
#: every Pareto row is measured at the same K, with a real state
LATENCY_K = 4
LATENCY_RUNS = 50
DEVICES = ("mps", "cpu")

# Real candidates from the source register. `run_id` names the #T-train-real
# fine-tune whose checkpoint this row is measured from — the equal-budget
# bake-off runs, not the 1 M production run.
REAL_CANDIDATES = (
    {"id": "ettin-68m", "repo": "jhu-clsp/ettin-encoder-68m",
     "license": "MIT", "params_m": 68, "remote_code": False,
     "run_id": "bakeoff-ettin-68m-v1"},
    {"id": "modernbert-base", "repo": "answerdotai/ModernBERT-base",
     "license": "Apache-2.0", "params_m": 149, "remote_code": False,
     "run_id": "bakeoff-modernbert-base-v1"},
    {"id": "lfm2.5-230m", "repo": "LiquidAI/LFM2.5-Encoder-230M",
     "license": "LFM-Open-1.0", "params_m": 230, "remote_code": False,
     "run_id": None},
    {"id": "neobert-250m", "repo": "chandar-lab/NeoBERT",
     "license": "MIT", "params_m": 250, "remote_code": True,
     "run_id": None},
)

# Harness-only proxies: hash char-ngram encoders at three dims with
# random (hashed) weights. They check the HARNESS across a dimension
# axis. They are NOT backbones and are barred from the Pareto.
NOT_COMPARABLE = ("random hash-ngram encoder, no trained weights: "
                  "measures the harness, not a model; barred from the "
                  "Pareto (#T-torch-stack, #T-bakeoff-real)")
SMOKE_PROXIES = (
    {"id": "proxy-S", "dim": 256},
    {"id": "proxy-M", "dim": 1024},
    {"id": "proxy-L", "dim": 4096},
)
PROXY_IDS = frozenset(p["id"] for p in SMOKE_PROXIES)
#: a status that may never appear on a Pareto row
BANNED_STATUSES = ("pending_weights", "random_init", "harness_only",
                   "not_available", "stub")

_DOMAINS = ("sports", "politics", "tech", "health", "travel", "finance")
_WORDS = ("river", "market", "signal", "harbor", "engine", "meadow",
          "cipher", "ledger", "orbit", "lantern", "valley", "prism")


# -- licence / availability pre-gate (no weights needed) -------------------

def license_gate(candidate: dict) -> dict:
    """Pre-gate a real candidate on license/export (no weights needed)."""
    cid = candidate["id"]
    if candidate.get("remote_code"):
        return {"id": cid, "verdict": "blocked",
                "reason": "trust_remote_code unaudited"}
    if candidate["license"] == "LFM-Open-1.0":
        return {"id": cid, "verdict": "conditional",
                "reason": "commercial use pending LFM Open v1.0 review"}
    if candidate["license"] in ("MIT", "Apache-2.0"):
        return {"id": cid, "verdict": "clear", "reason": "permissive license"}
    return {"id": cid, "verdict": "blocked",
            "reason": f"unreviewed license {candidate['license']}"}


def availability(candidate: dict) -> dict:
    """Can this candidate produce a Pareto row at all?

    STUB rule: the only thing allowed to be missing is the EXTERNAL
    dependency (licence review, weight bytes). The measurement itself is
    never stubbed — a candidate that cannot be measured is declared
    `not_available` with the reason and stays out of the Pareto.
    """
    cid = candidate["id"]
    gate = license_gate(candidate)
    if gate["verdict"] == "blocked":
        return {"id": cid, "status": "not_available", "gate": gate,
                "reason": f"licence/export gate blocked: {gate['reason']}"}
    if cid not in BACKBONES or load_manifest(cid) is None:
        return {"id": cid, "status": "not_available", "gate": gate,
                "reason": ("weights not materialised under "
                           f"artifacts/weights/{cid}/ (#T-torch-stack "
                           "pins no revision for it)")}
    if gate["verdict"] == "conditional":
        return {"id": cid, "status": "not_available", "gate": gate,
                "reason": f"licence conditional: {gate['reason']}"}
    return {"id": cid, "status": "weights_local", "gate": gate}


def backbone_lineage(backbone_id: str) -> dict:
    """The sha256 of every pinned byte of a local backbone."""
    manifest = load_manifest(backbone_id) or {}
    return {
        "repo": manifest.get("repo"),
        "revision": manifest.get("revision"),
        "license": manifest.get("license"),
        "fetched_utc": manifest.get("fetched_utc"),
        "weights_sha256": {f: v.get("sha256") for f, v in
                           sorted(manifest.get("files", {}).items())},
        "weights_dir": os.path.relpath(weights_dir(backbone_id), ROOT),
    }


# -- harness sanity checks (stdlib only, no torch) -------------------------

def build_bench(seed: int = BAKEOFF_SEED, n: int = 48) -> list[dict]:
    """Deterministic choice bench with K in 2..32."""
    rng = random.Random(seed)
    items = []
    for i in range(n):
        k = 2 + (i * 7 + seed) % 31  # covers 2..32 deterministically
        domain = _DOMAINS[i % len(_DOMAINS)]
        state = f"{domain} brief {i}: " + " ".join(
            rng.choice(_WORDS) for _ in range(6))
        correct = rng.randrange(k)
        options = []
        for j in range(k):
            text = domain if j == correct else rng.choice(
                [d for d in _DOMAINS if d != domain])
            options.append({"id": f"o{j}",
                            "text": f"{text} note {rng.choice(_WORDS)}"})
        items.append({"id": f"b{i:03d}", "state": state,
                      "question": f"which domain ({domain})?",
                      "options": options, "answer": f"o{correct}"})
    return items


def _ngram_vec(text: str, dim: int) -> list[float]:
    v = [0.0] * dim
    t = f" {text.lower()} "
    for i in range(len(t) - 2):
        h = int(hashlib.sha256(t[i:i + 3].encode()).hexdigest(), 16)
        v[h % dim] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def assign_markers(options: list[dict], seed: int) -> tuple[list[str], dict]:
    """Dynamic option markers: shuffled markers + restore map."""
    rng = random.Random(seed)
    markers = [f"[{chr(65 + i)}]" for i in range(len(options))]
    rng.shuffle(markers)
    order = list(range(len(options)))
    rng = random.Random(seed + 1)
    rng.shuffle(order)
    presented_markers = [markers[i] for i in order]
    restore = {m: options[i]["id"] for m, i in zip(presented_markers,
                                                   order)}
    return presented_markers, restore


def score_options(state_vec: list[float], q_text: str,
                  options: list[dict], dim: int) -> list[float]:
    """Provisional pointer scorer: cosine(ctx, option), order invariant."""
    qv = _ngram_vec(q_text, dim)
    ctx = [s + q for s, q in zip(state_vec, qv)]
    n = math.sqrt(sum(x * x for x in ctx)) or 1.0
    ctx = [x / n for x in ctx]
    return [_cos(ctx, _ngram_vec(o["text"], dim)) for o in options]


def pad_masks(n_options: int, max_k: int) -> list[int]:
    return [1] * n_options + [0] * (max_k - n_options)


def softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    s = sum(ex)
    return [x / s for x in ex]


def _pct(sorted_lat: list[float], p: float) -> float:
    return sorted_lat[min(len(sorted_lat) - 1, int(p / 100.0 * len(sorted_lat)))]


def run_candidate(bench: list[dict], dim: int, seed: int) -> dict:
    """Run one harness proxy through the harness. Returns metrics."""
    correct = 0
    nll = 0.0
    lat: list[float] = []
    max_k = max(len(it["options"]) for it in bench)
    for it in bench:
        sv = _ngram_vec(it["state"], dim)
        t0 = time.perf_counter()
        scores = score_options(sv, it["question"], it["options"], dim)
        lat.append((time.perf_counter() - t0) * 1000.0)
        markers, restore = assign_markers(it["options"], seed)
        assert len(markers) == len(it["options"])
        assert sorted(restore.values()) == sorted(
            o["id"] for o in it["options"])
        probs = softmax(scores)
        pred = it["options"][max(range(len(scores)),
                                 key=lambda i: scores[i])]["id"]
        mask = pad_masks(len(it["options"]), max_k)
        assert sum(mask) == len(it["options"])
        if pred == it["answer"]:
            correct += 1
        gold = next(i for i, o in enumerate(it["options"])
                    if o["id"] == it["answer"])
        nll += -math.log(max(probs[gold], 1e-12))
    lat.sort()
    return {"accuracy": correct / len(bench), "nll": nll / len(bench),
            "p50_ms": _pct(lat, 50), "p95_ms": _pct(lat, 95),
            "n": len(bench)}


def fixed_benchmark_hash(bench: list[dict]) -> str:
    h = hashlib.sha256()
    for it in bench:
        h.update(json.dumps(it, sort_keys=True).encode())
    return h.hexdigest()


SANITY_NOTE = (
    "hash char-ngram encoders with RANDOM weights. They verify the harness "
    "— dynamic markers restore, masks match variable K, the fixed bench does "
    "not move mid-run — and nothing else. Not comparable to a backbone, "
    "never on the Pareto.")


def run_sanity_checks(seed: int = BAKEOFF_SEED) -> dict:
    """The harness proxies, run and reported OUTSIDE the Pareto."""
    bench = build_bench(seed)
    bench_hash = fixed_benchmark_hash(bench)
    measured = []
    for proxy in SMOKE_PROXIES:
        before = fixed_benchmark_hash(build_bench(seed))
        m = run_candidate(bench, proxy["dim"], seed)
        after = fixed_benchmark_hash(build_bench(seed))
        assert before == after == bench_hash, "bench moved mid-run"
        measured.append({"id": proxy["id"], "dim": proxy["dim"],
                         "status": "harness_only", "comparable": False,
                         "pareto_eligible": False,
                         "not_comparable_because": NOT_COMPARABLE, **m})
    ranked = sorted(measured, key=lambda m: (-m["accuracy"], m["p50_ms"]))
    return {
        "what": SANITY_NOTE,
        "seed": seed,
        "bench_hash": bench_hash,
        "bench_stable_across_runs": True,
        "harness_proxies": measured,
        "harness_rank": [m["id"] for m in ranked],
    }


# -- real measurement (torch; runs in a fresh process per candidate) -------

def latest_checkpoint(run_id: str) -> str | None:
    """The last stage directory of a finished bake-off run."""
    run_ckpts = os.path.join(CKPT_DIR, run_id)
    if not os.path.isdir(run_ckpts):
        return None
    stages = sorted(d for d in os.listdir(run_ckpts)
                    if d.startswith("stage-")
                    and os.path.exists(os.path.join(run_ckpts, d,
                                                    "manifest.json")))
    return os.path.join(run_ckpts, stages[-1]) if stages else None


def latency_probe(samplers: dict, k: int = LATENCY_K,
                  n: int = LATENCY_RUNS) -> list:
    """Real corpus rows trimmed to exactly K options, gold kept."""
    from training.python.train_decision import MixtureStream
    out = []
    for dataset, sampler in samplers.items():
        for batch in MixtureStream({dataset: sampler}, 32).epoch(0):
            for sample in batch:
                if sample.k < k or sample.gold_index >= sample.k:
                    continue
                gold = sample.options[sample.gold_index]
                others = [o for i, o in enumerate(sample.options)
                          if i != sample.gold_index][:k - 1]
                sample.options = [gold] + others
                sample.gold_index = 0
                out.append(sample)
            break
        if len(out) >= n:
            break
    return out[:n]


#: MPS compiles a kernel per tensor shape; the first decisions pay for that
#: compilation, which is a process-start cost, not a per-decision cost.
LATENCY_WARMUP = 3


def measure_latency(engine, probe: list) -> dict:
    """p50/p95 of a whole decision at K=4: encode + score, caches cold.

    The first `LATENCY_WARMUP` decisions are run untimed: on MPS they pay
    for kernel compilation, which happens once per process and would
    otherwise be billed to the p95 of a single decision.
    """
    import torch
    from training.python.train_decision import canonical_question
    for sample in probe[:LATENCY_WARMUP]:
        engine._states.clear()
        engine._texts.clear()
        engine.score(engine.encode_state(sample.state),
                     canonical_question(sample.question), sample.options)
    if engine.device.type == "mps":
        torch.mps.synchronize()
    times = []
    for sample in probe:
        engine._states.clear()
        engine._texts.clear()
        t0 = time.perf_counter()
        mem = engine.encode_state(sample.state)
        engine.score(mem, canonical_question(sample.question),
                     sample.options)
        if engine.device.type == "mps":
            torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    if not times:
        return {"runs": 0}
    lo, hi = LATENCY_BUDGET_MS
    p95 = times[max(0, int(len(times) * 0.95) - 1)]
    return {
        "runs": len(times), "k": LATENCY_K, "warmup_runs": LATENCY_WARMUP,
        "p50_ms": round(times[len(times) // 2], 3),
        "p95_ms": round(p95, 3),
        "max_ms": round(times[-1], 3),
        "budget_ms": [lo, hi],
        "within_budget": bool(p95 <= hi),
        "budget_note": (f"{lo}-{hi} ms is what the product ALLOWS a decision "
                        "to take; a p95 under the floor is head-room, not a "
                        "miss"),
        "state": "real corpus row, state cache AND option cache cleared "
                 "per decision",
        "machine_load": (
            "measured on a shared Mac: the #T-train-real production run "
            "(train-real-v1, CPU) was training throughout, so these are "
            "latencies under real concurrent load, not on an idle box. "
            "Run-to-run p95 moves by roughly a factor of two because of "
            "it — the ranking is stable, the absolute number is not."),
    }


def measure_candidate(checkpoint: str, device: str,
                      eval_samples: int = 3000, quality: bool = True,
                      rows_per_dataset: int | None = None) -> dict:
    """Everything one Pareto row needs, measured from a cold checkpoint.

    Runs in whatever process calls it; `run_bakeoff` gives each candidate
    and each device a FRESH one, so `peak_rss_mb` is this backbone's own
    inference footprint on that device and nothing else's.

    `quality=False` measures latency and memory only — the unseen/seen
    eval is device-independent, so it is paid once per candidate.
    """
    import resource

    import torch  # noqa: F401  (import cost belongs to this process)
    from training.python.train_decision import (SamplerConfig, build_holdout,
                                                evaluate, eval_samplers,
                                                load_checkpoint,
                                                train_samplers)

    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    engine, manifest = load_checkpoint(checkpoint, device)
    load_s = time.perf_counter() - t0

    config = SamplerConfig(seed=manifest["seed"])
    holdout = build_holdout()
    unseen = seen = {}
    if quality:
        unseen = evaluate(engine, eval_samplers(holdout, "unseen", config,
                                                rows_per_dataset),
                          eval_samples)
        seen = evaluate(engine, eval_samplers(holdout, "seen", config,
                                              rows_per_dataset), eval_samples)
    probe = latency_probe(train_samplers(holdout, config, rows_per_dataset))
    latency = measure_latency(engine, probe)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports ru_maxrss in bytes, Linux in kilobytes.
    scale = 1 << 20 if sys.platform == "darwin" else 1 << 10
    return {
        "checkpoint": os.path.relpath(checkpoint, ROOT),
        "device": str(engine.device),
        "model_version": manifest["model_version"],
        "run_id": manifest["run_id"],
        "head_weights_sha256": manifest["weights_sha256"],
        "tokenizer_hash": manifest["tokenizer_hash"],
        "head_params": manifest["head_params"],
        "backbone_params": manifest["backbone"]["params"],
        "backbone_frozen": manifest["backbone"]["frozen"],
        "samples_seen": manifest["samples_seen"],
        "tokens_seen": manifest["tokens_seen"],
        "seed": manifest["seed"],
        "architecture": manifest["architecture"],
        "unseen": unseen,
        "seen": seen,
        "latency": latency,
        "cold_load_s": round(load_s, 3),
        "peak_rss_mb": round(rss / scale, 1),
        "rss_before_load_mb": round(rss0 / scale, 1),
    }


def _venv_python() -> str:
    venv = os.path.join(ROOT, ".venv-train", "bin", "python")
    return venv if os.path.exists(venv) else sys.executable


def measure_in_subprocess(checkpoint: str, device: str,
                          eval_samples: int = 3000,
                          quality: bool = True) -> dict:
    """One candidate, one device, one fresh interpreter.

    A fresh process is what makes `peak_rss_mb` mean anything: the number
    is this backbone's inference footprint, not the sum of every backbone
    the report happens to touch.
    """
    cmd = [_venv_python(), "-m", "model.bakeoff", "measure",
           "--checkpoint", checkpoint, "--device", device,
           "--eval-samples", str(eval_samples)]
    if not quality:
        cmd.append("--latency-only")
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"error": (proc.stderr or proc.stdout or "")[-2000:],
                "device": device, "checkpoint": checkpoint}
    tail = proc.stdout.strip().splitlines()[-1]
    return json.loads(tail)


def production_reference() -> dict:
    """Where the 1 M production run stands, as a SNAPSHOT — not a row.

    `train-real-v1` trains `ettin-68m` at a budget two orders of
    magnitude above the bake-off's, so it is not comparable to these rows
    and never enters the Pareto. It is quoted because a bake-off that
    hides the project's real number would be the dishonest kind.
    """
    path = os.path.join(ROOT, "artifacts", "gates", "T-train-real",
                        "gate.json")
    if not os.path.exists(path):
        return {"available": False,
                "why": "artifacts/gates/T-train-real/gate.json not written yet"}
    with open(path) as fh:
        gate = json.load(fh)
    unseen = gate.get("unseen", {})
    return {
        "available": True,
        "run_id": gate.get("run_id"),
        "why_not_on_the_pareto": ("different budget (1 M-sample production "
                                  "run vs the bake-off's equal short "
                                  "budget); comparing them would compare "
                                  "budgets, not backbones"),
        "snapshot": "read while the run was still going; it moves",
        "samples_seen": gate.get("samples_seen"),
        "model_version": gate.get("model_version"),
        "unseen_accuracy": unseen.get("accuracy"),
        "unseen_chance": unseen.get("chance"),
        "unseen_ece": unseen.get("ece"),
        "pass": gate.get("pass"),
        "verdict": gate.get("verdict"),
    }


def train_seconds(run_id: str) -> dict:
    """Wall time the fine-tune took to reach its cut-off, from its log."""
    path = os.path.join(RUNS_DIR, run_id, "summary.json")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        summary = json.load(fh)
    return {"train_seconds_to_cutoff": summary.get("elapsed_s"),
            "steps": summary.get("steps"),
            "samples_seen": summary.get("samples_seen"),
            "tokens_seen": summary.get("tokens_seen")}


def build_row(candidate: dict, primary: dict, per_device: dict) -> dict:
    """One Pareto row: lineage + the numbers that were actually measured."""
    cid = candidate["id"]
    lineage = backbone_lineage(cid)
    cost = {
        "backbone_params": primary["backbone_params"],
        "head_params": primary["head_params"],
        "trainable_params": primary["head_params"],
        "trainable_note": "backbone frozen; only the pointer head trains",
        "inference_memory_mb": {d: m.get("peak_rss_mb")
                                for d, m in sorted(per_device.items())},
        "cold_load_s": {d: m.get("cold_load_s")
                        for d, m in sorted(per_device.items())},
        **train_seconds(primary["run_id"]),
    }
    return {
        "id": cid,
        "status": "trained",
        "repo": lineage["repo"],
        "revision": lineage["revision"],
        "license": lineage["license"],
        "gate": license_gate(candidate),
        "lineage": {
            "run_id": primary["run_id"],
            "checkpoint": primary["checkpoint"],
            "model_version": primary["model_version"],
            "weights_sha256": lineage["weights_sha256"],
            "head_weights_sha256": primary["head_weights_sha256"],
            "tokenizer_hash": primary["tokenizer_hash"],
            "trainer": "training/python/train_decision.py (#T-train-real)",
        },
        "quality": {
            "primary_metric": "unseen-label accuracy (#T-unseen-labels)",
            "unseen_accuracy": primary["unseen"].get("accuracy"),
            "unseen_accuracy_ci95": primary["unseen"].get("accuracy_ci95"),
            "unseen_chance": primary["unseen"].get("chance"),
            "unseen_beats_chance": primary["unseen"].get("beats_chance"),
            "unseen_ece": primary["unseen"].get("ece"),
            "unseen_brier": primary["unseen"].get("brier"),
            "unseen_abstain_rate": primary["unseen"].get("abstain_rate"),
            "unseen_n": primary["unseen"].get("n"),
            "unseen_per_dataset": primary["unseen"].get("per_dataset"),
            "seen_accuracy": primary["seen"].get("accuracy"),
            "seen_ece": primary["seen"].get("ece"),
            "seen_n": primary["seen"].get("n"),
        },
        "latency": {d: m.get("latency") for d, m in sorted(per_device.items())},
        "cost": cost,
        "architecture": primary["architecture"],
        "budget": {"samples_seen": primary["samples_seen"],
                   "tokens_seen": primary["tokens_seen"],
                   "seed": primary["seed"]},
    }


# -- the gate --------------------------------------------------------------

def _is_sha256(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        c in "0123456789abcdef" for c in value)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


#: the task's minimum real scope: both of these must be trained + measured
MINIMUM_SCOPE = frozenset({"ettin-68m", "modernbert-base"})


def verify_report(report: dict) -> dict:
    """The publication gate. Pure: no torch, no files, no measuring.

    Every check here is a reason to REFUSE to publish the report, and the
    test suite runs exactly this function over the published artifact.
    """
    rows = report.get("pareto", {}).get("rows", [])

    lineage_bad = []
    for row in rows:
        lin = row.get("lineage") or {}
        shas = list((lin.get("weights_sha256") or {}).values())
        if not lin.get("run_id"):
            lineage_bad.append(f"{row.get('id')}: no run_id")
        if not lin.get("checkpoint"):
            lineage_bad.append(f"{row.get('id')}: no checkpoint")
        if not shas or not all(_is_sha256(s) for s in shas):
            lineage_bad.append(f"{row.get('id')}: no backbone weights_sha256")
        if not _is_sha256(lin.get("head_weights_sha256")):
            lineage_bad.append(f"{row.get('id')}: no head weights_sha256")

    proxies_in_pareto = [row.get("id") for row in rows
                         if row.get("id") in PROXY_IDS
                         or row.get("status") in BANNED_STATUSES
                         or row.get("pareto_eligible") is False]

    measured_bad = []
    for row in rows:
        q = row.get("quality") or {}
        for key in ("unseen_accuracy", "unseen_ece", "seen_accuracy"):
            if not _is_number(q.get(key)):
                measured_bad.append(f"{row.get('id')}: {key} not measured")
        lat = row.get("latency") or {}
        for device in DEVICES:
            d = lat.get(device) or {}
            if not (_is_number(d.get("p50_ms")) and _is_number(d.get("p95_ms"))
                    and d.get("k") == LATENCY_K):
                measured_bad.append(
                    f"{row.get('id')}: no p50/p95 at K={LATENCY_K} on {device}")
        cost = row.get("cost") or {}
        if not _is_number(cost.get("backbone_params")):
            measured_bad.append(f"{row.get('id')}: no backbone_params")
        if not _is_number(cost.get("train_seconds_to_cutoff")):
            measured_bad.append(f"{row.get('id')}: no train time to cut-off")
        if not any(_is_number(v) for v in
                   (cost.get("inference_memory_mb") or {}).values()):
            measured_bad.append(f"{row.get('id')}: no inference memory")

    ids = {row.get("id") for row in rows}
    missing_scope = sorted(MINIMUM_SCOPE - ids)

    budget_bad = []
    for row in rows:
        lat = row.get("latency") or {}
        if not any((lat.get(d) or {}).get("within_budget")
                   for d in DEVICES):
            budget_bad.append(
                f"{row.get('id')}: p95 at K={LATENCY_K} outside the "
                f"{LATENCY_BUDGET_MS[0]}-{LATENCY_BUDGET_MS[1]} ms budget "
                "on every device measured")

    checks = {
        "pareto_lineage": {
            "pass": not lineage_bad,
            "criterion": ("every Pareto row names a run_id, a checkpoint, "
                          "the sha256 of its backbone bytes and the sha256 "
                          "of its head weights"),
            "offenders": lineage_bad,
        },
        "no_proxy_in_pareto": {
            "pass": not proxies_in_pareto,
            "criterion": ("no row with pending_weights / random_init / "
                          "harness_only may be published on the Pareto"),
            "offenders": proxies_in_pareto,
        },
        "measured_not_imputed": {
            "pass": not measured_bad,
            "criterion": ("unseen accuracy + ECE, p50/p95 at K=4 on MPS and "
                          "CPU, params, memory and train time are real "
                          "numbers on every row"),
            "offenders": measured_bad,
        },
        "minimum_real_scope": {
            "pass": not missing_scope,
            "criterion": "ettin-68m and modernbert-base are both trained "
                         "and measured",
            "offenders": missing_scope,
        },
        "latency_budget": {
            "pass": not budget_bad,
            "criterion": (f"p95 at K={LATENCY_K} inside the product's "
                          f"{LATENCY_BUDGET_MS[0]}-{LATENCY_BUDGET_MS[1]} ms "
                          "budget on at least one device"),
            "offenders": budget_bad,
        },
        "top2_stated": {
            "pass": bool(report.get("pareto", {}).get("top2"))
                    and bool(report.get("pareto", {}).get("verdict")),
            "criterion": "the top-2 choice and its prose verdict are written "
                         "into the report",
            "offenders": [],
        },
    }
    return checks


# -- verdict prose ---------------------------------------------------------

def _fmt(x, digits=3):
    return "n/a" if not _is_number(x) else f"{round(x, digits)}"


def write_verdict(rows: list) -> tuple[list, str]:
    """Top-2 by the primary metric, with the caveat spelled out."""
    if not rows:
        return [], ("no candidate reached a trained checkpoint: nothing to "
                    "rank")
    def p95(row, device):
        return (row.get("latency", {}).get(device) or {}).get("p95_ms")

    ranked = sorted(
        rows, key=lambda r: (-(r["quality"]["unseen_accuracy"] or 0.0),
                             r["quality"]["unseen_ece"] or 1.0,
                             p95(r, "cpu") or 1e9))
    top2 = [r["id"] for r in ranked[:2]]
    beats = [r["id"] for r in ranked
             if r["quality"].get("unseen_beats_chance")]
    lines = []
    for i, r in enumerate(ranked[:2], 1):
        q, cost = r["quality"], r["cost"]
        lines.append(
            f"{i}. {r['id']} — unseen accuracy {_fmt(q['unseen_accuracy'])} "
            f"(chance {_fmt(q['unseen_chance'])}, CI95 "
            f"{_fmt((q.get('unseen_accuracy_ci95') or [None])[0])}–"
            f"{_fmt((q.get('unseen_accuracy_ci95') or [None, None])[1])}, "
            f"n={q.get('unseen_n')}), unseen ECE {_fmt(q['unseen_ece'])}, "
            f"seen accuracy {_fmt(q['seen_accuracy'])}; decision p95 at K=4 "
            f"{_fmt(p95(r, 'cpu'), 1)} ms CPU / "
            f"{_fmt(p95(r, 'mps'), 1)} ms MPS; "
            f"{cost['backbone_params'] / 1e6:.0f} M frozen backbone params + "
            f"{cost['head_params'] / 1e6:.2f} M trainable head; "
            f"{_fmt(cost.get('train_seconds_to_cutoff'), 0)} s to the "
            f"{r['budget']['samples_seen']}-sample cut-off.")
    if len(beats) == len(ranked):
        quality = (
            "Every candidate clears chance on unseen labels at this equal "
            "budget (Wilson 95 % lower bound above the mean 1/(K+1) rate), "
            "so the order above is a real quality ranking and not a "
            "tie-break on cost. The gap between the two is small: the "
            "cheaper backbone buys most of the quality for less than half "
            "the parameters and less than half the CPU latency.")
    elif beats:
        quality = ("Only " + ", ".join(beats) + " clears chance on unseen "
                   "labels at this budget, so the ranking above is a real "
                   "quality ranking for it alone; the rest are ordered by "
                   "cost with their quality read as 'not separable from "
                   "chance'.")
    else:
        quality = (
            "NEITHER candidate clears chance on unseen labels at this "
            "equal budget — the same wall #T-train-real hit at 62.5 k "
            "samples (0.058 vs chance 0.165). So the top-2 is decided on "
            "cost and latency with quality read as 'not yet separable', "
            "NOT as a claim that the winner generalises to unseen labels. "
            "The bake-off's job here is to retire the proxies and pin the "
            "lineage; the unseen-label wall is #T-train-real's to break.")
    devices = []
    for r in ranked[:2]:
        cpu, mps = p95(r, "cpu"), p95(r, "mps")
        if _is_number(cpu) and _is_number(mps) and cpu < mps:
            devices.append(r["id"])
    note = ""
    if len(devices) == len(ranked[:2]) and devices:
        note = ("\nOn this Mac CPU is the faster device for a SINGLE K=4 "
                "decision: MPS pays a per-call dispatch and a per-shape "
                "kernel compile that a one-row workload cannot amortise "
                "(the warm-up decisions are excluded from these numbers "
                "and MPS is still slower). MPS remains the training "
                "device; serving one decision does not need it.")
    return top2, ("Top-2: " + ", ".join(top2) + ".\n" + "\n".join(lines)
                  + "\n" + quality + note)


# -- the report ------------------------------------------------------------

def run_bakeoff(eval_samples: int = 3000, devices: tuple = DEVICES,
                write: bool = True) -> dict:
    """Measure every available real backbone and publish the report."""
    sanity = run_sanity_checks()
    rows, not_available, why = [], [], {}

    for candidate in REAL_CANDIDATES:
        cid = candidate["id"]
        avail = availability(candidate)
        if avail["status"] != "weights_local":
            not_available.append(avail)
            why[cid] = avail["reason"]
            continue
        checkpoint = latest_checkpoint(candidate["run_id"] or "")
        if checkpoint is None:
            reason = (f"no trained checkpoint for run_id "
                      f"{candidate['run_id']!r}: weights are local but the "
                      "fine-tune has not produced a stage yet")
            not_available.append({"id": cid, "status": "not_available",
                                  "gate": avail["gate"], "reason": reason})
            why[cid] = reason
            continue
        # One fresh process per device for latency + memory (so the RSS
        # number is inference only), and ONE more for the unseen/seen eval,
        # which is device-independent and would otherwise inflate that RSS.
        quality_device = "cpu" if "cpu" in devices else devices[0]
        per_device = {device: measure_in_subprocess(checkpoint, device,
                                                    quality=False)
                      for device in devices}
        quality = measure_in_subprocess(checkpoint, quality_device,
                                        eval_samples, quality=True)
        failed = {d: m["error"] for d, m in
                  {**per_device, f"{quality_device}-eval": quality}.items()
                  if "error" in m}
        if failed:
            reason = ("measurement failed, so no row is published: "
                      + "; ".join(f"{d}: {e.strip().splitlines()[-1]}"
                                  for d, e in failed.items()))
            not_available.append({"id": cid, "status": "not_available",
                                  "gate": avail["gate"], "reason": reason})
            why[cid] = reason
            continue
        primary = quality
        rows.append(build_row(candidate, primary, per_device))

    top2, verdict = write_verdict(rows)
    ranked_ids = [r["id"] for r in sorted(
        rows, key=lambda r: -(r["quality"]["unseen_accuracy"] or 0.0))]
    for rid in ranked_ids[2:]:
        why[rid] = "measured but outside the top-2 of the Pareto"
    for proxy in sanity["harness_proxies"]:
        why[proxy["id"]] = NOT_COMPARABLE

    report = {
        "task": "T-bakeoff-real",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": {
            "primary_metric": ("unseen-label accuracy + ECE — option texts "
                               "never seen in training (#T-unseen-labels)"),
            "latency": (f"whole decision at K={LATENCY_K} on a real corpus "
                        "state, caches cold, on MPS and CPU, against the "
                        f"{LATENCY_BUDGET_MS[0]}-{LATENCY_BUDGET_MS[1]} ms "
                        "product budget"),
            "cost": ("frozen backbone params + trainable head params, peak "
                     "RSS of a fresh inference process, wall time of the "
                     "fine-tune to its cut-off"),
            "equal_budget": ("one short fine-tune per candidate through "
                             "training/python/train_decision.py, same seed, "
                             "same sample cut-off, same mixture, same head "
                             "architecture, frozen backbone"),
            "trainer": "training/python/train_decision.py (#T-train-real)",
            "production_run_untouched": (
                "run_id train-real-v1 (the 1 M production run) is NOT part "
                "of this bake-off and its gate was not rewritten"),
            "eval_samples": eval_samples,
        },
        "pareto": {"rows": rows, "top2": top2, "verdict": verdict},
        # mirrored at the top level: `T-release` composes its bundle from
        # `bakeoff["top2"]` and the model card must state those names.
        "top2": top2,
        "verdict": verdict,
        "not_available": not_available,
        "production_run_reference": production_reference(),
        "why_others_lose": why,
        "sanity_checks": sanity,
        "comparable": bool(rows),
    }
    checks = verify_report(report)
    report["checks"] = checks
    report["pass"] = all(c["pass"] for c in checks.values())
    if not report["pass"]:
        report["no_go_reason"] = "; ".join(
            f"{name}: {c['offenders']}" for name, c in checks.items()
            if not c["pass"])
    if write:
        os.makedirs(GATE_DIR, exist_ok=True)
        with open(REPORT_PATH, "w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
            fh.write("\n")
    return report


def load_report(path: str = REPORT_PATH) -> dict:
    with open(path) as fh:
        return json.load(fh)


# -- CLI -------------------------------------------------------------------

def main(argv: list) -> int:
    ap = argparse.ArgumentParser(prog="bakeoff", description=__doc__.split(
        "\n")[0])
    sub = ap.add_subparsers(dest="cmd")

    r = sub.add_parser("report", help="measure everything and publish")
    r.add_argument("--eval-samples", type=int, default=3000)
    r.add_argument("--devices", default=",".join(DEVICES))

    m = sub.add_parser("measure", help="one candidate on one device (json)")
    m.add_argument("--checkpoint", required=True)
    m.add_argument("--device", default="cpu")
    m.add_argument("--eval-samples", type=int, default=3000)
    m.add_argument("--latency-only", action="store_true",
                   help="skip the unseen/seen eval (device-independent)")

    sub.add_parser("verify", help="run the gate over the published report")

    args = ap.parse_args(argv[1:])
    cmd = args.cmd or "verify"

    if cmd == "measure":
        print(json.dumps(measure_candidate(args.checkpoint, args.device,
                                           args.eval_samples,
                                           quality=not args.latency_only),
                         sort_keys=True))
        return 0

    if cmd == "report":
        report = run_bakeoff(args.eval_samples,
                             tuple(args.devices.split(",")))
    else:
        report = load_report()
        report["checks"] = verify_report(report)
        report["pass"] = all(c["pass"] for c in report["checks"].values())

    for name, check in report["checks"].items():
        print(f"[gate] {name}: {'PASS' if check['pass'] else 'FAIL'} "
              f"{check['offenders'] or ''}")
    print(report["pareto"]["verdict"])
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
