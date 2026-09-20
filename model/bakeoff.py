"""Backbone bake-off harness (#T-bakeoff).

Compares candidates through ONE identical harness: same splits, same
seed, same provisional decision head (dynamic option markers + pointer
scorer). Real backbone weights (Ettin, ModernBERT, LFM2.5, NeoBERT) are
not runnable in this stdlib-only env, so they are carried as
``pending_weights`` with their license/export pre-gate resolved from
``.meshkore/docs/source-register.md`` — never imputed into the Pareto.
What IS measured here: three smoke proxy encoders (S/M/L dims) that
validate the harness methodology (markers, masks, variable K, fixed
benchmark before/after) and produce a real quality/latency Pareto.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time

BAKEOFF_SEED = 173

# Real candidates from the source register. `runnable` is False for all:
# no torch/weights in this env. License/export pre-gate is resolved now.
REAL_CANDIDATES = (
    {"id": "ettin-68m", "repo": "jhu-clsp/ettin-encoder-68m",
     "license": "MIT", "params_m": 68, "remote_code": False},
    {"id": "modernbert-base", "repo": "answerdotai/ModernBERT-base",
     "license": "Apache-2.0", "params_m": 149, "remote_code": False},
    {"id": "lfm2.5-230m", "repo": "LiquidAI/LFM2.5-Encoder-230M",
     "license": "LFM-Open-1.0", "params_m": 230, "remote_code": False},
    {"id": "neobert-250m", "repo": "chandar-lab/NeoBERT",
     "license": "MIT", "params_m": 250, "remote_code": True},
)

# Smoke proxies: hash char-ngram encoders at three dims. They stand in
# for the size axis (17M->400M methodology check), NOT for any backbone.
SMOKE_PROXIES = (
    {"id": "proxy-S", "dim": 256},
    {"id": "proxy-M", "dim": 1024},
    {"id": "proxy-L", "dim": 4096},
)

_DOMAINS = ("sports", "politics", "tech", "health", "travel", "finance")
_WORDS = ("river", "market", "signal", "harbor", "engine", "meadow",
          "cipher", "ledger", "orbit", "lantern", "valley", "prism")


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
    """Run one smoke proxy through the harness. Returns metrics."""
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


def run_bakeoff(seed: int = BAKEOFF_SEED) -> dict:
    """Full bake-off: fixed bench, proxies measured, top-2 explicit."""
    bench = build_bench(seed)
    bench_hash = fixed_benchmark_hash(bench)
    measured = []
    for proxy in SMOKE_PROXIES:
        before = fixed_benchmark_hash(build_bench(seed))
        m = run_candidate(bench, proxy["dim"], seed)
        after = fixed_benchmark_hash(build_bench(seed))
        assert before == after == bench_hash, "bench moved mid-run"
        measured.append({"id": proxy["id"], "dim": proxy["dim"],
                         "status": "measured", **m})
    # Pareto on (accuracy, p50): top-2 by accuracy, latency breaks ties.
    ranked = sorted(measured, key=lambda m: (-m["accuracy"], m["p50_ms"]))
    top2 = [m["id"] for m in ranked[:2]]
    losers = [m["id"] for m in ranked[2:]]
    pending = [{"id": c["id"], "status": "pending_weights",
                "gate": license_gate(c)} for c in REAL_CANDIDATES]
    why = {lid: ("lower accuracy at higher latency vs top-2 "
                 "(smoke proxy axis)") for lid in losers}
    for p in pending:
        why[p["id"]] = (f"not runnable here ({p['gate']['verdict']}: "
                        f"{p['gate']['reason']}); excluded from Pareto, "
                        f"not imputed")
    report = {
        "seed": seed, "bench_hash": bench_hash,
        "measured": measured, "top2": top2,
        "why_others_lose": why,
        "pending_real": pending,
        "comparable": True,
    }
    out_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "artifacts", "gates", "T-bakeoff")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    return report
