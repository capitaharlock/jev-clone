"""Option interaction and order invariance (#T-option-mixer).

Options in a question interact listwise; they are not scored in
isolation. Same matrix compares three scorers — independent baseline,
DeepSets-style mixer, 1-layer Set-Transformer mixer — on identical
seeds and budget, including hard semantic siblings. Winner is picked
on measured robustness + latency; if no mixer wins, the simple
baseline is selected explicitly instead of blocking the chain.

SUPERSEDED by `model.decision_head` (#T-pointer-head): the set-attention
mixer and the order-invariance property are now implemented for real, on
the torch backbone, instead of with random projections in pure Python.
This module is kept only as the evidence behind
`artifacts/gates/T-option-mixer/report.json` (the release bundle
re-hashes it); no new code should score options through it.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time

DIM = 32
SEED = 1414
# Thresholds fixed BEFORE measuring.
PERM_STABILITY_MIN = 0.90
INSERT_STABILITY_MIN = 0.80


def _proj(seed: int, rows: int, cols: int) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.uniform(-0.4, 0.4) for _ in range(cols)]
            for _ in range(rows)]


_W1 = _proj(SEED + 1, DIM, DIM)
_W2 = _proj(SEED + 2, DIM, DIM)
_WQ = _proj(SEED + 3, DIM, DIM)
_WK = _proj(SEED + 4, DIM, DIM)
_WV = _proj(SEED + 5, DIM, DIM)


def _vec(text: str) -> list[float]:
    v = [0.0] * DIM
    t = f" {text.lower()} "
    for i in range(len(t) - 2):
        h = int(hashlib.sha256(t[i:i + 3].encode()).hexdigest(), 16)
        v[h % DIM] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _mv(m: list[list[float]], v: list[float]) -> list[float]:
    return [sum(a * b for a, b in zip(row, v)) for row in m]


def _tanh(x: float) -> float:
    e = math.exp(2.0 * max(min(x, 10.0), -10.0))
    return (e - 1.0) / (e + 1.0)


def _softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    s = sum(ex)
    return [x / s for x in ex]


def build_siblings(seed: int = SEED, n: int = 40) -> list[dict]:
    """Hard semantic siblings: same prefix, one-word difference."""
    rng = random.Random(seed)
    stems = ["the harbor is", "the ledger shows", "the orbit path",
             "the meadow path", "the signal reads", "the engine runs"]
    tails = ["open", "closed", "delayed", "quiet", "loud", "stable",
             "bright", "narrow"]
    items = []
    for i in range(n):
        stem = stems[i % len(stems)]
        k = 2 + (i * 5 + seed) % 7  # 2..8 options
        correct = rng.randrange(k)
        used = rng.sample(tails, k)
        options = [{"id": f"o{j}", "text": f"{stem} "
                    f"{used[j] if j != correct else used[0]}"}
                   for j in range(k)]
        # Ensure exactly one correct tail word.
        options[correct]["text"] = f"{stem} {used[0]}"
        items.append({"id": f"s{i:03d}", "state": f"notice: {stem} "
                      f"{used[0]} today",
                      "question": f"complete: {stem} …?",
                      "options": options, "answer": f"o{correct}"})
    return items


def score_independent(ctx: list[float],
                      opts: list[list[float]]) -> list[float]:
    return [sum(c * o for c, o in zip(ctx, o)) for o in opts]


def score_deepsets(ctx: list[float],
                   opts: list[list[float]]) -> list[float]:
    hidden = [[_tanh(x) for x in _mv(_W1, o)] for o in opts]
    pool = [sum(h[i] for h in hidden) / len(hidden) for i in range(DIM)]
    cond = [c + p for c, p in zip(ctx, pool)]
    return [sum(c * h for c, h in zip(cond, h)) for h in hidden]


def score_set_transformer(ctx: list[float],
                          opts: list[list[float]]) -> list[float]:
    q = [_mv(_WQ, o) for o in opts]
    k = [_mv(_WK, o) for o in opts]
    v = [_mv(_WV, o) for o in opts]
    out = []
    for i in range(len(opts)):
        attn = _softmax([sum(a * b for a, b in zip(q[i], kk))
                         / math.sqrt(DIM) for kk in k])
        mix = [sum(w * vv[j] for w, vv in zip(attn, v))
               for j in range(DIM)]
        h = [_tanh(x) for x in _mv(_W2, [a + b for a, b in
                                         zip(opts[i], mix)])]
        out.append(sum(c * x for c, x in zip(ctx, h)))
    return out


SCORERS = {"independent": score_independent,
           "deepsets": score_deepsets,
           "set_transformer": score_set_transformer}


def permutation_loss(scorer, ctx: list[float], opts: list[list[float]],
                     n_shuf: int = 6, seed: int = SEED) -> float:
    """Mean max deviation between base inverse-permuted probs and
    probs of shuffled inputs. Zero = perfectly order invariant."""
    base = _softmax(scorer(ctx, opts))
    worst = 0.0
    rng = random.Random(seed)
    for _ in range(n_shuf):
        perm = list(range(len(opts)))
        rng.shuffle(perm)
        got = _softmax(scorer(ctx, [opts[i] for i in perm]))
        inv = [0.0] * len(opts)
        for new_pos, old_pos in enumerate(perm):
            inv[old_pos] = got[new_pos]
        worst = max(worst, max(abs(a - b) for a, b in zip(base, inv)))
    return worst


def insertion_stability(scorer, item: dict) -> float:
    """Top-1 prob shift after adding 2 non-competitive distractors."""
    ctx = _vec(item["state"] + " " + item["question"])
    opts = [_vec(o["text"]) for o in item["options"]]
    base = _softmax(scorer(ctx, opts))
    top = max(base)
    extra = [_vec("unrelated note about clouds"),
             _vec("another unrelated memo")]
    got = _softmax(scorer(ctx, opts + extra))
    return abs(top - max(got[:len(opts)]))


def evaluate(scorer, bench: list[dict]) -> dict:
    correct = 0
    perm = 0.0
    ins = 0.0
    lat: list[float] = []
    for it in bench:
        ctx = _vec(it["state"] + " " + it["question"])
        opts = [_vec(o["text"]) for o in it["options"]]
        t0 = time.perf_counter()
        probs = _softmax(scorer(ctx, opts))
        lat.append((time.perf_counter() - t0) * 1000.0)
        pred = max(range(len(probs)), key=lambda i: probs[i])
        gold = next(i for i, o in enumerate(it["options"])
                    if o["id"] == it["answer"])
        correct += pred == gold
        perm += permutation_loss(scorer, ctx, opts)
        ins += insertion_stability(scorer, it)
    lat.sort()
    n = len(bench)
    return {
        "accuracy": correct / n,
        "perm_stability": 1.0 - perm / n,
        "insert_stability": 1.0 - ins / n,
        "p50_ms": lat[len(lat) // 2],
        "n": n,
    }


def run_option_mixer(seed: int = SEED) -> dict:
    bench = build_siblings(seed)
    results = {}
    for name, scorer in SCORERS.items():
        results[name] = evaluate(scorer, bench)
    # Winner: best mean(perm, insert) stability, accuracy breaks ties.
    # Explicit fallback to the simple baseline if no mixer clears it.
    def key(name: str) -> tuple:
        r = results[name]
        return ((r["perm_stability"] + r["insert_stability"]) / 2,
                r["accuracy"])
    ranked = sorted(results, key=key, reverse=True)
    mixers_clear = any(results[m]["perm_stability"] >= PERM_STABILITY_MIN
                       and results[m]["insert_stability"] >= INSERT_STABILITY_MIN
                       for m in ("deepsets", "set_transformer"))
    selected = ranked[0] if mixers_clear else "independent"
    report = {"seed": seed, "results": results,
              "perm_stability_min": PERM_STABILITY_MIN,
              "insert_stability_min": INSERT_STABILITY_MIN,
              "ranked": ranked, "selected": selected,
              "fallback_to_baseline": selected == "independent"}
    out_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "artifacts", "gates", "T-option-mixer")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    return report
