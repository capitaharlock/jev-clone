"""Shared-state V1 architecture (#T-shared-state).

Decisive experiment (plan §174): encode the state ONCE, then fuse per
question (1 cross-attention layer + option pooling + pointer head),
versus a full-cross-encoder that re-encodes [state+question+options]
per question. V1 scope: `choice` only (`boolean` = binary choice).

Everything is stdlib-only with tiny seeded vectors: the point is the
architecture contract (encode-once, cache invalidation, sublinear Q
scaling, parity single/batch and cached/uncached), not backbone quality.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random

DIM = 64
SEED = 174
# Quality margin fixed BEFORE looking at test results.
QUALITY_MARGIN = 0.05
# Length axis (tokens, simulated) and evidence positions.
LENGTHS = (512, 2048, 4096, 8192)
POSITIONS = ("start", "middle", "end")

_encode_calls = {"state": 0}


def reset_counters() -> None:
    _encode_calls["state"] = 0


def _proj(seed: int, rows: int, cols: int) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.uniform(-0.5, 0.5) for _ in range(cols)]
            for _ in range(rows)]


_WQ = _proj(SEED + 1, DIM, DIM)
_WK = _proj(SEED + 2, DIM, DIM)
_WV = _proj(SEED + 3, DIM, DIM)


def _hash_vec(text: str) -> list[float]:
    v = [0.0] * DIM
    t = f" {text.lower()} "
    for i in range(len(t) - 2):
        h = int(hashlib.sha256(t[i:i + 3].encode()).hexdigest(), 16)
        v[h % DIM] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _matvec(m: list[list[float]], v: list[float]) -> list[float]:
    return [sum(a * b for a, b in zip(row, v)) for row in m]


def encode_state(text: str, version: int = 1) -> list[float]:
    """State encoder API. Called exactly once per state in V1 path."""
    _encode_calls["state"] += 1
    v = _hash_vec(f"v{version}:{text}")
    return _matvec(_WQ, v)


def encode_light(text: str) -> list[float]:
    return _hash_vec("q:" + text)


def _softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    s = sum(ex)
    return [x / s for x in ex]


def fuse(state_vec: list[float], q_vec: list[float],
         option_vecs: list[list[float]]) -> list[float]:
    """1 cross-attention layer + option pooling + pointer head."""
    q = _matvec(_WQ, q_vec)
    keys = [_matvec(_WK, o) for o in option_vecs]
    vals = [_matvec(_WV, o) for o in option_vecs]
    attn = _softmax([sum(a * b for a, b in zip(q, k)) / math.sqrt(DIM)
                     for k in keys])
    pooled = [0.0] * DIM
    for w, v in zip(attn, vals):
        for i in range(DIM):
            pooled[i] += w * v[i]
    fused = [s + p for s, p in zip(state_vec, pooled)]
    return [sum(f * o for f, o in zip(fused, o)) for o in option_vecs]


class StateCache:
    """State cache with version/hash invalidation."""

    def __init__(self) -> None:
        self._store: dict[str, list[float]] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(text: str, version: int) -> str:
        return hashlib.sha256(f"{version}\x00{text}".encode()).hexdigest()

    def get(self, text: str, version: int = 1) -> list[float]:
        k = self.key(text, version)
        if k in self._store:
            self.hits += 1
            return list(self._store[k])
        self.misses += 1
        vec = encode_state(text, version)
        self._store[k] = list(vec)
        return vec


def answer_shared(state_text: str, questions: list[dict],
                  cache: StateCache | None = None,
                  version: int = 1) -> list[list[float]]:
    """V1 path: one state encode, fuse per question. Returns prob lists."""
    sv = cache.get(state_text, version) if cache else encode_state(
        state_text, version)
    out = []
    for q in questions:
        qv = encode_light(q["question"])
        ov = [encode_light(o["text"]) for o in q["options"]]
        mask = [1] * len(ov)
        assert sum(mask) == len(q["options"])
        assert 2 <= len(ov) <= 32
        out.append(_softmax(fuse(sv, qv, ov)))
    return out


def answer_full_cross(state_text: str, questions: list[dict]) -> list[list[float]]:
    """Baseline: re-encode [state+question] jointly per question."""
    out = []
    for q in questions:
        sv = encode_state(f"{state_text} || {q['question']}")
        qv = encode_light(q["question"])
        ov = [encode_light(o["text"]) for o in q["options"]]
        out.append(_softmax(fuse(sv, qv, ov)))
    return out


_FILLER = ("the quick brown fox jumps over the lazy dog near the river "
           "bank at dawn ")


def build_long_state(tokens: int, position: str,
                     evidence: str) -> tuple[str, int]:
    words = _FILLER.split()
    need = max(tokens - len(evidence.split()), 0)
    fill = (words * (need // len(words) + 1))[:need]
    ev = evidence.split()
    if position == "start":
        body = ev + fill
    elif position == "end":
        body = fill + ev
    else:
        body = fill[:len(fill) // 2] + ev + fill[len(fill) // 2:]
    return " ".join(body), len(body)


def build_multiq(n_q: int, seed: int = SEED) -> tuple[str, list[dict]]:
    rng = random.Random(seed)
    topics = ["harbor", "ledger", "orbit", "meadow"]
    state = "port manifest: " + ", ".join(
        f"{t} {rng.choice(['open', 'closed', 'delayed'])}" for t in topics)
    questions = []
    for i in range(n_q):
        t = topics[i % len(topics)]
        status = next(s for s in ("open", "closed", "delayed")
                      if f"{t} {s}" in state)
        others = [s for s in ("open", "closed", "delayed") if s != status]
        opts = [{"id": "o0", "text": f"{t} is {status}"},
                {"id": "o1", "text": f"{t} is {others[0]}"}]
        questions.append({"id": f"q{i}", "question": f"status of {t}?",
                          "options": opts, "answer": "o0"})
    return state, questions


def accuracy(probs: list[list[float]], questions: list[dict]) -> float:
    hits = 0
    for p, q in zip(probs, questions):
        pred = q["options"][max(range(len(p)), key=lambda i: p[i])]["id"]
        hits += pred == q["answer"]
    return hits / len(questions)


def run_shared_state(seed: int = SEED) -> dict:
    reset_counters()
    # Multi-Q scaling: encode-once vs per-question.
    q_axis = [1, 2, 4, 8]
    q_curves = []
    for nq in q_axis:
        state, questions = build_multiq(nq, seed)
        reset_counters()
        ps = answer_shared(state, questions)
        shared_calls = _encode_calls["state"]
        reset_counters()
        pf = answer_full_cross(state, questions)
        full_calls = _encode_calls["state"]
        q_curves.append({
            "Q": nq, "shared_encodes": shared_calls,
            "full_encodes": full_calls,
            "acc_shared": accuracy(ps, questions),
            "acc_full": accuracy(pf, questions),
        })
        assert shared_calls == 1, "state must encode exactly once"
        assert full_calls == nq
    # Length/evidence-position stress on a fixed probe.
    state0, qs0 = build_multiq(2, seed)
    evidence = state0
    len_curves = []
    for n in LENGTHS:
        row: dict = {"tokens": n}
        for pos in POSITIONS:
            long_state, got = build_long_state(n, pos, evidence)
            assert got >= n
            reset_counters()
            probs = answer_shared(long_state, qs0)
            row[pos] = accuracy(probs, qs0)
            assert _encode_calls["state"] == 1
        len_curves.append(row)
    # Cache parity + invalidation.
    cache = StateCache()
    p1 = answer_shared(state0, qs0, cache)
    p2 = answer_shared(state0, qs0, cache)
    assert cache.hits >= 1
    for a, b in zip(p1, p2):
        assert all(abs(x - y) < 1e-9 for x, y in zip(a, b)), \
            "cache vs uncached must match"
    before = (cache.hits, cache.misses)
    answer_shared(state0, qs0, cache, version=2)
    assert (cache.hits, cache.misses) != before, "version bump must miss"
    # Batch parity: concatenated questions in one call vs split calls.
    reset_counters()
    pall = answer_shared(state0, qs0)
    reset_counters()
    p_half = [answer_shared(state0, qs0[:1]), answer_shared(state0, qs0[1:])]
    flat = [r for half in p_half for r in half]
    for a, b in zip(pall, flat):
        assert all(abs(x - y) < 1e-9 for x, y in zip(a, b))
    # V1 limit: longest length where every position stays within margin
    # of the 512 baseline.
    base = min(len_curves[0][p] for p in POSITIONS)
    limit = LENGTHS[0]
    for row in len_curves:
        if all(row[p] >= base - QUALITY_MARGIN for p in POSITIONS):
            limit = row["tokens"]
    report = {"seed": seed, "quality_margin": QUALITY_MARGIN,
              "q_curves": q_curves, "len_curves": len_curves,
              "v1_state_limit_tokens": limit,
              "cache": {"hits": cache.hits, "misses": cache.misses}}
    out_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "artifacts", "gates", "T-shared-state")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    return report
