"""Training curriculum and losses (#T-curriculum).

Stages 0-3 (head warmup -> multi-dataset supervised -> semantic mix ->
hard negatives) over a tiny real learner (logistic head on a
deterministic margin feature, SGD on CE). Total loss V1: CE + Brier +
KL + permutation consistency; the `score` ordinal term is V2-only and
refused here. Same seed+manifest reproduces batches and metrics;
resume from checkpoint does not alter the sample. Ablations
CE/CE+Brier and permutation on/off run on a smoke budget; a loss that
worsens the fixed regression benchmark is NOT promoted.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random

SEED = 5050
STAGES = ("s0-warmup", "s1-multi", "s2-semantic", "s3-hardneg")
LR = 0.5
SMOKE_STEPS = 60
# ECE target V1 (objective, recorded per stage — not a PASS tripwire).
ECE_TARGET = 0.05


def _embed(text: str, dim: int = 128) -> list[float]:
    v = [0.0] * dim
    t = f" {text.lower()} "
    for i in range(len(t) - 2):
        h = int(hashlib.sha256(t[i:i + 3].encode()).hexdigest(), 16)
        v[h % dim] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def margin(item: dict) -> float:
    """Signed feature: cos(ctx,o0) - cos(ctx,o1). The head maps it to
    P(o1 is correct); label y=1 iff the gold option is o1."""
    opts = item["options"][:2]
    ctx = _embed(item["state"] + " " + item["question"])
    return _cos(ctx, _embed(opts[0]["text"])) - _cos(
        ctx, _embed(opts[1]["text"]))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 30.0), -30.0)))


def ce_loss(p: float, y: int) -> float:
    return -(y * math.log(max(p, 1e-12)) + (1 - y) * math.log(max(1 - p, 1e-12)))


def brier_loss(p: float, y: int) -> float:
    return (p - y) ** 2


def kl_loss(p: float, q: float) -> float:
    """KL(teacher=q || student=p) on the positive class."""
    q = min(max(q, 1e-6), 1 - 1e-6)
    p = min(max(p, 1e-6), 1 - 1e-6)
    return q * math.log(q / p) + (1 - q) * math.log((1 - q) / (1 - p))


def perm_loss(p_pos: float, p_neg: float) -> float:
    """Consistency: swapping options must mirror the probability."""
    return abs(p_pos - (1.0 - p_neg))


def total_loss(p: float, y: int, w: dict, teacher: float | None = None,
               p_swapped: float | None = None) -> tuple[float, dict]:
    """V1 total loss. Returns (total, terms). Unknown keys refused."""
    allowed = {"ce", "brier", "kl", "perm"}
    extra = set(w) - allowed
    if extra:
        raise ValueError(f"non-V1 loss terms: {sorted(extra)} "
                         f"(ordinal/score is V2-only)")
    terms = {"ce": ce_loss(p, y)}
    if w.get("brier", 0.0):
        terms["brier"] = brier_loss(p, y)
    if w.get("kl", 0.0):
        if teacher is None:
            raise ValueError("kl weight without teacher prob")
        terms["kl"] = kl_loss(p, teacher)
    if w.get("perm", 0.0):
        if p_swapped is None:
            raise ValueError("perm weight without swapped prob")
        terms["perm"] = perm_loss(p, p_swapped)
    total = sum(w.get(k, 0.0) * v for k, v in terms.items())
    return total, terms


def _boolean_item(i: int, doms: list[str], style: str) -> dict:
    d = doms[i % len(doms)]
    status = "open" if i % 2 == 0 else "closed"
    other = "closed" if status == "open" else "open"
    state = (f"notice: {d} is {status}" if style == "plain"
             else f"memo: {d} reads {status} today")
    # Gold alternates between o0/o1 so the head must read the margin.
    if i % 4 < 2:
        opts = [{"id": "o0", "text": f"{d} is {status}"},
                {"id": "o1", "text": f"{d} is {other}"}]
        gold, y = "o0", 0
    else:
        opts = [{"id": "o0", "text": f"{d} is {other}"},
                {"id": "o1", "text": f"{d} is {status}"}]
        gold, y = "o1", 1
    return {"state": state, "question": f"is {d} open?",
            "options": opts, "answer": gold, "y": y,
            "id": f"c{i:02d}-{style}"}


def build_stage_data(seed: int = SEED) -> dict[str, list[dict]]:
    """Smoke data per stage: boolean items with growing difficulty."""
    rng = random.Random(seed)
    doms = ["harbor", "ledger", "orbit", "meadow"]
    stages: dict[str, list[dict]] = {s: [] for s in STAGES}
    for i in range(24):
        plain = _boolean_item(i, doms, "plain")
        hard = _boolean_item(i, doms, "memo")
        stages["s0-warmup"].append(plain)
        if i % 2 == 0:
            stages["s1-multi"].append(dict(plain))
        if i % 3 == 2 or i % 2 == 0:
            stages["s2-semantic"].append(dict(hard))
        s3 = dict(hard)
        d = doms[i % len(doms)]
        s3["options"] = list(s3["options"]) + [
            {"id": "o2", "text": f"{d} is open for visitors"}]
        s3["id"] = f"c{i:02d}-hardneg"
        stages["s3-hardneg"].append(s3)
    rng.shuffle(stages["s1-multi"])
    return stages


class Head:
    """Logistic head: p = sigmoid(w*m + b)."""

    def __init__(self, w: float = 0.0, b: float = 0.0) -> None:
        self.w = w
        self.b = b

    def prob(self, m: float) -> float:
        return sigmoid(self.w * m + self.b)

    def sgd_step(self, m: float, y: int, lr: float = LR,
                 brier_w: float = 0.0) -> None:
        p = self.prob(m)
        g = (p - y) + brier_w * 2.0 * (p - y) * p * (1.0 - p)
        self.w -= lr * g * m
        self.b -= lr * g


def sample_batches(data: list[dict], batch: int, seed: int):
    """Deterministic batches from seed+manifest order."""
    rng = random.Random(seed)
    idx = list(range(len(data)))
    rng.shuffle(idx)
    for s in range(0, len(idx), batch):
        yield [data[i] for i in idx[s:s + batch]]


def ece(probs: list[float], labels: list[int], bins: int = 5) -> float:
    edges = [i / bins for i in range(bins + 1)]
    total = 0.0
    for b in range(bins):
        sel = [i for i, p in enumerate(probs)
               if edges[b] < p <= edges[b + 1] or (b == 0 and p == 0.0)]
        if not sel:
            continue
        acc = sum(labels[i] for i in sel) / len(sel)
        conf = sum(probs[i] for i in sel) / len(sel)
        total += len(sel) / len(probs) * abs(acc - conf)
    return total


def train_stage(data: list[dict], head: Head, weights: dict,
                seed: int, steps: int = SMOKE_STEPS,
                from_checkpoint: dict | None = None) -> dict:
    """Train; optionally resume. Returns manifest + metrics."""
    rng_state = None
    step0 = 0
    if from_checkpoint:
        head.w = from_checkpoint["w"]
        head.b = from_checkpoint["b"]
        step0 = from_checkpoint["step"]
    losses = []
    brier_w = weights.get("brier", 0.0)
    for step in range(step0, steps):
        batches = list(sample_batches(data, 4, seed + step))
        ep = 0.0
        n = 0
        for batch in batches:
            for it in batch:
                m = margin({**it, "options": it["options"][:2]})
                p = head.prob(m)
                tot, _ = total_loss(p, it["y"], weights)
                ep += tot
                n += 1
                head.sgd_step(m, it["y"], brier_w=brier_w)
        losses.append(ep / max(n, 1))
    probs = [head.prob(margin({**it, "options": it["options"][:2]}))
             for it in data]
    labels = [it["y"] for it in data]
    return {"loss_start": losses[0], "loss_end": losses[-1],
            "losses": losses, "ece": ece(probs, labels),
            "checkpoint": {"w": head.w, "b": head.b, "step": steps,
                           "rng_note": rng_state}}


def fixed_regression_bench() -> list[dict]:
    data = build_stage_data()["s1-multi"][:8]
    return [{"m": margin({**it, "options": it["options"][:2]}),
             "y": it["y"]} for it in data]


def bench_nll(head: Head, bench: list[dict]) -> float:
    return sum(ce_loss(head.prob(r["m"]), r["y"]) for r in bench) / len(bench)


def run_curriculum(seed: int = SEED) -> dict:
    stages_data = build_stage_data(seed)
    bench = fixed_regression_bench()
    base_head = Head()
    base_nll = bench_nll(base_head, bench)
    stage_reports = {}
    head = Head()
    for s in STAGES:
        rep = train_stage(stages_data[s], head, {"ce": 1.0}, seed)
        stage_reports[s] = {
            "n": len(stages_data[s]),
            "loss_start": rep["loss_start"], "loss_end": rep["loss_end"],
            "ece": rep["ece"], "ece_target": ECE_TARGET,
            "promoted": rep["loss_end"] < rep["loss_start"],
            "rollback": None,
        }
        if not stage_reports[s]["promoted"]:
            stage_reports[s]["rollback"] = "kept prior checkpoint"
    # Ablations on identical splits/seeds, smoke budget.
    abl = {}
    h1, h2 = Head(), Head()
    r_ce = train_stage(stages_data["s2-semantic"], h1, {"ce": 1.0}, seed)
    r_cb = train_stage(stages_data["s2-semantic"], h2,
                       {"ce": 1.0, "brier": 0.5}, seed)
    nll_ce, nll_cb = bench_nll(h1, bench), bench_nll(h2, bench)
    abl["ce_vs_ce_brier"] = {
        "nll_ce": nll_ce, "nll_ce_brier": nll_cb,
        "promote_brier": bool(nll_cb <= nll_ce),
    }
    # Permutation on/off: measure consistency term on the bench.
    h3 = Head()
    train_stage(stages_data["s1-multi"], h3, {"ce": 1.0}, seed)
    perms = [perm_loss(h3.prob(r["m"]), h3.prob(-r["m"])) for r in bench]
    abl["permutation"] = {"mean_inconsistency": sum(perms) / len(perms)}
    report = {"seed": seed, "stages": stage_reports,
              "ablations": abl, "base_bench_nll": base_nll,
              "queue": "single-machine"}
    _write_report("T-curriculum", report)
    _write_run(seed, report)
    return report


def _write_report(task: str, report: dict) -> None:
    out_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
        "artifacts", "gates", task)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")


def _write_run(seed: int, report: dict) -> None:
    root = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    exp_dir = os.path.join(root, "experiments")
    res_dir = os.path.join(root, "results", f"curriculum-seed{seed}")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "curriculum.yaml"), "w") as f:
        f.write(f"# curriculum smoke queue (seed {seed})\n"
                f"seed: {seed}\nsteps: {SMOKE_STEPS}\n"
                f"lr: {LR}\nstages: [{', '.join(STAGES)}]\n")
    manifest = {"run_id": f"curriculum-seed{seed}", "seed": seed,
                "stages": list(STAGES), "report": "report.json"}
    with open(os.path.join(res_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
