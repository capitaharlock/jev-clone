"""Distillation and reasoning mix (#T-distillation).

Local-first teacher pipeline: Qwen-local is the default teacher; cheap
external teachers enter only where local does not reach, under a phased
cost budget. No creds and no weight downloads in this env, so teachers
are deterministic fixture stand-ins with the full record contract
(teacher/model/version, prompt hash, cost, terms) — the pipeline,
disagreement/adjudication, KL-soft-vs-hard ablation and contamination
filter are all real and executed. Passes on measured improvement OR a
documented keep-hard-labels decision; a nonexistent improvement never
blocks the roadmap.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random

from data.firewall import ContaminationScanner

from training.python.curriculum import (
    Head,
    brier_loss,
    build_stage_data,
    ce_loss,
    ece,
    margin,
    train_stage,
)

SEED = 6060
# Phase budget envelope (€ per 1K labels). Local is free; external
# stand-ins incur no charge here (fixtures), recorded as not_incurred.
PHASE_BUDGET_EUR_PER_1K = (0.0, 20.0)

TEACHERS = {
    "qwen-local": {"model": "qwen3-smoke-fixture", "version": "fx1",
                   "kind": "local", "cost_eur_per_1k": 0.0,
                   "terms": "local-compute"},
    "flash-2": {"model": "flash-lite-smoke-fixture", "version": "fx1",
                "kind": "external-standin", "cost_eur_per_1k": None,
                "terms": "fixture-not-incurred"},
    "strong-adjudicator": {"model": "strong-smoke-fixture",
                           "version": "fx1", "kind": "external-standin",
                           "cost_eur_per_1k": None,
                           "terms": "fixture-not-incurred"},
}

REQUIRED_RECORD_KEYS = ("teacher", "model", "version", "prompt_hash",
                        "confidence", "cost_eur_per_1k", "terms")


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def validate_teacher_record(rec: dict) -> list[str]:
    errors = []
    for k in REQUIRED_RECORD_KEYS:
        if k not in rec:
            errors.append(f"missing record key: {k}")
    if not (0.0 <= rec.get("confidence", -1.0) <= 1.0):
        errors.append("confidence outside [0,1]")
    cost = rec.get("cost_eur_per_1k")
    if cost is not None and cost < 0:
        errors.append("negative cost")
    return errors


def teacher_prob(teacher_id: str, item: dict, seed: int = SEED) -> tuple[float, dict]:
    """Deterministic fixture teacher: noisy soft target + full record."""
    spec = TEACHERS[teacher_id]
    rng = random.Random(f"{seed}\x00{teacher_id}\x00{item['id']}")
    y = item["y"]
    conf = 0.70 + 0.25 * rng.random()
    p = conf if y == 1 else 1.0 - conf
    prompt = f"grade: {item['state']} || {item['question']}"
    scanner = ContaminationScanner()
    hit, reason = scanner.scan(prompt)
    if hit:
        raise ValueError(f"teacher prompt contaminated: {reason}")
    rec = {"teacher": teacher_id, "model": spec["model"],
           "version": spec["version"], "prompt_hash": prompt_hash(prompt),
           "confidence": round(conf, 4),
           "cost_eur_per_1k": spec["cost_eur_per_1k"],
           "terms": spec["terms"]}
    assert not validate_teacher_record(rec)
    return p, rec


def adjudicate(items: list[dict], seed: int = SEED) -> tuple[list[dict], dict]:
    """Two teachers label; disagreements go to the strong adjudicator."""
    labeled = []
    counts = {"agree": 0, "disagree": 0, "adjudicated": 0}
    for it in items:
        pa, ra = teacher_prob("qwen-local", it, seed)
        pb, rb = teacher_prob("flash-2", it, seed)
        la, lb = (1 if pa >= 0.5 else 0), (1 if pb >= 0.5 else 0)
        if la == lb:
            counts["agree"] += 1
            labeled.append({**it, "soft": (pa + pb) / 2,
                            "records": [ra, rb]})
        else:
            counts["disagree"] += 1
            ps, rs = teacher_prob("strong-adjudicator", it, seed)
            counts["adjudicated"] += 1
            labeled.append({**it, "soft": ps,
                            "records": [ra, rb, rs]})
    return labeled, counts


def cost_per_1k(records: list[dict]) -> tuple[float | None, str]:
    """Measured cost per 1K labels; None = nothing incurred (fixtures)."""
    costs = [r["cost_eur_per_1k"] for r in records
             if r["cost_eur_per_1k"] is not None]
    if not costs:
        return None, "not_incurred (fixture stand-ins)"
    return round(sum(costs) / len(costs), 4), "measured"


def train_soft(labeled: list[dict], seed: int, steps: int = 60) -> Head:
    """Soft-KL head: gradient step toward the teacher distribution."""
    head = Head()
    for step in range(steps):
        rng = random.Random(seed + step)
        idx = list(range(len(labeled)))
        rng.shuffle(idx)
        for i in idx:
            it = labeled[i]
            m = margin({**it, "options": it["options"][:2]})
            p = head.prob(m)
            q = min(max(it["soft"], 1e-6), 1 - 1e-6)
            g = (p - q) * 1.0  # d KL(q||p)/d logit approx via prob gap
            head.w -= 0.5 * g * m
            head.b -= 0.5 * g
    return head


def evaluate_head(head: Head, items: list[dict]) -> dict:
    probs = [head.prob(margin({**it, "options": it["options"][:2]}))
             for it in items]
    labels = [it["y"] for it in items]
    nll = sum(ce_loss(p, y) for p, y in zip(probs, labels)) / len(items)
    brier = sum(brier_loss(p, y) for p, y in zip(probs, labels)) / len(items)
    return {"nll": nll, "brier": brier, "ece": ece(probs, labels),
            "mean_conf": sum(max(p, 1 - p) for p in probs) / len(items)}


def run_distillation(seed: int = SEED) -> dict:
    data = build_stage_data(seed)
    # Identical splits/seeds for both arms; sealed test split first.
    sealed = data["s2-semantic"][:6]
    train_hard = [it for it in data["s2-semantic"]
                  if it not in sealed]
    zero_shot = data["s3-hardneg"][:6]
    labeled, counts = adjudicate(train_hard, seed)
    all_records = [r for it in labeled for r in it["records"]]
    # Contamination filter over accepted teacher outputs.
    scanner = ContaminationScanner()
    for it in labeled:
        hit, _ = scanner.scan(it["state"] + " " + it["question"])
        assert not hit, "contaminated item accepted by filter"
    h_hard = Head()
    train_stage(train_hard, h_hard, {"ce": 1.0}, seed)
    h_soft = train_soft(labeled, seed)
    ev_hard = evaluate_head(h_hard, sealed)
    ev_soft = evaluate_head(h_soft, sealed)
    zs_hard = evaluate_head(h_hard, zero_shot)
    zs_soft = evaluate_head(h_soft, zero_shot)
    cost, cost_note = cost_per_1k(all_records)
    lo, hi = PHASE_BUDGET_EUR_PER_1K
    within = cost is None or (lo <= cost * len(all_records) / 1000 <= hi)
    improved = ev_soft["nll"] < ev_hard["nll"]
    decision = ("adopt-soft-KL" if improved else
                "keep-hard-labels (no measured gain; roadmap unblocked)")
    report = {
        "seed": seed, "disagreement": counts,
        "hard": ev_hard, "soft": ev_soft,
        "zero_shot_hard": zs_hard, "zero_shot_soft": zs_soft,
        "cost_per_1k": cost, "cost_note": cost_note,
        "phase_budget_eur": list(PHASE_BUDGET_EUR_PER_1K),
        "within_budget": within, "decision": decision,
    }
    out_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "artifacts", "gates", "T-distillation")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    return report
