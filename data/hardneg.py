"""Hard-negative and unknown engine (#T-hardneg).

Nearest-label sampler over label embeddings (in-domain + cross-dataset
hard negatives with a reproducible difficulty score) plus OOD
generators (missing-answer, unrelated, corrupt, contradictory) for
explicit `unknown` training. Every generated artifact is re-scanned by
the eval firewall before persist: an eval-only canary is rejected, not
stored.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random

from .firewall import ContaminationScanner, canary_texts

SEED = 442
DIM = 256


def _embed(text: str, dim: int = DIM) -> list[float]:
    v = [0.0] * dim
    t = f" {text.lower()} "
    for i in range(len(t) - 2):
        h = int(hashlib.sha256(t[i:i + 3].encode()).hexdigest(), 16)
        v[h % dim] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def pools_from_fixtures() -> dict[str, list[str]]:
    """Label pools per P0 dataset, read from checked-in fixtures."""
    base = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "artifacts", "fixtures", "p0")
    pools: dict[str, set[str]] = {}
    for fn in sorted(os.listdir(base)):
        if not fn.endswith(".jsonl"):
            continue
        labels: set[str] = set()
        with open(os.path.join(base, fn)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                for key in ("label", "category"):
                    val = row.get(key)
                    if isinstance(val, str):
                        labels.add(val)
        pools[fn[:-len(".jsonl")]] = sorted(labels)
    return {k: v for k, v in pools.items() if v}


def nearest_labels(gold: str, pool: list[str], k: int,
                   dim: int = DIM) -> list[dict]:
    """Top-k nearest labels to gold (excluded), difficulty = cosine."""
    gv = _embed(gold, dim)
    ranked = sorted(((lbl, _cos(gv, _embed(lbl, dim))) for lbl in pool
                     if lbl != gold),
                    key=lambda t: -t[1])
    return [{"label": lbl, "difficulty": round(s, 6)}
            for lbl, s in ranked[:k]]


def sample_hardneg(gold: str, pools: dict[str, list[str]], own: str,
                   k_in: int = 3, k_cross: int = 2,
                   seed: int = SEED) -> dict:
    """In-domain + cross-dataset hard negatives, seeded and reproducible."""
    rng = random.Random(f"{seed}\x00{gold}\x00{own}")
    in_domain = nearest_labels(gold, pools.get(own, []), k_in + 2)[:k_in]
    cross = []
    others = sorted(d for d in pools if d != own)
    rng.shuffle(others)
    for d in others:
        for cand in nearest_labels(gold, pools[d], k_cross):
            cross.append({**cand, "from": d})
        if len(cross) >= k_cross:
            break
    return {"gold": gold, "in_domain": in_domain,
            "cross_dataset": cross[:k_cross],
            "seed": seed}


def gen_missing_answer(item: dict, seed: int) -> dict:
    rng = random.Random(seed)
    kept = [o for o in item["options"] if o["id"] != item["answer"]]
    rng.shuffle(kept)
    return {"kind": "missing-answer", "state": item["state"],
            "question": item["question"], "options": kept,
            "answer": "unknown", "seed": seed}


def gen_unrelated(item: dict, pool_texts: list[str], seed: int) -> dict:
    rng = random.Random(seed)
    return {"kind": "unrelated", "state": rng.choice(pool_texts),
            "question": item["question"], "options": item["options"],
            "answer": "unknown", "seed": seed}


def gen_corrupt(item: dict, seed: int) -> dict:
    rng = random.Random(seed)
    chars = list(item["state"])
    rng.shuffle(chars)
    cut = max(len(chars) * 3 // 4, 1)
    return {"kind": "corrupt", "state": "".join(chars[:cut]),
            "question": item["question"], "options": item["options"],
            "answer": "unknown", "seed": seed}


def gen_contradictory(item: dict, seed: int) -> dict:
    return {"kind": "contradictory",
            "state": item["state"] + " Actually, none of the above applies.",
            "question": item["question"], "options": item["options"],
            "answer": "unknown", "seed": seed}


OOD_GENERATORS = {"missing-answer": gen_missing_answer,
                  "unrelated": gen_unrelated,
                  "corrupt": gen_corrupt,
                  "contradictory": gen_contradictory}


def firewall_screen(items: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split generated items into (clean, rejected) via the firewall."""
    scanner = ContaminationScanner()
    clean, rejected = [], []
    for it in items:
        texts = [it.get("state", ""), it.get("question", "")]
        hits = []
        for t in texts:
            hit, reason = scanner.scan(t)
            if hit:
                hits.append(reason)
        (rejected if hits else clean).append(it)
    return clean, rejected


def _item_text(it: dict) -> str:
    return f"{it['state']} {it['question']}"


def unknown_metrics(known: list[dict], ood: list[dict],
                    threshold: float = 0.55) -> dict:
    """Smoke unknown-detection: top cosine below threshold -> unknown."""
    def verdict(it: dict) -> tuple[str, float]:
        ctx = _embed(_item_text(it))
        top = max(_cos(ctx, _embed(o["text"])) for o in it["options"])
        return ("unknown" if top < threshold else "known", top)
    tp = sum(verdict(o)[0] == "unknown" for o in ood)
    fp = sum(verdict(k)[0] == "unknown" for k in known)
    hi_conf_err = sum(1 for k in known
                      if verdict(k) == ("known", verdict(k)[1])
                      and verdict(k)[1] >= 0.9
                      and k.get("answer") == "unknown")
    prec = tp / max(tp + fp, 1)
    return {"unknown_recall": tp / max(len(ood), 1),
            "unknown_precision": prec,
            "high_confidence_errors": hi_conf_err,
            "threshold": threshold}


def _demo_items(seed: int) -> tuple[list[dict], list[str]]:
    rng = random.Random(seed)
    topics = ["harbor status", "ledger balance", "orbit window",
              "meadow survey", "signal check", "engine load"]
    items = []
    for i, t in enumerate(topics):
        opts = [{"id": f"o{j}", "text": f"{t} is {s}"} for j, s in
                enumerate(("open", "closed", "delayed"))]
        items.append({"state": f"notice: {t} is open",
                      "question": f"what is {t}?",
                      "options": opts, "answer": "o0",
                      "id": f"d{i:02d}"})
    return items, [f"unrelated memo {w}" for w in
                   rng.sample(["clouds", "rivers", "trains", "gardens",
                               "lanterns", "bridges"], 4)]


def run_hardneg(seed: int = SEED) -> dict:
    pools = pools_from_fixtures()
    assert pools, "P0 fixture pools must be non-empty"
    demo, memos = _demo_items(seed)
    # Sampler over the first pool with >= 4 labels.
    own = next(d for d in sorted(pools) if len(pools[d]) >= 4)
    gold = sorted(pools[own])[0]
    sampled = sample_hardneg(gold, pools, own, seed=seed)
    ood: list[dict] = []
    for i, it in enumerate(demo):
        ood.append(gen_missing_answer(it, seed + i))
        ood.append(gen_unrelated(it, memos, seed + 100 + i))
        ood.append(gen_corrupt(it, seed + 200 + i))
        ood.append(gen_contradictory(it, seed + 300 + i))
    clean, rejected = firewall_screen(ood)
    # The firewall must reject an eval-only canary before persist:
    # use a REAL canary text so the rejection is genuine, not staged.
    canary = {"kind": "probe", "state": canary_texts()[0],
              "question": "probe?", "options": [], "answer": "unknown",
              "seed": seed}
    _, canary_rej = firewall_screen([canary])
    metrics = unknown_metrics(demo, clean)
    diffs = [c["difficulty"] for c in sampled["in_domain"]]
    registry = {
        "revision": f"hardneg-r1-seed{seed}",
        "seed": seed, "pools": {k: len(v) for k, v in pools.items()},
        "sample": sampled,
        "ood_kinds": sorted({o["kind"] for o in clean}),
        "n_ood": len(clean), "n_rejected": len(rejected),
        "canary_rejected": len(canary_rej) == 1,
        "metrics": metrics,
        "difficulty_monotonic_note": (
            f"in-domain difficulties (near->far): {diffs}"),
    }
    assert registry["canary_rejected"], "eval-only canary must be rejected"
    out_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "artifacts", "gates", "T-hardneg")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ood_registry.json"), "w") as f:
        json.dump(registry, f, indent=2)
        f.write("\n")
    return registry
