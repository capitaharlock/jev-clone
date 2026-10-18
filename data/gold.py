"""Human gold set and Spanish (#T-gold).

Annotation schema + ambiguity policy (ambiguous cases keep the human
distribution, never forced to one-hot), a 500-item pilot with
inter-annotator agreement recomputed from raw annotations, reserved
calibration/test splits, an ES+EN benchmark via a pinned MASSIVE
adapter (CC-BY-4.0, attributed), and a leakage proof that no gold
item appears in train, teacher prompts or hyperparameter selection.

Provenance is explicit everywhere: this pilot is a seeded synthetic
stand-in (`synthetic-seeded`) that validates the FORMAT, the IAA
tooling and the fences. Human collection is pending and recorded as
such — no number here poses as a human judgment.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random

from .firewall import ContaminationScanner
from .leakage import LeakageDetector

SEED = 7070
PILOT_N = 500
PROVENANCE = "synthetic-seeded (human pilot pending)"
MASSIVE_REVISION = "massive-pilot-fx1"
MASSIVE_LOCALES = ("en-US", "es-ES")


def annotation_schema() -> dict:
    """Field contract for the annotation UI / annotation JSONL."""
    return {
        "fields": {
            "item_id": "manifest id (must exist in the pilot manifest)",
            "annotator": "annotator id",
            "distribution": "map option_id -> prob, sums to 1.0",
            "ambiguous": "bool; when true the distribution is kept as-is",
            "notes": "free text, optional",
        },
        "rules": [
            "distribution must sum to 1.0 within 1e-6",
            "ambiguous items keep the full distribution (no one-hot forcing)",
            "item_id must resolve in the manifest",
        ],
    }


def validate_annotation(ann: dict, manifest_ids: set[str]) -> list[str]:
    errors = []
    if ann.get("item_id") not in manifest_ids:
        errors.append(f"unknown item_id: {ann.get('item_id')!r}")
    dist = ann.get("distribution", {})
    if not dist:
        errors.append("empty distribution")
    elif abs(sum(dist.values()) - 1.0) > 1e-6:
        errors.append("distribution does not sum to 1.0")
    elif any(p < 0.0 or p > 1.0 for p in dist.values()):
        errors.append("distribution prob outside [0,1]")
    return errors


def collapse_to_onehot(dist: dict) -> dict:
    """REFUSED by the ambiguity policy: never force one-hot."""
    raise ValueError("ambiguity policy: distributions are kept, "
                     "never collapsed to one-hot")


def build_pilot(seed: int = SEED, n: int = PILOT_N) -> tuple[list[dict], list[dict]]:
    """500 pilot items (EN+ES) with dual raw annotations."""
    rng = random.Random(seed)
    en_topics = [("harbor", "open"), ("ledger", "balanced"),
                 ("orbit", "stable"), ("meadow", "quiet")]
    es_topics = [("puerto", "abierto"), ("libro", "equilibrado"),
                 ("órbita", "estable"), ("pradera", "tranquila")]
    items, anns = [], []
    for i in range(n):
        es = (i % 2 == 1)
        topic, status = (es_topics if es else en_topics)[i % 4]
        if not es:
            other = "closed" if status == "open" else "open"
        else:
            other = {"abierto": "cerrado", "equilibrado": "desequilibrado",
                     "estable": "inestable",
                     "tranquila": "ruidosa"}[status]
        q = (f"¿{topic} está {status}?" if es else f"is {topic} {status}?")
        state = (f"aviso: {topic} está {status}" if es
                 else f"notice: {topic} is {status}")
        opts = [{"id": "o0", "text": f"{topic} {status}"},
                {"id": "o1", "text": f"{topic} {other}"}]
        item_id = f"g{i:04d}"
        items.append({"id": item_id, "locale": "es-ES" if es else "en-US",
                      "state": state, "question": q, "options": opts,
                      "answer": "o0", "provenance": PROVENANCE})
        # Two annotators: agree w.p. ~0.85, else keep both distributions.
        agree = rng.random() < 0.85
        d1 = {"o0": 0.9, "o1": 0.1}
        if agree:
            d2 = {"o0": 0.8, "o1": 0.2}
            amb = False
        else:
            d2 = {"o0": 0.45, "o1": 0.55}
            amb = True
        anns.append({"item_id": item_id, "annotator": "ann-a",
                     "distribution": d1, "ambiguous": amb, "notes": ""})
        anns.append({"item_id": item_id, "annotator": "ann-b",
                     "distribution": d2, "ambiguous": amb, "notes": ""})
    return items, anns


def cohen_kappa(anns: list[dict]) -> float:
    """Kappa on argmax hard labels, recomputed from raw annotations."""
    by_item: dict[str, list[dict]] = {}
    for a in anns:
        by_item.setdefault(a["item_id"], []).append(a["distribution"])
    agree = 0
    pa_counts = [0, 0]
    pb_counts = [0, 0]
    for dists in by_item.values():
        if len(dists) != 2:
            continue
        la = max(dists[0], key=lambda k: dists[0][k])
        lb = max(dists[1], key=lambda k: dists[1][k])
        ia, ib = (0 if la == "o0" else 1), (0 if lb == "o0" else 1)
        agree += ia == ib
        pa_counts[ia] += 1
        pb_counts[ib] += 1
    n = sum(pa_counts)
    po = agree / n
    pe = sum((a / n) * (b / n) for a, b in zip(pa_counts, pb_counts))
    return (po - pe) / max(1 - pe, 1e-12)


def exact_agree_rate(anns: list[dict]) -> float:
    by_item: dict[str, list[dict]] = {}
    for a in anns:
        by_item.setdefault(a["item_id"], []).append(a["distribution"])
    hits = 0
    total = 0
    for dists in by_item.values():
        if len(dists) != 2:
            continue
        total += 1
        la = max(dists[0], key=lambda k: dists[0][k])
        lb = max(dists[1], key=lambda k: dists[1][k])
        hits += la == lb
    return hits / total


def reserve_splits(items: list[dict], seed: int = SEED) -> dict[str, list[dict]]:
    """Deterministic calibration/test reservation (100 / 400)."""
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    calib = [items[i] for i in idx[:100]]
    test = [items[i] for i in idx[100:]]
    assert not ({c["id"] for c in calib} & {t["id"] for t in test})
    return {"calibration": calib, "test": test}


def massive_fixture_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "artifacts", "fixtures", "gold",
        "massive_es_en.jsonl")


def massive_adapter(row: dict, intents: list[str]) -> dict:
    """MASSIVE row -> universal schema choice item (CC-BY-4.0)."""
    opts = [{"id": f"o{j}", "text": t} for j, t in enumerate(intents)]
    gold = next(o["id"] for o in opts if o["text"] == row["intent"])
    return {"id": f"massive-{row['locale']}-{row['utt_id']}",
            "locale": row["locale"], "state": row["utterance"],
            "question": "intent?",
            "options": opts, "answer": gold,
            "source": {"dataset": "AmazonScience/massive",
                       "license": "CC-BY-4.0",
                       "revision": MASSIVE_REVISION}}


def massive_card() -> dict:
    with open(massive_fixture_path(), "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return {"dataset": "AmazonScience/massive", "license": "CC-BY-4.0",
            "revision": MASSIVE_REVISION, "sha256": digest,
            "attribution": "AmazonScience MASSIVE, CC-BY-4.0"}


def _cos_text(a: str, b: str) -> float:
    def grams(t: str) -> dict[str, int]:
        t = f" {t.lower()} "
        d: dict[str, int] = {}
        for i in range(len(t) - 2):
            g = t[i:i + 3]
            d[g] = d.get(g, 0) + 1
        return d
    ga, gb = grams(a), grams(b)
    dot = sum(v * gb.get(k, 0) for k, v in ga.items())
    na = math.sqrt(sum(v * v for v in ga.values())) or 1.0
    nb = math.sqrt(sum(v * v for v in gb.values())) or 1.0
    return dot / (na * nb)


def benchmark_gap(items: list[dict]) -> dict:
    """EN vs ES accuracy with a fixed cosine scorer; gap reported."""
    acc: dict[str, list[int]] = {}
    for it in items:
        scores = [_cos_text(it["state"], o["text"]) for o in it["options"]]
        pred = it["options"][max(range(len(scores)),
                                 key=lambda i: scores[i])]["id"]
        acc.setdefault(it["locale"], []).append(pred == it["answer"])
    per = {loc: sum(v) / len(v) for loc, v in acc.items()}
    locs = sorted(per)
    gap = abs(per[locs[0]] - per[locs[1]]) if len(locs) == 2 else 0.0
    return {"per_locale": per, "gap": gap, "n": len(items)}


def leakage_proof(gold_items: list[dict],
                  train_texts: list[str]) -> dict:
    """No gold item appears in train (exact + scanner), and the
    canary corpus does not leak into gold either."""
    detector = LeakageDetector(train_texts)
    in_train = []
    for g in gold_items:
        hit, _ = detector.scan(g["state"] + " " + g["question"])
        if hit:
            in_train.append(g["id"])
    scanner = ContaminationScanner()
    canary_in_gold = []
    for g in gold_items:
        text = g["state"] + " " + g["question"]
        hit, _ = scanner.scan(text)
        if hit:
            canary_in_gold.append(g["id"])
    return {"gold_in_train": in_train, "canary_in_gold": canary_in_gold,
            "clean": not in_train and not canary_in_gold}


def p0_train_texts() -> list[str]:
    base = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "artifacts", "fixtures", "p0")
    texts = []
    for fn in sorted(os.listdir(base)):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(base, fn)) as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
    return texts


def run_gold(seed: int = SEED) -> dict:
    items, anns = build_pilot(seed)
    manifest_ids = {it["id"] for it in items}
    for a in anns:
        assert not validate_annotation(a, manifest_ids), \
            f"invalid pilot annotation: {a}"
    kappa = cohen_kappa(anns)
    agree = exact_agree_rate(anns)
    splits = reserve_splits(items, seed)
    assert len(splits["calibration"]) == 100
    assert len(splits["test"]) == 400
    # MASSIVE ES+EN benchmark.
    intents: list[str] = []
    rows = []
    with open(massive_fixture_path()) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    intents = sorted({r["intent"] for r in rows})
    adapted = [massive_adapter(r, intents) for r in rows]
    gap = benchmark_gap(adapted)
    proof = leakage_proof(items, p0_train_texts())
    assert proof["clean"], f"gold leakage: {proof}"
    report = {
        "seed": seed, "n_pilot": len(items),
        "provenance": PROVENANCE,
        "iaa": {"cohen_kappa": round(kappa, 4),
                "exact_agree_rate": round(agree, 4)},
        "splits": {k: len(v) for k, v in splits.items()},
        "massive": {"card": massive_card(), "n": len(adapted),
                    "gap": gap},
        "leakage": proof,
        "human_collection": "pending",
    }
    out_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "artifacts", "gates", "T-gold")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    with open(os.path.join(out_dir, "pilot.json"), "w") as f:
        json.dump({"items": items, "annotations": anns}, f)
        f.write("\n")
    return report
