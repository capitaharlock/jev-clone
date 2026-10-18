#!/usr/bin/env python3
"""Benchmark leak audit: are any held-out benchmark items hiding in train?

#T-halt-contam firewall evidence. Held-out = calibration/test splits of the
eval-only reasoning benchmarks (logiqa, reclor). Train corpus = train-split
rows of every job in `data/train_baseline.py::JOBS` (parsed via AST, so this
tool stays stdlib-only and can never drift from the trainer's job list).

Layers (all real, always run):
  1. exact      — raw sha256 of the state text.
  2. normalized — sha256 after leakage.normalize (case/whitespace folding).
  3. paraphrase — word-3-gram Jaccard >= threshold, with an inverted-index
     prefilter so the full cross product never materializes.
  4. semantic   — cosine over embeddings, gated by JEV_EMBEDDINGS_ENDPOINT +
     JEV_EMBEDDINGS_MODEL. When unset/unreachable the layer is a LOGGED STUB
     (exact + normalized + paraphrase stay real); business logic is never
     stubbed. When active it re-scores the paraphrase suspects.

Writes artifacts/gates/T-halt-contam/leak-report.json and prints a summary.
Exit 0 iff exact == 0 and normalized == 0 (zero tolerance); paraphrase
suspects are listed for triage but do not fail the gate on their own
(Jaccard has a known false-positive tail — see the firewall stress suite).

Usage:
  .venv-train/bin/python tools/check_benchmark_leak.py [--threshold 0.5]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.leakage import jaccard, normalize, text_hash, trigrams  # noqa: E402

EVAL_SPLITS = {"calibration", "test", "valid", "validation"}
HELDOUT_FILES = {"logiqa": "logiqa.jsonl", "reclor": "reclor.jsonl"}
# Minimum trigram-set size for a meaningful paraphrase claim (see scan loop).
MIN_TRIGRAMS = 4
GATE_DIR = ROOT / "artifacts" / "gates" / "T-halt-contam"
PREFETCH = ROOT / "artifacts" / "data-prefetch"


def train_job_files() -> list[tuple[str, str]]:
    """(name, file) pairs from the trainer's JOBS — single source of truth."""
    tree = ast.parse((ROOT / "data" / "train_baseline.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "JOBS" for t in node.targets
        ):
            pairs = []
            for elt in node.value.elts:
                d = {
                    k.value: v.value
                    for k, v in zip(elt.keys, elt.values)
                    if isinstance(k, ast.Constant)
                }
                pairs.append((d["name"], d["file"]))
            return pairs
    raise RuntimeError("JOBS not found in data/train_baseline.py")


def load_heldout() -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
    """Returns (items, exact_map, norm_map) over calibration/test splits."""
    items, exact_map, norm_map = [], {}, {}
    for bench, fn in HELDOUT_FILES.items():
        with open(PREFETCH / fn) as f:
            for line in f:
                r = json.loads(line)
                if r.get("split") not in EVAL_SPLITS or not r.get("state"):
                    continue
                item = {
                    "bench": bench,
                    "id": r.get("id", "?"),
                    "state": r["state"],
                }
                items.append(item)
                exact_map.setdefault(
                    hashlib.sha256(item["state"].encode()).hexdigest(), item
                )
                norm_map.setdefault(text_hash(item["state"]), item)
    return items, exact_map, norm_map


def build_trigram_index(items: list[dict]):
    """Inverted index trigram -> held-out idxs, plus per-item trigram sets."""
    tri_sets = [trigrams(it["state"]) for it in items]
    index: dict[str, list[int]] = {}
    for i, ts in enumerate(tri_sets):
        for t in ts:
            index.setdefault(t, []).append(i)
    return index, tri_sets


def cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    den = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    return num / den if den else 0.0


def embed_texts(texts: list[str], endpoint: str, model: str) -> list[list[float]]:
    """Real embedding fetch (OpenAI-compatible /embeddings). Stdlib only."""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), 100):
        payload = json.dumps(
            {"model": model, "input": texts[i : i + 100]}
        ).encode()
        req = urllib.request.Request(
            endpoint.rstrip("/") + "/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.load(resp)
        vectors.extend(d["embedding"] for d in body["data"])
    if len(vectors) != len(texts):
        raise RuntimeError(
            f"embedding count mismatch: {len(vectors)} != {len(texts)}"
        )
    return vectors


def semantic_layer(suspects, threshold):
    """Re-score paraphrase suspects with embeddings when configured.

    Returns a status dict; stubbed (with reason) when no endpoint is set.
    """
    endpoint = os.environ.get("JEV_EMBEDDINGS_ENDPOINT", "").strip()
    model = os.environ.get("JEV_EMBEDDINGS_MODEL", "").strip()
    if not endpoint or not model:
        reason = "JEV_EMBEDDINGS_ENDPOINT/JEV_EMBEDDINGS_MODEL unset"
        print(f"[semantic] STUBBED ({reason}); exact+normalized+paraphrase "
              f"remain real, {len(suspects)} suspect(s) left unscored",
              flush=True)
        return {
            "status": "stubbed",
            "reason": reason,
            "scored": 0,
            "hits": [],
        }
    try:
        train_vecs = embed_texts(
            [s["train_excerpt"] for s in suspects], endpoint, model)
        held_vecs = embed_texts(
            [s["heldout_excerpt"] for s in suspects], endpoint, model)
    except Exception as e:  # noqa: BLE001 — endpoint down is a stub reason
        reason = f"embeddings endpoint unreachable: {e}"
        print(f"[semantic] STUBBED ({reason})", flush=True)
        return {"status": "stubbed", "reason": reason,
                "scored": 0, "hits": []}
    hits = []
    for s, tv, hv in zip(suspects, train_vecs, held_vecs):
        sim = cosine(tv, hv)
        s["semantic_cosine"] = round(sim, 4)
        if sim >= threshold:
            hits.append({"train": s["train_ref"], "heldout": s["heldout_ref"],
                         "cosine": round(sim, 4)})
    print(f"[semantic] ACTIVE ({model}): scored {len(suspects)}, "
          f"{len(hits)} >= {threshold}", flush=True)
    return {"status": "active", "endpoint": endpoint, "model": model,
            "threshold": threshold, "scored": len(suspects), "hits": hits}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Jaccard paraphrase threshold")
    ap.add_argument("--semantic-threshold", type=float, default=0.90)
    ap.add_argument("--max-suspects", type=int, default=50,
                    help="cap on suspects listed in the report")
    args = ap.parse_args()
    t0 = time.time()

    jobs = train_job_files()
    print(f"[leak] train jobs from trainer JOBS: {[n for n, _ in jobs]}",
          flush=True)
    items, exact_map, norm_map = load_heldout()
    per_bench = Counter(it["bench"] for it in items)
    print(f"[leak] held-out items: {len(items)} "
          f"({dict(per_bench)})", flush=True)
    index, tri_sets = build_trigram_index(items)
    print(f"[leak] trigram index: {len(index)} keys over {len(items)} items",
          flush=True)

    exact_hits, norm_hits, suspects = [], [], []
    train_rows = Counter()
    for name, fn in jobs:
        path = PREFETCH / fn
        if not path.exists():
            print(f"[leak] SKIP {name}: {fn} missing", flush=True)
            continue
        n = 0
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("split") != "train" or not r.get("state"):
                    continue
                state = r["state"]
                n += 1
                exact = exact_map.get(
                    hashlib.sha256(state.encode()).hexdigest())
                if exact is not None:
                    exact_hits.append({
                        "train_ref": f"{name}:{r.get('id', '?')}",
                        "heldout_ref": f"{exact['bench']}:{exact['id']}",
                        "layer": "exact",
                        "excerpt": normalize(state)[:160],
                    })
                    continue
                normed = norm_map.get(text_hash(state))
                if normed is not None:
                    norm_hits.append({
                        "train_ref": f"{name}:{r.get('id', '?')}",
                        "heldout_ref": f"{normed['bench']}:{normed['id']}",
                        "layer": "normalized",
                        "excerpt": normalize(state)[:160],
                    })
                    continue
                # Paraphrase layer with overlap-bound pruning:
                # J(a,b) >= t  <=>  overlap >= t*(|a|+|b|)/(1+t).
                # Micro-texts (<4 trigrams: single tokens, spaceless CJK
                # fragments) are skipped here — Jaccard over a 1-2 element
                # set is statistically meaningless (every "A" would match
                # every passage ending in " A"). Verbatim copies of short
                # texts are still caught by the exact/normalized layers.
                ts = trigrams(state)
                if len(ts) < MIN_TRIGRAMS:
                    continue
                counts: Counter[int] = Counter()
                for t in ts:
                    for i in index.get(t, ()):
                        counts[i] += 1
                for i, overlap in counts.items():
                    b = len(tri_sets[i])
                    bound = args.threshold * (len(ts) + b) / (1 + args.threshold)
                    if overlap < math.ceil(bound):
                        continue
                    sim = jaccard(ts, tri_sets[i])
                    if sim >= args.threshold:
                        suspects.append({
                            "train_ref": f"{name}:{r.get('id', '?')}",
                            "heldout_ref": (f"{items[i]['bench']}:"
                                            f"{items[i]['id']}"),
                            "jaccard": round(sim, 4),
                            "train_excerpt": state[:2000],
                            "heldout_excerpt": items[i]["state"][:2000],
                        })
                        break  # one suspect record per train row is enough
        train_rows[name] = n
        print(f"[leak] {name}: {n:,} train rows scanned", flush=True)

    listed = suspects[: args.max_suspects]
    semantic = semantic_layer(listed, args.semantic_threshold)

    verdict_pass = not exact_hits and not norm_hits
    report = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "task": "T-halt-contam",
        "train_jobs": [n for n, _ in jobs],
        "train_rows_scanned": dict(train_rows),
        "heldout": {"total": len(items), "per_bench": dict(per_bench),
                    "splits": sorted(EVAL_SPLITS)},
        "layers": {
            "exact": {"hits": len(exact_hits), "rows": exact_hits[:20]},
            "normalized": {"hits": len(norm_hits), "rows": norm_hits[:20]},
            "paraphrase": {"threshold": args.threshold,
                           "min_trigrams": MIN_TRIGRAMS,
                           "suspects": len(suspects),
                           "listed": listed},
            "semantic": semantic,
        },
        "verdict": ("PASS — no held-out benchmark item in train"
                    if verdict_pass else
                    "FAIL — benchmark content found in train"),
        "pass": verdict_pass,
        "seconds": round(time.time() - t0, 1),
    }
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    (GATE_DIR / "leak-report.json").write_text(json.dumps(report, indent=2))
    print(f"[leak] exact={len(exact_hits)} normalized={len(norm_hits)} "
          f"paraphrase_suspects={len(suspects)} "
          f"semantic={semantic['status']} in {report['seconds']}s",
          flush=True)
    print(f"[leak] {report['verdict']} -> {GATE_DIR / 'leak-report.json'}",
          flush=True)
    return 0 if verdict_pass else 1


if __name__ == "__main__":
    sys.exit(main())
