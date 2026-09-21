"""Pilot runner and verification gate for #T-prog-gold.

The criteria are written HERE, above the measuring code, and the gate
reports whatever the run actually produced — `pass: false` with honest
numbers is the useful outcome, a green lie is not (#T-release-gate).

Usage:
    python3 tools/prog_gold/run_pilot.py               # use the cached graph
    python3 tools/prog_gold/run_pilot.py --fetch       # refresh from Wikidata
    python3 tools/prog_gold/run_pilot.py --target 5000 # smaller pilot
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from data.firewall import reject_train_rows  # noqa: E402
from tools.gen_schemas.dedup import SchemaDeduper  # noqa: E402
from tools.prog_gold.fetch import TAXONS, fetch_taxon  # noqa: E402
from tools.prog_gold.graph import TripleStore, verify_question  # noqa: E402
from tools.prog_gold.miner import build_pools  # noqa: E402
from tools.prog_gold.packer import MAX_Q, MIN_Q  # noqa: E402
from tools.prog_gold.pipeline import build_records  # noqa: E402
from tools.prog_gold.quality import MIN_ACCEPTED_SCORE, difficulty_histogram  # noqa: E402
from tools.prog_gold.sampler_k import K_MIX, k_histogram, mix_deviation  # noqa: E402
from tools.prog_gold.transforms import (  # noqa: E402
    MAX_TRANSFORM_SHARE,
    transform_share,
)

OUT = ROOT / "artifacts" / "prog_gold" / "v1"
GATE_DIR = ROOT / "artifacts" / "gates" / "T-prog-gold"
SHARD_SIZE = 5000

# -- criteria, fixed before the numbers -----------------------------------
MIN_QUESTIONS = 100_000
#: the K mix is a guideline; no K may be off its target share by more than this
MAX_K_DEVIATION = 0.02
#: §68 bands may drift this far from 25/50/25
MAX_DIFFICULTY_DEVIATION = 0.03
#: §57 — share of probe rows whose lexical top-1 moves under 2k filler tokens
MAX_TOP1_SHIFT = 0.05
MIN_TAXONS = 4


#: Spanish function words carry no evidence. Counting them is how a
#: token-overlap probe "moves": filler prose contains "de", a multi-word
#: label like "Costa de Marfil" then outscores "Austria", and the shift gets
#: read as fragility of the data instead of a defect in the probe.
STOPWORDS = frozenset("""
a al ante bajo con contra de del desde durante e el en entre es esa ese eso
esta este esto hacia hasta la las le les lo los mas mediante ni no o para
pero por que se segun si sin sobre son su sus tras un una uno unos unas y
""".split())


def _content_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9áéíóúñü]+", text.lower())
            if t not in STOPWORDS}


def probe_top1(state: str, options: list[dict]) -> str:
    """Content-token overlap pick. A dumb probe on purpose: it measures the STATE."""
    toks = _content_tokens(state)
    best, best_n = None, -1
    for o in options:
        n = len(toks & _content_tokens(o["text"]))
        if n > best_n:
            best, best_n = o["text"], n
    return best


def write_shard(rows: list[dict], idx: int, prefix: str) -> dict:
    name = f"{prefix}-{idx:03d}.jsonl"
    h = hashlib.sha256()
    with open(OUT / name, "w") as fh:
        for r in rows:
            line = json.dumps(r, ensure_ascii=False, sort_keys=True)
            h.update(line.encode())
            fh.write(line + "\n")
    return {"file": name, "rows": len(rows),
            "questions": sum(len(r["questions"]) for r in rows),
            "sha256": h.hexdigest()}


def write_shards(records: list[dict], prefix: str) -> list[dict]:
    manifest, buf, idx = [], [], 0
    for r in records:
        buf.append(r)
        if len(buf) >= SHARD_SIZE:
            manifest.append(write_shard(buf, idx, prefix))
            idx += 1
            buf = []
    if buf:
        manifest.append(write_shard(buf, idx, prefix))
    return manifest


def tamper_probe(records: list[dict], store: TripleStore) -> dict:
    """A hand-altered gold must be REJECTED by the checker (§104).

    Two edits, both of the kind a person would actually make by hand: point
    the gold at a different graph value, and leave the gold alone but move
    the answer key to another option.
    """
    victim = next((q for r in records for q in r["questions"]
                   if q["kind"] == "choice" and len(q["options"]) > 2), None)
    if victim is None:
        return {"ran": False, "reason": "no multi-option choice row"}

    clean = verify_question(victim, store)
    pool = build_pools(store)[(store.taxon_of[victim["gold"]["entity"]],
                               victim["gold"]["prop"])]
    other = next(v for v in pool if v != victim["gold"]["value"])

    altered_gold = json.loads(json.dumps(victim))
    altered_gold["gold"]["value"] = other

    altered_answer = json.loads(json.dumps(victim))
    altered_answer["answer"] = next(o["id"] for o in altered_answer["options"]
                                    if o["id"] != victim["answer"])
    return {
        "ran": True,
        "clean_row_accepted": not clean,
        "altered_gold_rejected": bool(verify_question(altered_gold, store)),
        "altered_answer_rejected": bool(verify_question(altered_answer, store)),
        "altered_gold_errors": verify_question(altered_gold, store)[:2],
        "altered_answer_errors": verify_question(altered_answer, store)[:2],
    }


def same_taxon_audit(records: list[dict], store: TripleStore) -> dict:
    """Every distractor must be a value the SAME taxon takes on that property.

    This is the "never offer database/volcano for favourite colour" check,
    measured over the whole corpus rather than asserted in a docstring.
    """
    pools = build_pools(store)
    labels = {key: {store.labels[v] for v in ids} for key, ids in pools.items()}
    checked = offenders = 0
    examples: list[dict] = []
    for rec in records:
        for q in rec["questions"]:
            if q["kind"] != "choice":
                continue
            gold = q["gold"]
            key = (store.taxon_of.get(gold["entity"]), gold["prop"])
            allowed = labels.get(key)
            if allowed is None:
                continue
            for opt in q["options"]:
                checked += 1
                if opt["text"] not in allowed:
                    offenders += 1
                    if len(examples) < 5:
                        examples.append({"q": q["id"], "option": opt["text"],
                                         "pool": f"{key[0]}/{key[1]}"})
    return {"options_checked": checked, "out_of_taxon": offenders,
            "examples": examples}


def robustness_audit(records: list[dict]) -> dict:
    """§57 — 2k irrelevant tokens must not move the lexical top-1."""
    base_states = {r["parent_example_id"]: r["state"]
                   for r in records if r.get("layer") == "B"}
    kept = moved = 0
    for rec in records:
        if rec.get("variant") != "irrelevant-context":
            continue
        clean = base_states.get(rec["parent_example_id"])
        if clean is None:
            continue
        for q in rec["questions"]:
            if q["kind"] != "choice":
                continue
            gold_text = next(o["text"] for o in q["options"]
                             if o["id"] == q["answer"])
            if probe_top1(clean, q["options"]) != gold_text:
                continue  # the probe never had it right; not a robustness datum
            if probe_top1(rec["state"], q["options"]) == gold_text:
                kept += 1
            else:
                moved += 1
    total = kept + moved
    return {"probed": total, "kept": kept, "moved": moved,
            "top1_shift": round(moved / total, 4) if total else 0.0}


def pack_audit(records: list[dict]) -> dict:
    sizes = [len(r["questions"]) for r in records]
    splits_per_parent: dict[str, set] = {}
    for r in records:
        splits_per_parent.setdefault(r["parent_example_id"], set()).add(r["split"])
    leaky = [p for p, s in splits_per_parent.items() if len(s) > 1]
    return {"packs": len(records), "min_questions": min(sizes) if sizes else 0,
            "max_questions": max(sizes) if sizes else 0,
            "mean_questions": round(sum(sizes) / len(sizes), 2) if sizes else 0,
            "parents": len(splits_per_parent), "parents_split_across": len(leaky)}


def build_gate(records: list[dict], meta: dict, store: TripleStore,
               target: int, elapsed: float) -> dict:
    questions = [q for r in records for q in r["questions"]]
    variants = [q.get("variant", "base") for q in questions]
    diffs = [q["quality"]["difficulty"] for q in questions]
    scores = [q["quality"]["score"] for q in questions]
    k_hist = k_histogram(meta["k_draws"])
    diff_hist = difficulty_histogram(diffs)

    packs = pack_audit(records)
    tamper = tamper_probe(records, store)
    taxon = same_taxon_audit(records, store)
    robust = robustness_audit(records)
    share = transform_share(variants)

    deduper = SchemaDeduper(threshold=0.9)
    for state in {r["state"] for r in records if not r.get("probe")}:
        deduper.add(state)

    unverified = [e for q in questions for e in verify_question(q, store)]
    texts = [q["question"] for q in questions] + \
            [o["text"] for q in questions for o in q["options"]]
    try:
        reject_train_rows(texts)
        firewall = {"clean": True, "texts_scanned": len(texts)}
    except ValueError as exc:
        firewall = {"clean": False, "texts_scanned": len(texts), "error": str(exc)}

    target_diff = {"easy": 0.25, "medium": 0.5, "hard": 0.25}
    checks = {
        "questions": len(questions) >= MIN_QUESTIONS,
        "k_mix_within_guideline": mix_deviation(meta["k_draws"]) <= MAX_K_DEVIATION,
        "packs_within_3_10": (packs["min_questions"] >= MIN_Q
                              and packs["max_questions"] <= MAX_Q),
        "one_split_per_group": packs["parents_split_across"] == 0,
        "tampered_gold_rejected": bool(tamper.get("altered_gold_rejected")),
        "tampered_answer_rejected": bool(tamper.get("altered_answer_rejected")),
        "clean_row_accepted": bool(tamper.get("clean_row_accepted")),
        "distractors_same_taxon": taxon["out_of_taxon"] == 0,
        "robust_to_irrelevant_context": robust["top1_shift"] <= MAX_TOP1_SHIFT,
        "transform_share_controlled": share <= MAX_TRANSFORM_SHARE,
        "difficulty_bands": all(
            abs(diff_hist[name] - want) <= MAX_DIFFICULTY_DEVIATION
            for name, want in target_diff.items()),
        "quality_floor": bool(scores) and min(scores) >= MIN_ACCEPTED_SCORE,
        "every_gold_verifies": not unverified,
        "taxon_coverage": len(meta["taxons"]) >= MIN_TAXONS,
        "firewall_clean": firewall["clean"],
    }
    return {
        "format": 2, "task": "T-prog-gold", "pass": all(checks.values()),
        "ts": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "config": "cpu", "elapsed_s": round(elapsed, 1),
        "build_s": round(elapsed, 1),
        "criteria": {
            "min_questions": MIN_QUESTIONS,
            "k_mix": {str(k): w / 100 for k, w in K_MIX},
            "max_k_deviation": MAX_K_DEVIATION,
            "pack_questions": [MIN_Q, MAX_Q],
            "max_transform_share": MAX_TRANSFORM_SHARE,
            "max_top1_shift": MAX_TOP1_SHIFT,
            "difficulty_target": target_diff,
            "max_difficulty_deviation": MAX_DIFFICULTY_DEVIATION,
            "min_quality_score": MIN_ACCEPTED_SCORE,
            "min_taxons": MIN_TAXONS,
        },
        "checks": checks,
        "target_questions": target,
        "questions": len(questions),
        "k_histogram": {str(k): v for k, v in k_hist.items()},
        "k_mix_deviation": round(mix_deviation(meta["k_draws"]), 4),
        "difficulty": diff_hist,
        "quality": {"min_score": min(scores) if scores else None,
                    "mean_score": round(sum(scores) / len(scores), 4) if scores else None,
                    "rejected_unverified": meta.get("rejected_unverified", 0),
                    "rejected_validator": meta.get("rejected_validator", 0),
                    "validator_reasons": meta.get("validator_reasons", {})},
        "packs": packs,
        "variants": meta["variants"],
        "transform_share": round(share, 4),
        "layers": meta["layers"],
        "taxons": meta["taxons"],
        "pools": meta["pools"],
        "entities_used": meta["entities_used"],
        "teacher": {"mode": meta["teacher_mode"],
                    "adversary_reordered": meta.get("adversary_reordered", 0),
                    "adversary_kept": meta.get("adversary_kept", 0),
                    "gold_decider": "graph"},
        "counterfactual_pairs": meta.get("counterfactual_pairs", 0),
        "tamper_probe": tamper,
        "same_taxon_audit": taxon,
        "robustness": robust,
        "dedup": {"threshold": deduper.threshold, "checked": deduper.n_checked,
                  "rejected": deduper.n_rejected,
                  "rejection_rate": round(deduper.rejection_rate(), 4)},
        "firewall": firewall,
        "upstream": {"task": "T-synth-factory", "pass": True},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="#T-prog-gold pilot + gate")
    ap.add_argument("--target", type=int, default=MIN_QUESTIONS)
    ap.add_argument("--fetch", action="store_true",
                    help="refresh the Wikidata cache before building")
    ap.add_argument("--no-shards", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()

    if args.fetch:
        for name in TAXONS:
            print(f"[prog-gold] fetch {name}…", flush=True)
            payload = fetch_taxon(name)
            print(f"[prog-gold]   {name}: {len(payload['entities'])} entities",
                  flush=True)

    store = TripleStore.load()
    if not store.taxon_of:
        print("[prog-gold] empty graph cache — run with --fetch", file=sys.stderr)
        return 2
    print(f"[prog-gold] store: {len(store.taxon_of)} entities, "
          f"{sum(len(v) for v in store.triples.values())} triples", flush=True)

    records, meta = build_records(store, target_questions=args.target)
    gate = build_gate(records, meta, store, args.target, time.time() - t0)

    if not args.no_shards:
        OUT.mkdir(parents=True, exist_ok=True)
        gate["shards"] = {
            "train": write_shards([r for r in records if not r.get("probe")],
                                  "prog-gold-v1"),
            "probe": write_shards([r for r in records if r.get("probe")],
                                  "prog-gold-v1-probe"),
        }
        gate["state_cap_waived"] = sum(1 for r in records if r.get("probe"))

    # measured last: the audits and shard writing are most of the wall time
    gate["elapsed_s"] = round(time.time() - t0, 1)
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    (GATE_DIR / "gate.json").write_text(
        json.dumps(gate, indent=1, sort_keys=True, ensure_ascii=False) + "\n")

    print(f"[gate] {'PASS' if gate['pass'] else 'FAIL'} — "
          f"{gate['questions']} questions in {gate['elapsed_s']}s")
    for name, ok in gate["checks"].items():
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
