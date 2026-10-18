"""Verification gate for #T-gen-schemas.

Writes ``artifacts/gates/T-gen-schemas/gate.json`` with the diversity of the
generated corpus per axis and the dedup rejection rate, and passes only when
every criterion in the task's verification gate holds. The criteria are
written here, before the numbers — the same discipline as #T-release-gate.
"""
from __future__ import annotations

import json
from pathlib import Path

from data.firewall import ContaminationScanner, reject_train_rows

from .budget import Budget
from .diversity import diversity
from .generator import generate

ROOT = Path(__file__).resolve().parent.parent.parent
GATE_DIR = ROOT / "artifacts" / "gates" / "T-gen-schemas"

# Criteria, fixed before measuring (task #T-gen-schemas, "Verification gate").
MIN_SCHEMAS = 1000
MIN_SKELETONS = 200
MIN_DOMAINS = 5
MIN_LANGUAGES = 2
MIN_K_VALUES = 4
UNKNOWN_RATE_RANGE = (0.05, 0.35)
MIN_HARD_NEGATIVE_RATE = 0.8


def run(per_cell: int = 8, seed: int = 20260921, log=print) -> dict:
    budget = Budget.plan(per_cell=per_cell)
    scanner = ContaminationScanner()
    res = generate(budget, seed=seed, scanner=scanner, log=log)
    report = diversity(res.schemas, res.rejections, res.deduper, budget,
                       res.proposer)

    # Injected near-duplicate: an accepted schema re-offered with its proper
    # nouns churned must be rejected, which is the whole point of masking
    # entities before the Jaccard (finding C).
    probe = "not run"
    if res.schemas:
        churned = res.schemas[0].norm_text
        accepted, sim = res.deduper.add(churned)
        probe = {"rejected": not accepted, "similarity": round(sim, 4)}

    # Benchmark firewall (#T-halt-contam): nothing generated may carry
    # eval-only content. reject_train_rows raises on any hit.
    texts = [txt for s in res.schemas for txt in s.texts()]
    try:
        reject_train_rows(texts)
        firewall = {"clean": True, "texts_scanned": len(texts)}
    except ValueError as exc:
        firewall = {"clean": False, "texts_scanned": len(texts), "error": str(exc)}

    checks = {
        "schemas": report["schemas"] >= MIN_SCHEMAS,
        "unique_skeletons": report["unique_skeletons"] >= MIN_SKELETONS,
        "domains": len(report["domains"]) >= MIN_DOMAINS,
        "languages": len(report["languages"]) >= MIN_LANGUAGES,
        "k_cardinalities": len(report["k_distribution"]) >= MIN_K_VALUES,
        "unknown_rate": (UNKNOWN_RATE_RANGE[0] <= report["unknown_rate"]
                         <= UNKNOWN_RATE_RANGE[1]),
        "hard_negative_rate": report["hard_negative_rate"] >= MIN_HARD_NEGATIVE_RATE,
        "budget_full": bool(budget.report()["full"]),
        "no_quota_overflow": not budget.overflow(),
        "dedup_rejects_duplicate": isinstance(probe, dict) and probe["rejected"],
        "firewall_clean": firewall["clean"],
    }
    gate = {
        "task": "T-gen-schemas",
        "pass": all(checks.values()),
        "seed": seed,
        "criteria": {
            "min_schemas": MIN_SCHEMAS, "min_unique_skeletons": MIN_SKELETONS,
            "min_domains": MIN_DOMAINS, "min_languages": MIN_LANGUAGES,
            "min_k_cardinalities": MIN_K_VALUES,
            "unknown_rate_range": list(UNKNOWN_RATE_RANGE),
            "min_hard_negative_rate": MIN_HARD_NEGATIVE_RATE,
        },
        "checks": checks,
        "diversity": report,
        "dedup_probe": probe,
        "firewall": firewall,
        "stopped": res.stopped,
        "attempts": res.attempts,
    }
    return gate


def write(gate: dict, gate_dir: Path = GATE_DIR) -> Path:
    gate_dir.mkdir(parents=True, exist_ok=True)
    path = gate_dir / "gate.json"
    path.write_text(json.dumps(gate, indent=2, ensure_ascii=False) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="#T-gen-schemas verification gate")
    ap.add_argument("--per-cell", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    log = (lambda *a, **k: None) if args.quiet else print
    gate = run(per_cell=args.per_cell, seed=args.seed, log=log)
    path = write(gate)
    print(f"[gate] {'PASS' if gate['pass'] else 'FAIL'} -> {path}")
    for name, ok in gate["checks"].items():
        print(f"  {'ok ' if ok else 'FAIL'} {name}")
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
