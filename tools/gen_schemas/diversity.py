"""Measured, published diversity for the generated corpus (#T-gen-schemas).

The old loop published rows/min. Rows/min said nothing: 258 700 rows carried
190 skeletons. What is published here is the axis set that actually decides
whether the corpus teaches anything — unique skeletons, domains, languages,
K distribution, unknown rate, hard-negative coverage — plus the skeleton
growth curve, so a flattening curve is visible instead of inferred.
"""
from __future__ import annotations

from collections import Counter


def _share(counter: Counter, total: int) -> dict:
    return {k: {"n": v, "share": round(v / total, 4)}
            for k, v in sorted(counter.items(), key=lambda kv: str(kv[0]))}


def skeleton_curve(schemas, buckets: int = 10) -> list[dict]:
    """Unique skeletons seen after each 1/buckets of the run, in order."""
    if not schemas:
        return []
    step = max(1, len(schemas) // buckets)
    seen: set[str] = set()
    curve: list[dict] = []
    for i, s in enumerate(schemas, 1):
        seen.add(s.skeleton)
        if i % step == 0 or i == len(schemas):
            curve.append({"schemas": i, "unique_skeletons": len(seen)})
    return curve


def diversity(schemas, rejections: Counter | None = None,
              deduper=None, budget=None, proposer=None) -> dict:
    n = len(schemas)
    rejections = Counter(rejections or {})
    if not n:
        return {"schemas": 0, "rejections": dict(rejections)}
    skeletons = {s.skeleton for s in schemas}
    unknown = sum(1 for s in schemas if s.unknown)
    with_hard = sum(1 for s in schemas if s.hard_negatives)
    report = {
        "schemas": n,
        "unique_skeletons": len(skeletons),
        "skeletons_per_schema": round(len(skeletons) / n, 4),
        "skeleton_curve": skeleton_curve(schemas),
        "domains": _share(Counter(s.domain for s in schemas), n),
        "languages": _share(Counter(s.language for s in schemas), n),
        "k_distribution": _share(Counter(s.k for s in schemas), n),
        "state_formats": _share(Counter(s.state_format for s in schemas), n),
        "queried_fields": _share(Counter(s.queried_field for s in schemas), n),
        "splits": _share(Counter(s.split for s in schemas), n),
        "sources": _share(Counter(s.source for s in schemas), n),
        "unknown_rate": round(unknown / n, 4),
        "hard_negative_rate": round(with_hard / n, 4),
        "mean_options": round(sum(s.k for s in schemas) / n, 3),
        "rejections": dict(sorted(rejections.items())),
        "rejected_total": sum(rejections.values()),
    }
    if deduper is not None:
        report["dedup"] = {
            "threshold": deduper.threshold,
            "checked": deduper.n_checked,
            "rejected": deduper.n_rejected,
            "rejection_rate": round(deduper.rejection_rate(), 4),
        }
    if budget is not None:
        report["budget"] = budget.report()
    if proposer is not None:
        report["proposer"] = proposer.report()
    return report
