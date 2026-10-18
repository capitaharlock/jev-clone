"""Diversity dashboard and token ledger for an assembled stage (§§63, 65).

Shares and a Simpson index on every axis the strategy asks to publish —
domain/family, question type, K bucket, language, §86 layer, origin — plus
the §63 token ledger: examples, state tokens, candidate tokens and mean K
per SHARD, so "how big is this corpus really" has an answer that is not a
row count.

Everything here reads counts that `data.mix.realise()` produced. Nothing
here decides membership.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.mix import k_bucket, simpson  # noqa: E402,F401 (re-exported)

from .strata import layer_report, type_report  # noqa: E402


def shares(counts: dict) -> dict:
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in sorted(counts.items())}


def dashboard(counts: dict) -> dict:
    """Shares + Simpson on every published axis.

    Accepts either a `data.mix.realise()` count block or the stratum
    counts of `tools.mix_1m.sampler.assemble()`: they carry the same axis
    names on purpose.
    """
    axes = {}
    for axis in ("dataset", "family", "layer", "qtype", "lang", "origin"):
        axes[axis] = shares(counts.get(axis, {}))
    if counts.get("k_bucket"):
        axes["k_bucket"] = shares(counts["k_bucket"])
    else:
        buckets: dict = {}
        for k, n in counts.get("k", {}).items():
            buckets[k_bucket(int(k))] = buckets.get(k_bucket(int(k)), 0) + n
        axes["k_bucket"] = shares(buckets)
    out = dict(axes)
    out["simpson"] = {a: round(simpson(s), 4) for a, s in axes.items()}
    return out


def token_ledger(counts: dict) -> dict:
    """The §63 ledger: totals plus one row per shard."""
    return {"total": counts.get("tokens", {}),
            "by_shard": counts.get("tokens_by_shard", {})}


def full_dashboard(counts: dict) -> dict:
    """Everything the #T-mix-1m gate publishes about one assembled stage."""
    strata = counts.get("strata", counts)
    return {
        "axes": dashboard(strata),
        "layer_plan": layer_report(strata),
        "type_mix": type_report(strata),
        "hard": {"n": strata.get("hard", 0),
                 "share": round(strata.get("hard", 0)
                                / max(strata.get("rows", 0), 1), 6)},
        "tokens": token_ledger(counts),
    }
