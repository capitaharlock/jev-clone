"""Assembly of `decision-mix-clean-1m` on top of the one mixture authority.

There is exactly ONE place that decides what enters a training mixture and
in what proportion: `data/mix.py`. This module does not re-implement any
of it. It asks for the §86 corpus by:

1. taking the registry MINUS the §§18/77 benchmark fence
   (`data.mix.clean_datasets()`);
2. planning with the TRAINER's own `build_mix()` — holdout included —
   pulled by the §86 layer targets (`weights=`) and clamped by the
   §§65-66 caps, which stay hard, so the published manifest describes the
   corpus that actually trains;
3. realising with `data.mix.realise()` and hanging the fence and the
   stratum tagging off its `on_member` observer, so the fence sees the
   same rows the caps counted;
4. verifying the REALISED composition with `data.mix.verify_mix()` and
   `data.mix.verify_guardrails()` — both of which raise.

Feasibility is arithmetic, not opinion. `cap_units()` over the fenced
registry says what share of a mixture these sources can cover at all; if
it is below 1.0 there is no mixture of any size, and `assert_feasible()`
reports that with the numbers instead of quietly rebalancing.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.mix import (  # noqa: E402
    CAP_SAFETY_MARGIN,
    MAX_DATASET_FRACTION,
    MAX_FAMILY_FRACTION,
    MAX_SYNTHETIC_FRACTION,
    MIN_HARD_FRACTION,
    MIN_HUMAN_FRACTION,
    MixGuardrailError,
    MixInfeasible,
    cap_units,
    clean_datasets,
    effective_supply,
    families,
    max_feasible_target,
    realise,
    scan_supply,
    verify_guardrails,
    verify_mix,
)

from . import fence as fence_mod  # noqa: E402
from .strata import layer_supply, layer_weights, tag  # noqa: E402

#: the §65 breach ("nunca 89 % sin que el pipeline lo marque como error")
#: is `data.mix`'s error, not a second exception type of our own.
CapViolation = MixGuardrailError

#: 100 rows in a million, and it is not a relaxation of the §§65-66 caps
#: — it is what makes them hold on the REALISED mixture.
#:
#: `data.mix.realise` trims a source at ROW granularity, because the
#: loader hands out whole rows: a source whose rows carry two or three
#: questions (snli, goemotions, detox-attack) therefore stops one or two
#: members short of its quota. The realised total then lands a handful
#: below the target — and a source planned at EXACTLY 15 % of the target
#: is above 15 % of that smaller total. A 1 M-row corpus aborted on
#: 150,000/999,999 = 15.0000015 %.
#:
#: So the plan is allocated against a cap this much tighter and the HARD
#: cap is still checked, unmoved, on what was actually built. The bound
#: on the loss is (questions per row - 1) per source, a few dozen rows;
#: this margin covers it two orders of magnitude over and moves a
#: published share by at most 0.01 pp.
CAP_MARGIN = 0.0001


class SupplyShortfall(Exception):
    """The clean supply cannot fill the requested stage under the caps."""


def clean_supply(root=None, refresh: bool = False) -> tuple:
    """`(scan, supply)` for the fenced registry. One pass, cached.

    This is the RAW supply, before the unseen-label holdout cuts it; it
    answers "what clean corpus exists on disk". What a stage can actually
    plan against is `plan_clean()`, which uses the trainer's own builder.
    """
    datasets = fence_mod.assert_spec_clean(clean_datasets())
    kwargs = {"refresh": refresh}
    if root is not None:
        kwargs["root"] = root
    scan = scan_supply(datasets, **kwargs)
    return scan, effective_supply(scan)


def plan_clean(target, seed: int, root=None, drop=()) -> tuple:
    """Plan the clean corpus with the TRAINER's builder. Returns `(spec, scan)`.

    `training.python.train_decision.build_mix` is what a training run
    calls, holdout and all: it scans, carves the unseen-label holdout,
    computes the supply that SURVIVES that carving and plans on it. Going
    through the same function is what makes the published manifest the
    corpus that trains — a manifest planned on the raw supply would
    describe a mixture the loader then cannot fill.

    Importing the trainer is safe without torch: `train_decision` imports
    it lazily and `build_mix` never touches it.
    """
    from training.python.train_decision import build_mix

    datasets = fence_mod.assert_spec_clean(
        [d for d in clean_datasets() if d not in set(drop)])
    weights = layer_weights(datasets)
    kwargs = {"datasets": datasets, "cap_margin": CAP_MARGIN,
              "weights": weights}
    if root is not None:
        kwargs["root"] = root
    spec, scan, _pools, holdout = build_mix(seed, target=target, **kwargs)
    return spec, scan, holdout


def feasibility(supply: dict, cap_margin: float = CAP_SAFETY_MARGIN) -> dict:
    """What size of clean mixture the caps allow — with the arithmetic.

    Two ceilings are reported, because they differ and the difference is
    the finding: the HARD caps (§§65-66 as written) and the planner's
    slightly tighter caps, which exist so a mixture cannot drift over a
    hard cap between the plan and the run.
    """
    datasets = sorted(supply)
    units = cap_units(datasets)
    margin_units = cap_units(datasets, MAX_DATASET_FRACTION - cap_margin,
                             MAX_FAMILY_FRACTION - cap_margin)
    return {
        "datasets": datasets,
        "families": {f: sorted(ds) for f, ds in families(datasets).items()},
        "supply": dict(sorted(supply.items())),
        "supply_total": sum(supply.values()),
        "cap_units": round(units, 6),
        "cap_units_with_margin": round(margin_units, 6),
        "cap_margin": cap_margin,
        "max_target_hard_caps": max_feasible_target(supply),
        "max_target_with_margin": max_feasible_target(
            supply, MAX_DATASET_FRACTION - cap_margin,
            MAX_FAMILY_FRACTION - cap_margin),
        "rule": ("a family contributes min(family_cap, n x dataset_cap) of "
                 "any mixture; covering 100 % needs the sum of those to "
                 "reach 1.0, which no target size can change"),
    }


def assert_feasible(target, supply: dict,
                    cap_margin: float = CAP_SAFETY_MARGIN) -> dict:
    """Refuse a target the caps cannot cover, with the arithmetic. Raises.

    The §86 layer plan enters the allocator as WEIGHTS, never as a cap:
    the strategy asks for 30 % human-expert and 20 % programmatic, the
    guardrails say no source over 15 % and no family over 30 %, and where
    the two disagree the guardrails win and the gap gets published.
    """
    feas = feasibility(supply, cap_margin)
    ceiling = feas["max_target_hard_caps"]
    if ceiling <= 0:
        raise SupplyShortfall(
            f"no clean mixture exists at any size: the fenced registry "
            f"covers at most {feas['cap_units']:.4f} of a mixture "
            f"(needs >= 1.0). {feas['rule']}. Families: {feas['families']}")
    if target is not None and target > ceiling:
        raise SupplyShortfall(
            f"stage target {target:,} rows is above the clean ceiling "
            f"{ceiling:,}: with supply {feas['supply']} and the §§65-66 "
            f"caps, {target - ceiling:,} rows have nowhere to come from. "
            f"Add an independent source — a bigger target cannot fix it.")
    return feas


def assemble(spec, root=None, jevals_ids=None, jevals_hashes=None) -> tuple:
    """Realise the stage, fence it, verify it. Returns `(counts, fence)`.

    Every raise here is deliberate: a fenced row, a breached cap or a
    stratum that drifts past a guardrail stops the build. `pass: false`
    with honest numbers is a result; a mixture assembled around a broken
    guardrail is not.
    """
    fence = fence_mod.Fence(jevals_hashes, jevals_ids)
    strata = {"dataset": {}, "family": {}, "layer": {}, "qtype": {},
              "lang": {}, "k_bucket": {}, "origin": {}, "hard": 0}

    def observe(dataset, index, row, question):
        state = row.get("state", "")
        allowed, reason = fence.check(dataset, state, question.get("id", ""))
        if not allowed:
            raise fence_mod.FenceBreach(
                f"{reason}: {dataset} row {index} question "
                f"{question.get('id', '')!r} was selected into the mixture")
        t = tag(dataset, state, question)
        for axis in ("dataset", "family", "layer", "qtype", "lang",
                     "k_bucket", "origin"):
            strata[axis][t[axis]] = strata[axis].get(t[axis], 0) + 1
        strata["hard"] += int(t["hard"])

    counts = (realise(spec, root, observe) if root is not None
              else realise(spec, on_member=observe))
    total = counts["rows"]
    # the caps, on the composition that was actually built
    counts["guardrails"] = verify_mix(counts["dataset"], total)
    strata["rows"] = total
    counts["strata"] = strata
    counts["fence"] = fence.assert_clean()
    return counts, fence


def check_guardrails(counts: dict) -> dict:
    """§§65-66 + §68 over the realised stage. Raises `CapViolation`."""
    return verify_guardrails(
        {"dataset": counts["dataset"], "family": counts["family"],
         "origin": counts["origin"], "hard": counts.get("hard", 0)},
        counts["rows"])


def guardrail_report(counts: dict) -> dict:
    """The same checks WITHOUT raising, for a gate that must still be written.

    A guardrail that fails has to end up in `gate.json` as a false, with
    its measured share next to its floor — not as a traceback that leaves
    no artifact behind.
    """
    total = counts.get("rows", 0) or 1
    origin = counts.get("origin", {})
    checks = {}
    for dataset, n in sorted(counts.get("dataset", {}).items()):
        checks[f"dataset:{dataset}<=15%"] = _c(n / total,
                                               MAX_DATASET_FRACTION, "max")
    for family, n in sorted(counts.get("family", {}).items()):
        checks[f"family:{family}<=30%"] = _c(n / total,
                                             MAX_FAMILY_FRACTION, "max")
    checks["synthetic<=50%"] = _c(origin.get("synthetic", 0) / total,
                                  MAX_SYNTHETIC_FRACTION, "max")
    checks["human>=20%"] = _c(origin.get("human", 0) / total,
                              MIN_HUMAN_FRACTION, "min")
    checks["hard>=10%"] = _c(counts.get("hard", 0) / total,
                             MIN_HARD_FRACTION, "min")
    failed = sorted(k for k, v in checks.items() if not v["ok"])
    return {"total": counts.get("rows", 0), "checks": checks,
            "failed": failed, "pass": not failed,
            "enforced_by": ("data.mix.verify_guardrails, which raises "
                            "MixGuardrailError on the realised mixture")}


def _c(share: float, bound: float, kind: str) -> dict:
    ok = share <= bound + 1e-9 if kind == "max" else share >= bound - 1e-9
    return {"share": round(share, 6), "bound": bound, "kind": kind, "ok": ok}


def layer_gap(supply: dict) -> dict:
    """§86 layer supply against the plan, before any row is read."""
    return {"supply_by_layer": dict(sorted(layer_supply(supply).items())),
            "weights": {d: round(w, 6)
                        for d, w in sorted(layer_weights(sorted(supply)).items())}}


__all__ = ["CAP_MARGIN", "CapViolation", "MixInfeasible",
           "SupplyShortfall", "assemble",
           "assert_feasible", "check_guardrails", "clean_supply", "feasibility",
           "guardrail_report", "layer_gap", "plan_clean", "verify_guardrails"]
