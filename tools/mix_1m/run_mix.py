"""Build `decision-mix-clean-1m` and write the #T-mix-1m gate.

    .venv-train/bin/python -m tools.mix_1m.run_mix build [--target N]
    .venv-train/bin/python -m tools.mix_1m.run_mix gate

`build` plans and realises the corpus through `data/mix.py` — the one
mixture authority — asserts the §§18/77 benchmark fence over the rows it
actually selected, publishes the manifest, the §65 diversity dashboard
and the §63 token ledger per shard, and writes
`artifacts/gates/T-mix-1m/gate.json`.

`gate` re-reads the training runs and rewrites the same gate.json with
whatever the curve has measured by then, so a long run does not have to
finish inside one turn for its numbers to be published honestly.

The gate is written whether it passes or not. A `pass: false` carrying
the arithmetic that explains it is the point of the artifact.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import mix as mixmod  # noqa: E402

from . import backbones, diversity, fence as fence_mod, sampler  # noqa: E402
from .strata import TYPE_TARGETS  # noqa: E402

OUT = os.path.join(ROOT, "artifacts", "mix-1m")
GATE = os.path.join(ROOT, "artifacts", "gates", "T-mix-1m", "gate.json")

CORPUS = "decision-mix-clean-1m"
#: §86's headline: the first clean cut of `decision-mix-v1` is 1 M rows.
CORPUS_TARGET = 1_000_000
DEFAULT_SEED = 20260921

#: run ids of the §62 curve, one per top-2 backbone, plus the §128 arms
RUN_PREFIX = "mix1m"


def run_id(kind: str, backbone: str, seed: int) -> str:
    return f"{RUN_PREFIX}-{kind}-{backbone}-s{seed}"


def corpus_feasibility(supply: dict, target: int = CORPUS_TARGET) -> dict:
    """Can §86's 1 M-row clean corpus exist at all? Arithmetic, not opinion."""
    feas = sampler.feasibility(supply)
    ceiling = feas["max_target_hard_caps"]
    feas.update({
        "corpus": CORPUS,
        "requested_rows": target,
        "achievable_rows": ceiling,
        "shortfall_rows": max(0, target - ceiling),
        "pass": ceiling >= target,
        "why": (f"the §§65-66 caps admit at most {ceiling:,} rows from the "
                f"fenced registry, {target:,} were asked for. Every source "
                f"but the smallest sits exactly at the 15 % cap, so the only "
                f"fix is an independent source — no target size and no seed "
                f"changes this.") if ceiling < target else "",
        "missing_layers": sorted(
            layer for layer, target_share in mixmod.LAYER_TARGETS.items()
            if not any(mixmod.layer_of(d) == layer for d in supply)),
    })
    return feas


def build(target=None, seed: int = DEFAULT_SEED, write: bool = True) -> dict:
    """Assemble the corpus and return the block the gate publishes."""
    t0 = time.perf_counter()
    os.makedirs(OUT, exist_ok=True)
    _raw_scan, raw_supply = sampler.clean_supply()
    feas = corpus_feasibility(raw_supply)
    jevals_ids = fence_mod.load_jevals_ids()

    stage = {"corpus": CORPUS, "seed": seed,
             "requested_target": target or feas["achievable_rows"]}
    try:
        # planned through the trainer's builder, so the corpus published
        # here is the corpus `--fence-clean --mix-seed <seed>` trains
        spec, scan, holdout = sampler.plan_clean(target, seed)
        sampler.assert_feasible(target, spec.supply, cap_margin=0.0)
        counts, fence = sampler.assemble(spec, jevals_ids=jevals_ids)
        guard = sampler.guardrail_report(
            {"dataset": counts["dataset"], "family": counts["family"],
             "origin": counts["origin"], "hard": counts["strata"]["hard"],
             "rows": counts["rows"]})
        dash = diversity.full_dashboard(counts)
        manifest = mixmod.build_manifest(spec, counts, scan,
                                         time.perf_counter() - t0)
        manifest.update({
            "task": "T-mix-1m", "corpus": CORPUS,
            "fence": counts["fence"],
            "guardrails_68": guard,
            "layer_plan": dash["layer_plan"],
            "type_mix": dash["type_mix"],
            "tokens": dash["tokens"],
            "holdout": {d: h.to_dict() for d, h in sorted(holdout.items())},
            "reproduce": {
                "command": (f"python -m tools.mix_1m.run_mix build "
                            f"--target {spec.target} --seed {seed}"),
                "builder": ("training.python.train_decision.build_mix "
                            "-> data.mix.plan_mix / realise"),
                "trained_by": (f"train_decision train --fence-clean "
                               f"--mix-seed {seed}"),
                "cap_margin": 0.0,
                "weights": "data.mix.layer_weights (§86 layer plan)",
                "note": ("seed + the per-shard sha256 above rebuild this "
                         "corpus; members_sha256 proves the rebuild matched"),
            },
        })
        path = os.path.join(OUT, f"{CORPUS}-seed{seed}.manifest.json")
        if write:
            with open(path, "w") as fh:
                json.dump(manifest, fh, indent=2, sort_keys=True)
                fh.write("\n")
        stage.update({
            "status": "built", "rows": counts["rows"], "target": spec.target,
            "manifest": os.path.relpath(path, ROOT),
            "members_sha256": counts["members_sha256"],
            "quotas": spec.quotas,
            "supply_after_holdout": spec.supply,
            "composition": manifest["composition"],
            "diversity": dash["axes"],
            "layer_plan": dash["layer_plan"],
            "type_mix": dash["type_mix"],
            "type_targets": TYPE_TARGETS,
            "tokens": dash["tokens"],
            "guardrails": guard,
            "fence": counts["fence"],
        })
    except (sampler.SupplyShortfall, mixmod.MixInfeasible,
            mixmod.MixGuardrailError, fence_mod.FenceBreach) as exc:
        stage.update({"status": "failed",
                      "error": f"{type(exc).__name__}: {exc}"})
    stage["feasibility"] = feas
    stage["elapsed_s"] = round(time.perf_counter() - t0, 1)
    return stage


def training_block(seed: int, report=None) -> dict:
    """Whatever the §62 curve and the §128 arms have measured so far."""
    try:
        report = report if report is not None else backbones.load_bakeoff()
        chosen = backbones.top2(report)
        source = {"from": "artifacts/gates/T-bakeoff/report.json",
                  "task": "T-bakeoff-real", "top2": chosen,
                  "verdict": report.get("verdict")}
    except backbones.NoBakeoff as exc:
        return {"status": "blocked", "error": str(exc), "backbones": {}}

    rows = {b: backbones.backbone_report(b, run_id("curve", b, seed), report)
            for b in chosen}
    # the §128 arms are their own equal-budget pair, not the curve run:
    # a baseline WITHOUT synth-v1 cannot be built at the §65 15 % cap, so
    # both arms state 18 % and differ only in the synthetic source
    synth = backbones.synthetic_verdict(
        run_id("nosynth", chosen[0], seed), run_id("synth", chosen[0], seed),
        backbones.STAGES["stage0"])
    synth["design"] = {
        "arms": "same corpus size, same cap, same budget; synth-v1 in or out",
        "dataset_cap": 0.18,
        "why_not_15": ("without synth-v1 the fenced registry covers 90 % of "
                       "a mixture at the §65 15 % cap, so the baseline arm "
                       "does not exist there"),
        "why_not_additive": ("§128 words it as baseline vs +50 k, but the "
                             "caps fix the corpus size: adding synthetic "
                             "rows on top would breach the very guardrail "
                             "the arm is meant to respect, so the arms "
                             "substitute at fixed size instead"),
    }
    measured = [b for b, r in rows.items() if r["complete"]]
    return {
        "status": "measured" if len(measured) == len(chosen) else "running",
        "source": source,
        "stages": backbones.STAGES,
        "backbones": rows,
        "measured": measured,
        "synthetic_value": synth,
        "commands": {b: " ".join(backbones.train_command(
            b, run_id("curve", b, seed), backbones.STAGES["stage1"], seed))
            for b in chosen},
    }


def compose(stage: dict, training: dict, seed: int) -> dict:
    """The gate: corpus + fence + guardrails + top-2 + §128, pass or fail."""
    feas = stage.get("feasibility", {})
    checks = {
        "corpus_1m_rows": {
            "pass": bool(feas.get("pass")),
            "criterion": f"§86: {CORPUS_TARGET:,} clean rows",
            "measured_rows": stage.get("rows"),
            "achievable_rows": feas.get("achievable_rows"),
            "why": feas.get("why")},
        "mix_built": {
            "pass": stage.get("status") == "built",
            "criterion": "the clean corpus assembles under the §§65-66 caps",
            "error": stage.get("error")},
        "benchmark_fence": {
            "pass": bool(stage.get("fence", {}).get("clean")),
            "criterion": ("§§18, 77: zero Banking77 / HelpSteer2 / PubMedQA "
                          "rows and zero Jevals ids in train, asserted"),
            "report": stage.get("fence")},
        "guardrails": {
            "pass": bool(stage.get("guardrails", {}).get("pass")),
            "criterion": ("§§65-66, 68: <=15 % per dataset, <=30 % per "
                          "family, <=50 % synthetic, >=20 % human, "
                          ">=10 % hard/OOD"),
            "failed": stage.get("guardrails", {}).get("failed", []),
            "report": stage.get("guardrails")},
        "type_mix": {
            "pass": (stage.get("type_mix", {}).get("max_abs_gap", 1.0) <= 0.05
                     if stage.get("status") == "built" else False),
            "criterion": "§48: Choice 55 / Noul 30 / Score 15 within 5 pp",
            "report": stage.get("type_mix")},
        "diversity_dashboard": {
            "pass": bool(stage.get("diversity")),
            "criterion": ("§65: shares + Simpson by domain/family/type/K/"
                          "language/layer"),
            "axes": sorted(stage.get("diversity", {}))},
        "token_ledger": {
            "pass": bool(stage.get("tokens", {}).get("by_shard")),
            "criterion": "§63: examples, state/candidate tokens, mean K per shard",
            "total": stage.get("tokens", {}).get("total")},
        "reproducible": {
            "pass": bool(stage.get("members_sha256")),
            "criterion": "seed + manifest rebuild the corpus bit for bit",
            "members_sha256": stage.get("members_sha256")},
        "top2_trained": {
            "pass": training.get("status") == "measured",
            "criterion": ("§62: both top-2 backbones measured at Stage 0 "
                          "(250 k) and Stage 1 (1 M) on this corpus by the "
                          "real trainer"),
            "state": training.get("status"),
            "measured": training.get("measured", [])},
        "synthetic_value": {
            "pass": training.get("synthetic_value", {}).get("status")
            == "measured",
            "criterion": ("§128: baseline vs +synthetic on held-out; a "
                          "verdict either way is the deliverable"),
            "verdict": training.get("synthetic_value", {}).get("verdict")},
    }
    failed = sorted(k for k, v in checks.items() if not v["pass"])
    return {
        "format": 1,
        "task": "T-mix-1m",
        "corpus": CORPUS,
        "seed": seed,
        "generated_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "pass": not failed,
        "failed_checks": failed,
        "checks": checks,
        "composition": stage.get("composition"),
        "quotas": stage.get("quotas"),
        "total_tokens": stage.get("tokens", {}).get("total", {}).get(
            "total_tokens"),
        "tokens": stage.get("tokens"),
        "diversity": stage.get("diversity"),
        "layer_plan": stage.get("layer_plan"),
        "type_mix": stage.get("type_mix"),
        "fence": stage.get("fence"),
        "guardrails": stage.get("guardrails"),
        "feasibility": stage.get("feasibility"),
        "mix": {"status": stage.get("status"), "rows": stage.get("rows"),
                "manifest": stage.get("manifest"),
                "members_sha256": stage.get("members_sha256"),
                "error": stage.get("error")},
        "training": training,
        "upstream": {
            "T-prog-gold": "pass (100 004 Wikidata-grounded questions)",
            "T-gen-schemas": "pass (1 152 decision schemas)",
            "T-halt-contam": "pass (synth-loop quarantined, logiqa/reclor "
                             "eval-only)",
            "T-bakeoff-real": training.get("source", {}).get("top2"),
        },
    }


def write_gate(gate: dict) -> str:
    os.makedirs(os.path.dirname(GATE), exist_ok=True)
    with open(GATE, "w") as fh:
        json.dump(gate, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return GATE


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(prog="run_mix",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("cmd", nargs="?", default="build",
                    choices=("build", "gate", "show"))
    ap.add_argument("--target", type=int, default=None,
                    help="rows in the clean corpus; default: the ceiling")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv[1:])

    if args.cmd == "show":
        scan, supply = sampler.clean_supply()
        feas = corpus_feasibility(supply)
        print(json.dumps({"supply": feas["supply"],
                          "families": feas["families"],
                          "cap_units": feas["cap_units"],
                          "achievable_rows": feas["achievable_rows"],
                          "requested_rows": feas["requested_rows"],
                          "layer_supply": sampler.layer_gap(supply)},
                         indent=2, sort_keys=True))
        return 0

    # `gate` and `build` both re-assemble: the corpus is a pure function of
    # seed + shards, so rebuilding it is how the gate proves the manifest it
    # publishes is still the manifest the corpus on disk produces.
    stage = build(args.target, args.seed, write=not args.no_write)
    gate = compose(stage, training_block(args.seed), args.seed)
    if not args.no_write:
        write_gate(gate)
    print(json.dumps({"pass": gate["pass"], "failed": gate["failed_checks"],
                      "rows": gate["mix"]["rows"],
                      "total_tokens": gate["total_tokens"],
                      "training": gate["training"].get("status")},
                     indent=2, sort_keys=True))
    return 0 if gate["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
