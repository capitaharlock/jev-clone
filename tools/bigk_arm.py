"""The full-space arm against its K≤8 control — #T-bigk-optsets.

One command that scores a matched PAIR of checkpoints and applies the
verdict that `artifacts/gates/T-bigk-optsets/verdict.md` pre-registered
before either number existed:

* **primary** — accuracy at full cardinality (BANKING77, 77 labels, chance
  0.012987) on the frozen DEVELOPMENT cut (`eval.cuts` DEV, n=1 000). Arms
  are chosen here; the reserved cut is read separately, once, with a reason
  (rule R7).
* **abstention** — `unknown` measured in the new regime: the model's
  abstention rate at small K and at K=|space| on the same rows, in this
  same artifact (done-when 5), plus the SUPERVISION rate the sampler emits
  in each regime, which is the other half of the same question.
* **diagnostic** — the trainer's stage eval at K≤8, read out of each
  checkpoint's manifest, so the two arms are compared in the regime every
  phase-1 arm was measured in (rule R1: the divergence is declared, not
  discovered).

Nothing here chooses a threshold after the fact: the verdict function
implements the file that was committed before the run.

CLI:
    .venv-train/bin/python -m tools.bigk_arm report \\
        --arm  artifacts/checkpoints/decision/<full-space run>/stage-000250000 \\
        --control artifacts/checkpoints/decision/<K<=8 run>/stage-000250000 \\
        --device mps
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import mix as mixmod  # noqa: E402
from data.optset import (FULL_SPACE, OptionSetSampler,  # noqa: E402
                         SamplerConfig)
from eval import cuts as CUTS  # noqa: E402
from eval import fullspace as F  # noqa: E402

TASK = "T-bigk-optsets"
GATE_DIR = os.path.join(ROOT, "artifacts", "gates", TASK)
GATE_PATH = os.path.join(GATE_DIR, "gate.json")

#: the old published number this task must place its own next to
OLD_FULLSPACE = {
    "checkpoint": ("artifacts/checkpoints/decision/"
                   "leverstack-d512-prior-ettin-68m-s20260922/"
                   "stage-001000000"),
    "cut": "banking77 official test (RESERVED), 3 080 rows",
    "cardinality": 77,
    "n": 3080,
    "hits": 38,
    "accuracy": 0.012338,
    "accuracy_ci95": [0.009018, 0.016882],
    "chance": 0.012987,
    "beats_chance": False,
    "source": "artifacts/gates/T-teacher-probe/fullspace.json (2026-09-23)",
    "reading": "indistinguishable from chance — the interval contains it",
}

#: K sweep for the abstention/accuracy curve. 77 is the whole space.
SWEEP_KS = (5, 8, 20, 41, 77)
#: datasets whose label space the supervision block is counted on
SUPERVISION_DATASETS = ("huffpost", "massive", "boolq")
SUPERVISION_ROWS = 600


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: str) -> str:
    return os.path.relpath(path, ROOT)


# ------------------------------------------------------------ supervision

def supervision() -> dict:
    """`unknown` as the sampler SUPPLIES it, in both regimes.

    The abstention the head shows is measured on the model below; this is
    the supervision behind it — what fraction of training rows had the gold
    withheld, and what K those rows offered. The full-space answer needed a
    decision and it is written down: an `unknown` row offers the space
    MINUS the withheld gold, so its K is |space| - 1.
    """
    out = {}
    for name, cfg in (("sampled_k3_8", SamplerConfig()),
                      ("full_space", SamplerConfig(k_min=FULL_SPACE,
                                                   k_max=FULL_SPACE))):
        sampler = OptionSetSampler(datasets=SUPERVISION_DATASETS, config=cfg,
                                   rows_per_dataset=SUPERVISION_ROWS)
        samples = list(sampler.epoch(0))
        comp = sampler.composition(samples)
        unknown = [s for s in samples if s.is_unknown]
        out[name] = {
            "regime": comp["regime"],
            "samples": comp["samples"],
            "mean_k": comp["mean_k"],
            "pct_k_is_full_space": comp["pct_k_is_full_space"],
            "pct_unknown_rows": comp["pct_unknown"],
            "target_unknown_pct": 100.0 * cfg.unknown_fraction,
            "mean_k_of_unknown_rows": round(
                sum(s.k for s in unknown) / max(len(unknown), 1), 4),
            "gold_really_absent": all(
                s.dropped_gold not in s.option_ids() for s in unknown),
            "cross_space_violations": len(
                sampler.cross_space_violations(samples)),
            "fenced_label_hits": len(sampler.foreign_label_violations(
                samples, {d: {o["id"] for o in _pool(d)}
                          for d in mixmod.MIX_1M_FENCED
                          if d not in SUPERVISION_DATASETS
                          and _has(d)})),
            "per_dataset_pool_size": {
                d: v["pool_size"] for d, v in comp["per_dataset"].items()},
        }
    return out


def _pool(dataset: str):
    from data.optset import label_pool
    return label_pool(dataset)


def _has(dataset: str) -> bool:
    from data.optset import dataset_path
    return os.path.exists(dataset_path(dataset))


# ----------------------------------------------------------------- scoring

def arm_report(ckpt_dir: str, samples: list, device: str = "auto",
               batch_size: int = 16, log=print) -> dict:
    """Primary + the K/abstention curve for one checkpoint, one cut."""
    from eval import unseen as U
    engine, manifest = U.T.load_checkpoint(ckpt_dir, device)
    primary = F.score(engine, samples, batch_size)
    log(f"[arm] {rel(ckpt_dir)} primary K={primary['cardinality']}: "
        f"acc {primary['accuracy']} (chance {primary['chance']}, "
        f"beats {primary['beats_chance']}, abstain "
        f"{primary['abstain_rate']})")
    curve = {}
    for k in SWEEP_KS:
        rows = samples if k >= primary["cardinality"] else F.resize(samples, k)
        rep = F.score(engine, rows, batch_size)
        rep["lift_over_chance"] = (round(rep["accuracy"] / rep["chance"], 4)
                                   if rep.get("accuracy") and rep["chance"]
                                   else None)
        curve[str(k)] = rep
        log(f"[arm]   K={k}: acc {rep['accuracy']} "
            f"(chance {rep['chance']}, abstain {rep['abstain_rate']})")
    metrics = manifest.get("metrics") or {}
    stage = {cut: {k: v for k, v in (metrics.get(cut) or {}).items()
                   if k in ("n", "accuracy", "accuracy_ci95", "chance",
                            "mean_k", "cardinality", "abstain_rate",
                            "accuracy_options_only", "chance_options_only",
                            "beats_chance", "ece")}
             for cut in ("seen", "unseen")}
    return {
        "checkpoint": rel(ckpt_dir),
        "run_id": manifest.get("run_id"),
        "model_version": manifest.get("model_version"),
        "samples_seen": manifest.get("samples_seen"),
        "cardinality_regime": manifest.get("cardinality_regime") or {
            "mode": "sampled options",
            "recorded": False,
            "why": ("this checkpoint predates the `cardinality_regime` "
                    "block, so the regime is not in its manifest. It is "
                    "stated here from two facts on disk and not from "
                    "memory: the sampler defaults at its commit "
                    "(k_min=3, k_max=8) and the `mean_k` its own stage "
                    "eval recorded, "
                    f"{((manifest.get('metrics') or {}).get('unseen') or {}).get('mean_k')}"),
        },
        "primary_dev": primary,
        "k_curve": curve,
        "abstention": {
            "small_k": {"k": SWEEP_KS[1],
                        "abstain_rate": curve[str(SWEEP_KS[1])]
                        ["abstain_rate"]},
            "full_space": {"k": primary["cardinality"],
                           "abstain_rate": primary["abstain_rate"]},
            "note": ("same rows, same checkpoint, two cardinalities: this "
                     "is the head's abstention, not the sampler's "
                     "supervision rate"),
        },
        "stage_eval_k_le_8": stage,
    }


# ----------------------------------------------------------------- verdict

def verdict(arm: dict, control: dict) -> dict:
    """The rule from `verdict.md`, applied — not re-chosen."""
    p = arm["primary_dev"]
    c = control["primary_dev"]
    beats = bool(p.get("beats_chance"))
    over_control = bool(p.get("accuracy") is not None
                        and c.get("accuracy") is not None
                        and p["accuracy"] > c["accuracy"])
    collapsed = p.get("abstain_rate") is not None and p["abstain_rate"] > 0.95
    arm_unseen = ((arm["stage_eval_k_le_8"].get("unseen") or {})
                  .get("accuracy"))
    ctl_unseen = ((control["stage_eval_k_le_8"].get("unseen") or {})
                  .get("accuracy"))
    diagnostic_fell = bool(arm_unseen is not None and ctl_unseen is not None
                           and arm_unseen < ctl_unseen)
    go = bool(beats and over_control and not collapsed)
    if collapsed:
        reading = ("the head abstains on nearly every row at full "
                   "cardinality, so the primary accuracy is not "
                   "interpretable as a ranking result — rule 4 of the "
                   "pre-registered verdict")
    elif go:
        reading = ("the full-space objective moves the primary above "
                   "chance on the development cut and above its matched "
                   "K<=8 control: the reserved cut may be read once, with "
                   "a reason, and the number published beside the old one")
    else:
        reading = ("the primary interval contains chance: the CARDINALITY "
                   "OF THE OBJECTIVE is discarded as the cause of the "
                   "transfer failure. Next suspect, in order: the "
                   "REPRESENTATION — #T-encoder-finetune first (a frozen "
                   "backbone received no pressure at K<=8; with the whole "
                   "space it does), then backbone capacity (68 M -> "
                   "149 M+). The sampled extension "
                   "(#T-fullspace-objective) stops being the continuation "
                   "of this hypothesis and becomes what the "
                   "non-enumerable spaces need")
    return {
        "go": go,
        "decided_by": ("accuracy_ci95[0] > chance on the development cut "
                       "AND accuracy above the matched control AND the "
                       "head is not abstaining on everything"),
        "pre_registered": rel(os.path.join(GATE_DIR, "verdict.md")),
        "beats_chance": beats,
        "above_control": over_control,
        "abstention_collapse": collapsed,
        "diagnostic_fell_vs_control": diagnostic_fell,
        "reading": reading,
        "r9": ("a NO-GO at this budget LIMITS SPEND AND DOES NOT ESTABLISH "
               "CAUSE. What is discarded is discarded at "
               f"{arm.get('samples_seen')} rows, one seed, frozen "
               "backbone; any causal claim needs the winning arm repeated "
               "on another seed and at a larger budget"),
        "r4": ("this verdict is valid inside the regime it was measured "
               "in; it does not transfer to another one"),
    }


# ----------------------------------------------------------------- compose

def compose(arm: dict, control: dict, sup: dict, cut, cost: dict | None,
            write: bool = True) -> dict:
    doc = {
        "format": "jev.gate.v1",
        "task": TASK,
        "artifact": "gate",
        "generated_utc": utcnow(),
        "question": ("does training with the row's WHOLE label space, "
                     "instead of 3-8 sampled candidates, move the number "
                     "at full cardinality?"),
        "pair": {
            "arm": arm["checkpoint"], "control": control["checkpoint"],
            "only_variable": "the cardinality of the training objective",
            "shared": ("corpus (decision-mix-clean-1m, --fence-clean), "
                       "seed 20260922, mix-seed 20260922, prior penalty "
                       "1.0, ettin-68m frozen, d_model 512, 2 layers, "
                       "8 heads, batch 64, max_length 256"),
            "same_rows": True,
            "same_rows_why": ("same mixture seed and same round-robin "
                           "order, so the compared stages consumed the "
                           "same rows in the same sequence"),
        },
        "cut": {
            "name": cut.name, "dataset": cut.dataset, "split": cut.split,
            "reserved": cut.reserved, "rows": arm["primary_dev"]["n"],
            "cardinality": arm["primary_dev"]["cardinality"],
            "chance": arm["primary_dev"]["chance"],
            "rule": ("R7 — arms are chosen on the development cut; the "
                     "reserved cut is read separately and logged"),
        },
        "primary": {"arm": arm["primary_dev"],
                    "control": control["primary_dev"]},
        "verdict": verdict(arm, control),
        "arm": arm,
        "control": control,
        "previously_published": OLD_FULLSPACE,
        "unknown_supervision": sup,
        "cost": cost or {"see": rel(os.path.join(GATE_DIR, "cost.json"))},
        "rules_honoured": [
            "R1 — the training regime and the stage-eval regime are both "
            "declared in each checkpoint's `cardinality_regime`; the "
            "primary metric is measured at full cardinality",
            "R2 — every accuracy here carries its K, its chance and its "
            "95 % interval",
            "R3 — the teacher's 0.924 is a citation, not a measurement "
            "made here; it is not restated in this file as a comparison",
            "R4 — the verdict names the regime it is valid in",
            "R5 — the slope is shown at a small budget before anything is "
            "scaled",
            "R7 — this artifact is the development cut; reads of the "
            "reserved cut are logged in "
            "artifacts/gates/T-eval-cardinality/test-queries.json",
            "R9 — a cheap NO-GO limits spend and does not establish cause",
        ],
    }
    if write:
        os.makedirs(GATE_DIR, exist_ok=True)
        with open(GATE_PATH, "w") as fh:
            json.dump(doc, fh, indent=2, ensure_ascii=False, sort_keys=True)
            fh.write("\n")
    return doc


def report(arm_ckpt: str, control_ckpt: str, device: str = "auto",
           limit: int | None = None, batch_size: int = 16,
           write: bool = True, log=print) -> dict:
    cut = CUTS.DEV
    samples = CUTS.samples(cut, limit)
    log(f"[arm] development cut: {len(samples)} rows, "
        f"K={len(samples[0].options) if samples else 0}")
    arm = arm_report(arm_ckpt, samples, device, batch_size, log)
    control = arm_report(control_ckpt, samples, device, batch_size, log)
    sup = supervision()
    cost_path = os.path.join(GATE_DIR, "cost.json")
    cost = (json.load(open(cost_path)) if os.path.exists(cost_path) else None)
    cost = ({"per_k": cost["per_k"], "device": cost["device"],
             "slowdown_vs_k8": cost["slowdown_vs_k8"],
             "head_slowdown_vs_k8": cost["head_slowdown_vs_k8"],
             "artifact": rel(cost_path)} if cost else None)
    doc = compose(arm, control, sup, cut, cost, write)
    log(f"[arm] verdict: {'GO' if doc['verdict']['go'] else 'NO-GO'}")
    log(f"[arm] {doc['verdict']['reading']}")
    if write:
        log(f"[arm] artifact: {rel(GATE_PATH)}")
    return doc


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(prog="tools.bigk_arm")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="score the pair and write the gate")
    r.add_argument("--arm", required=True)
    r.add_argument("--control", required=True)
    r.add_argument("--device", default="auto")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--batch-size", type=int, default=16)
    r.add_argument("--no-write", action="store_true")
    sub.add_parser("supervision",
                   help="the `unknown` supervision block only")
    args = ap.parse_args(argv[1:])
    if args.cmd == "supervision":
        print(json.dumps(supervision(), indent=2, ensure_ascii=False))
        return 0
    doc = report(args.arm, args.control, args.device, args.limit or None,
                 args.batch_size, write=not args.no_write)
    print(json.dumps({"primary": doc["primary"], "verdict": doc["verdict"]},
                     indent=2, ensure_ascii=False))
    return 0 if doc["verdict"]["go"] else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
