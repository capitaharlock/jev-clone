"""The five arms of #T-gen-objective, and the rule that picks a winner.

`#T-antiscale-diag` measured what breaks: 75.4 % of the 250 k -> 1 M
unseen fall is a RANKING collapse (axis 4) and 100 % of the 1 M mixture is
answerable by a text -> label map (axis 1). Every extra row is another row
of that map, so the objective gets exactly what it optimises. The four
candidates of the task attack that incentive; the backbone stays FROZEN in
all five arms (capacity is `#T-unfreeze-backbone`, not this task).

Comparability — the part that makes the comparison mean anything
----------------------------------------------------------------
All five arms are ONE trainer invocation each, differing by exactly one
flag. They share:

* `SEED` — the training seed AND, through `--fence-clean`, the corpus seed
  (`data.mix.CLEAN_1M_SEED`), so the mixture is the published
  `decision-mix-clean-1m` in every arm;
* `BUDGET` decisions, taken from the FRONT of the same deterministic
  `MixtureStream` order — the arms see the same rows in the same order;
* the same eval samplers (`seen` / `unseen` cuts of the trainer's own
  sibling-label holdout) at the same stage points.

`compose()` asserts that: `arms_comparable` fails the gate if two arms
disagree on the mixture digest, the seed, the backbone or the budget. A
declared-identical subset that was not identical is exactly the kind of
thing this repo has published before.

Why CPU / ettin-68m, and why 125 k
----------------------------------
MPS is held by the `antiscale-wide` job (axis 3 of the ablation), and the
measured throughput of `ettin-68m` on CPU (117 decisions/s on
`train-real-v1`) is as fast as `modernbert-base` on MPS (74/s). So the two
measurements run side by side instead of one waiting ~14 h for the other.
125 000 decisions is the largest budget that fits the time box for five
arms, and it buys TWO points of `DEFAULT_STAGES` (62 500 and 125 000) —
a SLOPE per arm, which is what the task asks about, not a single point.

It is NOT the 1 M point where the fall is deepest. That is stated in the
gate as a limit of this measurement, and the adopted candidate (if any)
earns the 1 M run; it is not extrapolated here.

The adoption rule, written before the first number
--------------------------------------------------
A candidate is ELIGIBLE at the final budget point iff BOTH:

1. Wilson 95 % lower bound of unseen accuracy  >  mean chance 1/(K+1);
2. Wilson 95 % lower bound of unseen ranking   >  mean chance 1/K
   (`accuracy_options_only`, the `unknown` logit out of the race).

Criterion 2 is the task's own: "no basta con acertar, el orden tiene que
ser informativo". Among eligible candidates the adopted one is the highest
unseen accuracy at the final point, ties broken by ranking. If none is
eligible the gate is `pass: false`, `adopted: null` — a valid, honest
outcome — and every arm keeps its number and its numeric discard reason.

CLI:
    .venv-train/bin/python -m tools.gen_objective.arms run     # train + gate
    python3 -m tools.gen_objective.arms gate                   # compose only
    python3 -m tools.gen_objective.arms plan                   # print argv
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.mix import CLEAN_1M_SEED  # noqa: E402

TASK = "T-gen-objective"
VENV = os.path.join(ROOT, ".venv-train", "bin", "python")
RUNS = os.path.join(ROOT, "artifacts", "runs")
GATE_DIR = os.path.join(ROOT, "artifacts", "gates", TASK)
GATE_PATH = os.path.join(GATE_DIR, "gate.json")

#: shared by every arm — the declaration the gate checks back against
SEED = 20260922
BUDGET = 125_000
BACKBONE = "ettin-68m"
DEVICE = "cpu"
BATCH_SIZE = 64
EVAL_SAMPLES = 3000
#: torch CPU threads per arm. The five arms run CONCURRENTLY — MPS is held
#: by `antiscale-wide`, so the CPU is the resource, and five arms sharing
#: 16 cores three at a time finish the 62 500 stage of every arm in the time
#: one sequential arm would take to finish alone. The cap is identical for
#: every arm, so no arm is advantaged and the float reduction order (which
#: thread count does change) is the same in all five.
THREADS_PER_ARM = 3
#: the stage points `train_decision.DEFAULT_STAGES` reaches under BUDGET
STAGE_POINTS = (62_500, 125_000)

#: chance is a property of the eval cut, not of an arm; it is read back per
#: arm from the run and cross-checked, never assumed from these
CHANCE_ACCURACY_EXPECTED = 0.165
CHANCE_RANKING_EXPECTED = 0.202

#: `(id, flags, candidate number in the task body, what it does)`
ARMS = (
    {"id": "baseline", "candidate": 0, "flags": [],
     "what": "listwise cross-entropy over [K+1], unchanged — the control"},
    {"id": "label-dropout", "candidate": 1,
     "flags": ["--label-dropout", "0.2"],
     "what": ("candidate 1: every option text is replaced with probability 0.2 by "
              "a novel sentinel string, gold included, positions and "
              "gold_index untouched — a memorised text -> label map cannot "
              "answer the replaced rows")},
    {"id": "episodic", "candidate": 2, "flags": ["--episodic-resample"],
     "what": ("candidate 2: the distractor SET is redrawn per step from "
              "#T-optset-sampler's own hard/easy buckets, gold kept and "
              "reshuffled — the same question is never scored against the "
              "same offer set twice")},
    {"id": "contrastive", "candidate": 3,
     "flags": ["--contrastive-weight", "0.5"],
     "what": ("candidate 3: 0.5 x question<->option-TEXT InfoNCE (tau "
              "0.07) over the head's own q_proj/opt_proj, added to the "
              "listwise CE — the signal is text, not index")},
    {"id": "prior", "candidate": 4, "flags": ["--prior-penalty", "1.0"],
     "what": ("candidate 4: each option logit is discounted by "
              "log(train gold count) of that label, `unknown` untouched — "
              "a frequent label stops winning by being frequent")},
)


def run_id(arm: str, seed: int = SEED) -> str:
    return f"genobj-{arm}-{BACKBONE}-s{seed}"


def train_command(arm: dict, budget: int = BUDGET, seed: int = SEED) -> list:
    """One arm's argv. Everything but `arm['flags']` is shared, by design."""
    return [VENV, "-m", "training.python.train_decision", "train",
            "--run-id", run_id(arm["id"], seed),
            "--backbone", BACKBONE,
            "--max-samples", str(budget),
            "--batch-size", str(BATCH_SIZE),
            "--seed", str(seed),
            "--mix-seed", str(CLEAN_1M_SEED),
            "--device", DEVICE,
            "--eval-samples", str(EVAL_SAMPLES),
            "--log-every", "50",
            "--fence-clean",
            "--no-gate"] + list(arm["flags"])


def arm_env(threads: int = THREADS_PER_ARM) -> dict:
    """Identical thread caps for every arm — see `THREADS_PER_ARM`."""
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": str(threads),
                "MKL_NUM_THREADS": str(threads),
                "VECLIB_MAXIMUM_THREADS": str(threads),
                "TORCH_NUM_THREADS": str(threads),
                "PYTORCH_ENABLE_MPS_FALLBACK": "0"})
    return env


def run_dir(arm: str, seed: int = SEED) -> str:
    return os.path.join(RUNS, run_id(arm, seed))


def read_stages(arm: str, seed: int = SEED) -> dict:
    """Stage records of an arm from `metrics.jsonl` — live runs included."""
    path = os.path.join(run_dir(arm, seed), "metrics.jsonl")
    out: dict = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:  # a half-written last line of a live run
                continue
            if rec.get("t") == "stage":
                out[int(rec["stage"])] = rec
    return out


def read_run(arm: str, seed: int = SEED) -> dict:
    path = os.path.join(run_dir(arm, seed), "run.json")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


# -- one arm's numbers -----------------------------------------------------

def _point(record: dict) -> dict:
    """One stage record reduced to the numbers the gate publishes."""
    unseen, seen = record.get("unseen", {}), record.get("seen", {})
    return {
        "samples": record.get("samples"),
        "tokens": record.get("tokens"),
        "unseen_n": unseen.get("n"),
        "unseen_accuracy": unseen.get("accuracy"),
        "unseen_accuracy_ci95": unseen.get("accuracy_ci95"),
        "unseen_chance": unseen.get("chance"),
        "unseen_ranking": unseen.get("accuracy_options_only"),
        "unseen_ranking_ci95": unseen.get("accuracy_options_only_ci95"),
        "unseen_ranking_chance": unseen.get("chance_options_only"),
        "unseen_abstain_rate": unseen.get("abstain_rate"),
        "unseen_ece": unseen.get("ece"),
        "unseen_nll": unseen.get("nll"),
        "unseen_per_dataset": unseen.get("per_dataset"),
        "seen_accuracy": seen.get("accuracy"),
        "seen_n": seen.get("n"),
        "checkpoint": record.get("checkpoint"),
        "model_version": record.get("model_version"),
        "elapsed_s": record.get("elapsed_s"),
    }


def _margin(point: dict) -> dict:
    """Signed distance of each Wilson lower bound from its own chance.

    Positive = the bound clears chance, which is what eligibility asks.
    Both criteria are evaluated on the LOWER bound, never on the point
    estimate: an unseen cut of ~3 000 rows has a CI wide enough that a
    point estimate above chance means nothing on its own.
    """
    acc_lo = (point.get("unseen_accuracy_ci95") or [0.0, 0.0])[0]
    rank_lo = (point.get("unseen_ranking_ci95") or [0.0, 0.0])[0]
    acc_chance = point.get("unseen_chance") or 0.0
    rank_chance = point.get("unseen_ranking_chance") or 0.0
    return {
        "accuracy_ci95_lower": round(acc_lo, 6),
        "accuracy_chance": round(acc_chance, 6),
        "accuracy_margin": round(acc_lo - acc_chance, 6),
        "accuracy_beats_chance": bool(acc_lo > acc_chance),
        "ranking_ci95_lower": round(rank_lo, 6),
        "ranking_chance": round(rank_chance, 6),
        "ranking_margin": round(rank_lo - rank_chance, 6),
        "ranking_beats_chance": bool(rank_lo > rank_chance),
    }


def _slope(points: list) -> dict:
    """First -> last stage delta: the quantity the task calls la pendiente."""
    if len(points) < 2:
        return {"measured": False,
                "why": "a slope needs two stage points; this arm has "
                       f"{len(points)}"}
    a, b = points[0], points[-1]
    return {
        "measured": True,
        "from_samples": a["samples"], "to_samples": b["samples"],
        "unseen_accuracy": round((b["unseen_accuracy"] or 0.0)
                                 - (a["unseen_accuracy"] or 0.0), 6),
        "unseen_ranking": round((b["unseen_ranking"] or 0.0)
                                - (a["unseen_ranking"] or 0.0), 6),
        "unseen_abstain_rate": round((b["unseen_abstain_rate"] or 0.0)
                                     - (a["unseen_abstain_rate"] or 0.0), 6),
    }


def _discard_reason(margin: dict) -> str:
    """The NUMERIC reason an arm is not adopted. Never a verdict word."""
    parts = []
    if not margin["accuracy_beats_chance"]:
        parts.append(
            f"unseen accuracy CI95 lower bound {margin['accuracy_ci95_lower']}"
            f" does not clear chance {margin['accuracy_chance']} "
            f"(short by {abs(margin['accuracy_margin'])})")
    if not margin["ranking_beats_chance"]:
        parts.append(
            f"unseen ranking CI95 lower bound {margin['ranking_ci95_lower']}"
            f" does not clear its own chance {margin['ranking_chance']} "
            f"(short by {abs(margin['ranking_margin'])})")
    return "; ".join(parts)


def arm_report(arm: dict, seed: int = SEED, budget: int = BUDGET) -> dict:
    """One arm: every stage it reached, or why it has no number at all."""
    rid = run_id(arm["id"], seed)
    stages = read_stages(arm["id"], seed)
    run = read_run(arm["id"], seed)
    out = {
        "candidate": arm["candidate"],
        "what": arm["what"],
        "flags": list(arm["flags"]),
        "run_id": rid,
        "command": " ".join(train_command(arm, budget, seed)[1:]),
        "gen_objective": run.get("gen_objective"),
    }
    points = [_point(stages[s]) for s in sorted(stages)]
    if not points:
        out.update({
            "measured": False,
            "not_measured": (
                f"no stage record exists for {rid}: "
                + ("the run has not started" if not run else
                   "the run started but reached no stage point yet")),
            "why_not_estimated": (
                "an arm without a stage record has NO number here. It is "
                "not interpolated from its neighbours and not carried over "
                "from another arm — #T-mix-1m exists because a number "
                "nobody measured was published as one that was"),
        })
        return out
    final = points[-1]
    margin = _margin(final)
    complete = final["samples"] is not None and final["samples"] >= budget
    out.update({
        "measured": True,
        "complete": bool(complete),
        "budget_decisions": budget,
        "points": points,
        "final": final,
        "final_margin": margin,
        "slope": _slope(points),
        "eligible": bool(margin["accuracy_beats_chance"]
                         and margin["ranking_beats_chance"]),
    })
    if not complete:
        out["partial"] = (
            f"the run is at {final['samples']} of {budget} decisions; the "
            f"numbers above are the stages it HAS reached, and the final "
            f"point is not the budget point")
    if not out["eligible"]:
        out["discard_reason"] = _discard_reason(margin)
    return out


# -- the gate -------------------------------------------------------------

ADOPTION_RULE = {
    "pre_registered": "written in tools/gen_objective/arms.py before the "
                      "first arm was trained",
    "criterion_1": "Wilson 95 % lower bound of unseen accuracy > mean "
                   "chance 1/(K+1) at the final budget point",
    "criterion_2": "Wilson 95 % lower bound of unseen ranking "
                   "(accuracy_options_only, `unknown` out of the race) > "
                   "its own mean chance 1/K — the task's own second gate: "
                   "'no basta con acertar, el orden tiene que ser "
                   "informativo'",
    "tie_break": "among eligible arms, highest unseen accuracy at the "
                 "final point; ties broken by unseen ranking",
    "if_none_eligible": "pass: false, adopted: null. That is a valid "
                        "result: it says the four candidates do not move "
                        "the incentive at this budget, and no winner is "
                        "forced",
}


def choose(reports: dict) -> tuple:
    """(adopted arm id or None, ranking of the eligible arms)."""
    eligible = [(rid, r) for rid, r in reports.items()
                if r.get("measured") and r.get("eligible")
                and r.get("complete")]
    eligible.sort(key=lambda t: (-(t[1]["final"]["unseen_accuracy"] or 0.0),
                                 -(t[1]["final"]["unseen_ranking"] or 0.0),
                                 t[0]))
    return (eligible[0][0] if eligible else None,
            [{"arm": rid,
              "unseen_accuracy": r["final"]["unseen_accuracy"],
              "unseen_ranking": r["final"]["unseen_ranking"]}
             for rid, r in eligible])


def _comparability(reports: dict, budget: int, seed: int) -> dict:
    """Every measured arm must share seed, backbone, corpus digest, budget.

    A sweep whose arms differ by more than their one flag measures the
    difference between two corpora and calls it the difference between two
    objectives. So the digests are compared, not trusted.
    """
    axes: dict = {}
    for rid, report in reports.items():
        if not report.get("measured"):
            continue
        run = read_run(rid, seed)
        axes[rid] = {
            "seed": run.get("seed"),
            "backbone": (run.get("backbone") or {}).get("id"),
            "frozen": (run.get("backbone") or {}).get("frozen"),
            "members_sha256": (run.get("mix") or {}).get("members_sha256"),
            "mix_seed": (run.get("mix") or {}).get("seed"),
            "max_samples": run.get("max_samples"),
            "epoch_samples": run.get("epoch_samples"),
            "batch_size": run.get("batch_size"),
            "lr": run.get("lr"),
            "d_model": (run.get("architecture") or {}).get("d_model"),
            "unseen_chance": (report.get("final") or {}).get("unseen_chance"),
        }
    shared = [k for k in ("seed", "backbone", "frozen", "members_sha256",
                          "mix_seed", "max_samples", "epoch_samples",
                          "batch_size", "lr", "d_model", "unseen_chance")]
    disagreements = {}
    for key in shared:
        values = {rid: a.get(key) for rid, a in axes.items()}
        distinct = {json.dumps(v, sort_keys=True) for v in values.values()}
        if len(distinct) > 1:
            disagreements[key] = values
    return {
        "pass": not disagreements and len(axes) > 1,
        "criterion": ("every measured arm shares seed, backbone, frozen "
                      "flag, corpus digest, mix seed, budget, batch size, "
                      "lr, head width and eval chance — only its one flag "
                      "differs"),
        "n_measured_arms": len(axes),
        "per_arm": axes,
        "disagreements": disagreements,
        "why_it_fails_when_one_arm": ("a single arm cannot be compared to "
                                      "anything" if len(axes) <= 1 else None),
    }


def compose(budget: int = BUDGET, seed: int = SEED,
            write: bool = True) -> dict:
    """Assemble `artifacts/gates/T-gen-objective/gate.json` from disk."""
    reports = {a["id"]: arm_report(a, seed, budget) for a in ARMS}
    adopted, ranking = choose(reports)
    measured = [r for r in reports.values() if r.get("measured")]
    complete = [r for r in measured if r.get("complete")]
    unmeasured = sorted(rid for rid, r in reports.items()
                        if not r.get("measured"))
    partial = sorted(rid for rid, r in reports.items()
                     if r.get("measured") and not r.get("complete"))
    baseline = reports.get("baseline", {})
    checks = {
        "arms_comparable": _comparability(reports, budget, seed),
        "all_arms_measured": {
            "pass": not unmeasured and not partial,
            "criterion": ("all five arms reach the declared budget; an arm "
                          "that does not is published `measured: false` (or "
                          "`complete: false`) with its reason, never "
                          "estimated"),
            "n_arms": len(ARMS),
            "n_measured": len(measured),
            "n_complete": len(complete),
            "not_measured": unmeasured,
            "partial": partial,
        },
        "adopted_beats_chance": {
            "pass": adopted is not None,
            "criterion": ADOPTION_RULE["criterion_1"] + " AND "
                         + ADOPTION_RULE["criterion_2"],
            "adopted": adopted,
            "eligible_ranking": ranking,
            "why_false": (None if adopted is not None else
                          "no arm clears both pre-registered bounds at the "
                          "final budget point; the four candidates are all "
                          "recorded with their numeric discard reason and "
                          "none is adopted"),
        },
        "candidate_beats_baseline": {
            "pass": bool(adopted is not None and adopted != "baseline"),
            "criterion": ("the adopted arm is one of the four candidates, "
                          "not the control: a baseline that wins means the "
                          "interventions cost accuracy"),
            "baseline_final": baseline.get("final", {}).get(
                "unseen_accuracy") if baseline.get("measured") else None,
            "baseline_slope": baseline.get("slope"),
            "deltas_vs_baseline": {
                rid: {
                    "unseen_accuracy": round(
                        (r["final"]["unseen_accuracy"] or 0.0)
                        - (baseline["final"]["unseen_accuracy"] or 0.0), 6),
                    "unseen_ranking": round(
                        (r["final"]["unseen_ranking"] or 0.0)
                        - (baseline["final"]["unseen_ranking"] or 0.0), 6),
                }
                for rid, r in reports.items()
                if rid != "baseline" and r.get("measured")
                and baseline.get("measured")
            },
        },
    }
    gate = {
        "task": TASK,
        "pass": all(c["pass"] for c in checks.values()),
        "adopted": adopted,
        "backbone_frozen": True,
        "declared_subset": {
            "budget_decisions": budget,
            "stage_points": list(STAGE_POINTS),
            "corpus": "decision-mix-clean-1m (#T-mix-1m), --fence-clean",
            "seed": seed,
            "mix_seed": CLEAN_1M_SEED,
            "backbone": BACKBONE,
            "device": DEVICE,
            "batch_size": BATCH_SIZE,
            "eval_samples": EVAL_SAMPLES,
            "threads_per_arm": THREADS_PER_ARM,
            "arms_run_concurrently": True,
            "same_subset_for_every_arm": (
                "all five arms consume the FRONT of the same deterministic "
                "MixtureStream order, so they see the same decisions in the "
                "same sequence; `arms_comparable` checks the digest"),
            "why_not_1m": (
                f"the fall #T-antiscale-diag measured is deepest at 1 M, and "
                f"this measurement is at {budget:,}. Five 1 M arms are ~13 h "
                f"of wallclock on a machine whose MPS is held by the "
                f"`antiscale-wide` job (axis 3 of the same ablation), so the "
                f"arms run on CPU at a budget that buys TWO stage points — a "
                f"slope — per arm. The adopted candidate earns the 1 M run; "
                f"nothing here is extrapolated to 1 M"),
            "why_cpu_ettin": (
                "measured throughput: ettin-68m on CPU did 1 M decisions in "
                "8 540 s on `train-real-v1` (117/s), against 74/s for "
                "modernbert-base on MPS. Running the arms on CPU costs "
                "nothing in speed and leaves MPS to the axis-3 job"),
        },
        "measured_baseline_on_the_same_subset": (
            "the control arm is trained here, not read off the 1 M run's "
            f"stage at {budget:,}: that stage belongs to a cosine schedule "
            "over 1 M steps and is not the same model as a run whose budget "
            "IS this one"),
        "adoption_rule": ADOPTION_RULE,
        "chance": {
            "expected_accuracy": CHANCE_ACCURACY_EXPECTED,
            "expected_ranking": CHANCE_RANKING_EXPECTED,
            "note": ("chance is a property of the eval cut and is read back "
                     "per arm from its own run; `arms_comparable` fails if "
                     "two arms disagree on it"),
        },
        "arms": reports,
        "checks": checks,
        "verdict": ("GO: adopt " + adopted if adopted
                    else "NO-GO: no candidate clears both pre-registered "
                         "bounds"),
        "honesty": (
            "every number here comes from a stage record in "
            "artifacts/runs/<run_id>/metrics.jsonl. An arm with no record "
            "is `measured: false` with its reason; a discarded arm keeps "
            "its numbers and its numeric discard reason instead of being "
            "deleted. No arm's number is estimated from another's"),
        "next": [
            ("run the adopted candidate at 1 M so the gain is measured "
             "where the slope is negative today, then re-evaluate "
             "#T-release-gate on that checkpoint")
            if adopted else
            ("no candidate is adopted: the next move is NOT a 1 M run of a "
             "losing arm. #T-antiscale-diag axis 3 (job `antiscale-wide`) "
             "decides whether #T-unfreeze-backbone goes next"),
        ],
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if write:
        os.makedirs(GATE_DIR, exist_ok=True)
        with open(GATE_PATH, "w") as fh:
            json.dump(gate, fh, indent=2, sort_keys=True)
            fh.write("\n")
    return gate



# -- the gate's own guardrail ---------------------------------------------

#: keys that only a MEASURED arm may carry. An arm without a stage record
#: must not hold any of them — that is how a number nobody measured gets
#: into a gate, and it has happened in this repo before (#T-mix-1m).
NUMERIC_ARM_KEYS = ("points", "final", "final_margin", "slope", "eligible",
                    "complete", "discard_reason")


def audit_gate(gate: dict, seed: int = SEED,
               strict_disk: bool = True) -> dict:
    """Refuse a gate whose numbers are not on disk. Structural + factual.

    Structural (always): an arm with `measured: false` carries no numeric
    key at all, and the adopted arm is a measured, complete, eligible one.

    Factual (`strict_disk`): every published point of every measured arm
    must match a stage record in that arm's own `metrics.jsonl` — same
    `samples`, same `unseen_accuracy`, same `unseen_ranking`. A measured
    arm whose run directory does not exist is a violation, not a gap: it
    is exactly the shape a fabricated number has.

    `artifacts/runs/` is gitignored, so a fresh clone cannot run the
    factual half; callers pass `strict_disk=False` there and say so.
    """
    violations = []
    arms = gate.get("arms") or {}
    if not arms:
        violations.append("the gate publishes no arms at all")
    for rid, report in sorted(arms.items()):
        if not report.get("measured"):
            if not report.get("not_measured"):
                violations.append(
                    f"{rid}: measured is false but no `not_measured` reason "
                    f"is given")
            leaked = [k for k in NUMERIC_ARM_KEYS if k in report]
            if leaked:
                violations.append(
                    f"{rid}: not measured, yet carries {leaked} — a number "
                    f"without a measurement behind it")
            continue
        for key in ("points", "final", "run_id"):
            if not report.get(key):
                violations.append(f"{rid}: measured but has no {key!r}")
        if not strict_disk:
            continue
        stages = read_stages(rid, seed)
        if not stages:
            violations.append(
                f"{rid}: measured, but no stage record exists on disk for "
                f"run_id {report.get('run_id')!r}")
            continue
        for point in report.get("points") or []:
            record = stages.get(point.get("samples"))
            if record is None:
                violations.append(
                    f"{rid}: publishes a point at {point.get('samples')} "
                    f"decisions that no stage record on disk carries")
                continue
            unseen = record.get("unseen", {})
            for gate_key, disk_key in (("unseen_accuracy", "accuracy"),
                                       ("unseen_ranking",
                                        "accuracy_options_only")):
                if point.get(gate_key) != unseen.get(disk_key):
                    violations.append(
                        f"{rid} @ {point.get('samples')}: gate says "
                        f"{gate_key}={point.get(gate_key)}, disk says "
                        f"{unseen.get(disk_key)}")
    adopted = gate.get("adopted")
    if adopted is not None:
        report = arms.get(adopted)
        if report is None:
            violations.append(f"adopted {adopted!r} is not one of the arms")
        elif not (report.get("measured") and report.get("complete")
                  and report.get("eligible")):
            violations.append(
                f"adopted {adopted!r} is not a measured, complete, eligible "
                f"arm: measured={report.get('measured')} "
                f"complete={report.get('complete')} "
                f"eligible={report.get('eligible')}")
    return {"pass": not violations, "violations": violations,
            "strict_disk": strict_disk,
            "criterion": ("no arm carries a number it did not measure, and "
                          "every published number equals the stage record "
                          "on disk")}


# -- CLI ------------------------------------------------------------------

def main(argv: list) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="gen_objective.arms",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("run", "gate", "plan", "audit"),
                    nargs="?",
                    default="gate")
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--only", default=None, help="run only this arm id")
    ap.add_argument("--threads", type=int, default=THREADS_PER_ARM)
    ap.add_argument("--parallel", action="store_true",
                    help=("run every arm at once with THREADS_PER_ARM "
                          "threads each, instead of one after another. The "
                          "shared stage points then land for all five arms "
                          "at about the same time, so a sweep that runs out "
                          "of wallclock still publishes a COMPLETE five-arm "
                          "comparison at the stage it reached"))
    args = ap.parse_args(argv[1:])

    if args.cmd == "audit":
        if not os.path.exists(GATE_PATH):
            print(f"{os.path.relpath(GATE_PATH, ROOT)} does not exist")
            return 1
        with open(GATE_PATH) as fh:
            report = audit_gate(json.load(fh), args.seed)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["pass"] else 1

    if args.cmd == "plan":
        for arm in ARMS:
            print(f"# arm {arm['id']} — candidate {arm['candidate']}")
            print(" ".join(train_command(arm, args.budget, args.seed)), "\n")
        return 0

    if args.cmd == "run":
        todo = [a for a in ARMS
                if (not args.only or a["id"] == args.only)
                and not os.path.exists(os.path.join(
                    run_dir(a["id"], args.seed), "summary.json"))]
        for arm in ARMS:
            if arm not in todo and (not args.only or arm["id"] == args.only):
                print(f"[arms] {arm['id']}: already complete, skipping")
        env = arm_env(args.threads)
        procs = {}
        for arm in todo:
            cmd = train_command(arm, args.budget, args.seed)
            print(f"[arms] start {arm['id']}: {' '.join(cmd)}", flush=True)
            procs[arm["id"]] = subprocess.Popen(cmd, cwd=ROOT, env=env)
            if not args.parallel:
                code = procs.pop(arm["id"]).wait()
                gate = compose(args.budget, args.seed)
                print(f"[arms] {arm['id']}: exit {code}; gate pass="
                      f"{gate['pass']} adopted={gate['adopted']}", flush=True)
        # a gate after every arm that lands, so a killed sweep still leaves
        # the stages it reached published instead of nothing
        while procs:
            for rid in list(procs):
                code = procs[rid].poll()
                if code is None:
                    continue
                procs.pop(rid)
                gate = compose(args.budget, args.seed)
                print(f"[arms] {rid}: exit {code}; gate pass={gate['pass']} "
                      f"adopted={gate['adopted']}", flush=True)
                if code != 0:
                    print(f"[arms] {rid} failed; the gate records what it "
                          f"reached, and nothing it did not", flush=True)
            if procs:
                time.sleep(30)
        compose(args.budget, args.seed)
        return 0

    gate = compose(args.budget, args.seed)
    print(json.dumps({k: v for k, v in gate.items() if k != "arms"},
                     indent=2, sort_keys=True))
    for rid, report in gate["arms"].items():
        if not report.get("measured"):
            print(f"[arm] {rid}: NOT MEASURED — {report['not_measured']}")
            continue
        f = report["final"]
        print(f"[arm] {rid}: n={f['unseen_n']} samples={f['samples']} "
              f"unseen={f['unseen_accuracy']} {f['unseen_accuracy_ci95']} "
              f"rank={f['unseen_ranking']} {f['unseen_ranking_ci95']} "
              f"eligible={report['eligible']}")
        if report.get("discard_reason"):
            print(f"        discarded: {report['discard_reason']}")
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
