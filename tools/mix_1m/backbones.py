"""Top-2 backbones on the §62 curve, measured by the real trainer.

Nothing in this module trains a proxy. The two candidates come from the
#T-bakeoff-real Pareto (`artifacts/gates/T-bakeoff/report.json`), which
already refuses to publish a row with `pending_weights` or `random_init`,
and they are trained by `training/python/train_decision.py` — the same
listwise trainer the production run uses — over
`decision-mix-clean-1m`.

Two stages of §62 come out of ONE run per backbone: the trainer
checkpoints and evaluates at every point of `DEFAULT_STAGES`, so Stage 0
(250 k decisions) and Stage 1 (1 M) are two records of the same curve
rather than two runs that would differ by more than their budget.

The §128 synthetic-value question is answered the same way: two real runs
at an identical budget, one with the grounded-synthetic source and one
with `--drop-dataset synth-v1`, compared on the SAME held-out cut. If
held-out quality does not improve, the verdict is NO-GO and the generator
is not scaled.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

BAKEOFF = os.path.join(ROOT, "artifacts", "gates", "T-bakeoff", "report.json")
RUNS = os.path.join(ROOT, "artifacts", "runs")
VENV = os.path.join(ROOT, ".venv-train", "bin", "python")

#: §62: Stage 0 validates the architecture, Stage 1 has to show the model
#: generalises past a dataset. Both are decisions SEEN, not corpus rows.
STAGES = {"stage0": 250_000, "stage1": 1_000_000}

#: §113 — what a backbone row has to publish.
METRICS = ("accuracy", "nll", "brier", "ece")

#: a row carrying any of these is not a measurement
PROXY_MARKERS = ("pending_weights", "random_init", "harness_only")


class NoBakeoff(RuntimeError):
    """The top-2 decision has not been published, so there is none to use."""


def load_bakeoff(path: str = BAKEOFF) -> dict:
    if not os.path.exists(path):
        raise NoBakeoff(f"{os.path.relpath(path, ROOT)} does not exist: "
                        f"#T-bakeoff-real has not published a Pareto")
    with open(path) as fh:
        return json.load(fh)


def top2(report: dict | None = None) -> list:
    """The two backbones #T-bakeoff-real chose, refusing any proxy row."""
    report = report if report is not None else load_bakeoff()
    chosen = list(report.get("top2") or [])
    if len(chosen) != 2:
        raise NoBakeoff(f"the bake-off report names {len(chosen)} top "
                        f"backbones, not 2: {chosen}")
    rows = {r["id"]: r for r in report.get("pareto", {}).get("rows", [])}
    for candidate in chosen:
        row = rows.get(candidate)
        if row is None:
            raise NoBakeoff(f"{candidate!r} is named top-2 but has no Pareto "
                            f"row with lineage behind it")
        proxy = [m for m in PROXY_MARKERS if m in row]
        if proxy or row.get("status") != "trained":
            raise NoBakeoff(f"{candidate!r} is a proxy row ({proxy or row.get('status')}): "
                            f"#T-mix-1m trains measured backbones only")
    return chosen


def lineage(report: dict, backbone: str) -> dict:
    for row in report.get("pareto", {}).get("rows", []):
        if row["id"] == backbone:
            return {"repo": row.get("repo"), "revision": row.get("revision"),
                    "license": row.get("license"),
                    "bakeoff_run_id": row.get("lineage", {}).get("run_id"),
                    "bakeoff_unseen_accuracy":
                        row.get("quality", {}).get("unseen_accuracy")}
    return {}


def train_command(backbone: str, run_id: str, max_samples: int,
                  seed: int = 1789, device: str = "cpu",
                  batch_size: int = 64, eval_samples: int = 3000,
                  drop: tuple = (), fence_clean: bool = True) -> list:
    """The real trainer's argv for one curve run. No mock loop exists here."""
    cmd = [VENV, "-m", "training.python.train_decision", "train",
           "--backbone", backbone, "--run-id", run_id,
           "--max-samples", str(max_samples), "--seed", str(seed),
           "--device", device, "--batch-size", str(batch_size),
           "--eval-samples", str(eval_samples), "--log-every", "25",
           "--no-gate"]
    if fence_clean:
        cmd.append("--fence-clean")
    for dataset in drop:
        cmd += ["--drop-dataset", dataset]
    return cmd


def run_dir(run_id: str) -> str:
    return os.path.join(RUNS, run_id)


def read_stages(run_id: str) -> dict:
    """Every §62 stage a run has reached so far, from its metrics log.

    Reads `metrics.jsonl` rather than `summary.json` so a run that is
    still going reports the stages it HAS reached instead of nothing.
    """
    path = os.path.join(run_dir(run_id), "metrics.jsonl")
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


def stage_metrics(record: dict) -> dict:
    """§113 metrics of one stage, seen and unseen, nothing imputed."""
    if not record:
        return {}
    seen, unseen = record.get("seen", {}), record.get("unseen", {})
    return {
        "samples": record.get("samples"),
        "tokens": record.get("tokens"),
        "checkpoint": record.get("checkpoint"),
        "model_version": record.get("model_version"),
        "elapsed_s": record.get("elapsed_s"),
        "unseen": {m: unseen.get(m) for m in METRICS},
        "unseen_chance": unseen.get("chance"),
        "unseen_n": unseen.get("n"),
        "unseen_abstain_rate": unseen.get("abstain_rate"),
        "seen": {m: seen.get(m) for m in METRICS},
        "seen_n": seen.get("n"),
        "primary_metric": "unseen-label accuracy (#T-unseen-labels)",
    }


def backbone_report(backbone: str, run_id: str, report: dict | None = None,
                    stages: dict | None = None) -> dict:
    """One row of the #T-mix-1m top-2 table."""
    stages = stages if stages is not None else STAGES
    reached = read_stages(run_id)
    out = {
        "id": backbone,
        "run_id": run_id,
        "trainer": "training/python/train_decision.py (#T-train-real)",
        "corpus": "decision-mix-clean-1m",
        "lineage": lineage(report, backbone) if report else {},
        "stages": {},
        "run_json": (os.path.relpath(os.path.join(run_dir(run_id), "run.json"),
                                     ROOT)
                     if os.path.exists(os.path.join(run_dir(run_id),
                                                    "run.json")) else None),
    }
    for name, samples in stages.items():
        record = reached.get(samples)
        out["stages"][name] = ({"status": "measured", "target_samples": samples,
                                **stage_metrics(record)} if record else
                               {"status": "pending", "target_samples": samples,
                                "why": ("the run has not reached this point "
                                        "of the curve yet")})
    out["complete"] = all(s.get("status") == "measured"
                          for s in out["stages"].values())
    return out


def synthetic_verdict(baseline_run: str, plus_run: str,
                      samples: int, metric: str = "nll") -> dict:
    """§128: does grounded synthetic data improve held-out quality?

    Both arms are real training runs at the SAME budget over the same
    clean corpus; the only difference is whether `synth-v1` is in the
    mixture. The comparison is made on the unseen-label cut, which is the
    project's primary metric, and on held-out NLL, which moves before
    accuracy does.
    """
    base = read_stages(baseline_run).get(samples)
    plus = read_stages(plus_run).get(samples)
    if not base or not plus:
        missing = [n for n, r in ((baseline_run, base), (plus_run, plus))
                   if not r]
        return {"status": "pending", "runs": {"baseline": baseline_run,
                                              "plus_synthetic": plus_run},
                "compare_at_samples": samples,
                "why": f"no stage record at {samples:,} samples for {missing}",
                "verdict": None}
    b_unseen, p_unseen = base.get("unseen", {}), plus.get("unseen", {})
    # NLL is a loss: lower is better, so the gain is base - plus.
    gain_nll = round(b_unseen.get("nll", 0.0) - p_unseen.get("nll", 0.0), 6)
    gain_acc = round(p_unseen.get("accuracy", 0.0)
                     - b_unseen.get("accuracy", 0.0), 6)
    improved = gain_nll > 0 and gain_acc >= 0
    return {
        "status": "measured",
        "compare_at_samples": samples,
        "metric": metric,
        "runs": {"baseline": baseline_run, "plus_synthetic": plus_run},
        "baseline_unseen": {m: b_unseen.get(m) for m in METRICS},
        "plus_synthetic_unseen": {m: p_unseen.get(m) for m in METRICS},
        "gain": {"unseen_nll": gain_nll, "unseen_accuracy": gain_acc},
        "verdict": "GO-scale" if improved else "NO-GO-fix-prompts",
        "fallback": None if improved else (
            "do not scale the generator (§128). Correct the prompts first: "
            "the grounded-synthetic source is K<=5 over a 7-label pool with "
            "one teacher pair, so it adds volume without adding decision "
            "diversity. Re-measure with #T-gen-schemas' 1 152 schemas behind "
            "the generator before any scale-up."),
        "rule": ("held-out unseen-label NLL must fall and accuracy must not "
                 "fall; anything else is NO-GO"),
    }
