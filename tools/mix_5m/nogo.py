"""The #T-mix-5m scaling decision, and the rule that keeps it honest.

§136 makes 5 M conditional on 1 M having demonstrated a gain. The measured
answer is NO: across 250 k -> 1 M, on the same corpus, the same trainer and
the same fixed evals, every reasoning cut stayed inside its own chance
interval and the unseen-label metric fell. So this module publishes a
`pass: false` / `NO-GO` gate instead of a mixture.

Two halves, and the split matters:

* `build_gate()` READS the evidence off disk — `artifacts/gates/T-mix-1m/
  gate.json`, the per-checkpoint #T-unseen-labels artifacts and each run's
  `summary.json`. No number in the gate is typed by hand.
* `violations()` is the rule, and it runs without any of that: the gate
  carries its own evidence, so CI can assert the published decision is
  consistent with the published numbers even though `artifacts/runs/` and
  `artifacts/mix-1m/curve.jsonl` are gitignored.

The rule is one sentence: **a gate may not claim GO while the reasoning
evals are indistinguishable from chance.** It is the C1 idea of
`eval/gate_rules.py` ("agreement indistinguishable from chance cannot sit
next to a PASS") applied to the one decision this task exists to make.

"Indistinguishable from chance" is `eval/unseen.py`'s test, not a looser
one: the CI95 LOWER bound of the options-only accuracy must sit strictly
above the cut's own options-only chance rate. Options-only because an
abstaining head scores near zero on raw accuracy for reasons that have
nothing to do with reasoning.

CLI (stdlib only, any python3, run from the repo root):

    python3 -m tools.mix_5m.nogo write     # rebuild the gate from artifacts
    python3 -m tools.mix_5m.nogo check     # audit the published gate, exit 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GATE_DIR = os.path.join(ROOT, "artifacts", "gates", "T-mix-5m")
GATE = os.path.join(GATE_DIR, "gate.json")
UPSTREAM_GATE = os.path.join(ROOT, "artifacts", "gates", "T-mix-1m", "gate.json")
BY_CHECKPOINT = os.path.join(
    ROOT, "artifacts", "gates", "T-unseen-labels", "by-checkpoint")
BAKEOFF = os.path.join(ROOT, "artifacts", "gates", "T-bakeoff", "report.json")
CURVE = os.path.join(ROOT, "artifacts", "mix-1m", "curve.jsonl")

#: Held out by #T-halt-contam — eval-only, never a training row. These are
#: the cuts that answer "does it reason", as opposed to "does it match text".
REASONING_EVALS = ("logiqa-mc", "logiqa-nli", "reclor")

#: In-corpus datasets whose accuracy is the reasoning signal inside the
#: mixture itself. Read from the 1 M re-score, so they are TRAINING scores:
#: a training score at chance is a stronger finding than a held-out one.
IN_CORPUS_REASONING = ("swag", "synth-v1", "prog-gold")

#: The stages §111 asks the curve to be read across.
STAGES = ("stage0", "stage1")

#: Verdicts that authorise spending. `NO-GO` is deliberately NOT a prefix
#: match of `GO` — see `is_green()`.
GREEN_VERDICTS = frozenset({"GO", "GO-SCALE", "PASS", "OK", "GREEN"})


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ------------------------------------------------------------------- the rule

def beats_chance(metrics: dict) -> bool:
    """`eval/unseen.py`'s ranking test: CI95 low > the cut's own chance."""
    lo = metrics["accuracy_options_only_ci95"][0]
    return bool(lo > metrics["chance_options_only"])


def is_green(gate: dict) -> bool:
    """Does this artifact authorise the spend?

    A top-level `pass` is authoritative (same precedence as
    `eval.gate_rules.green_claim`); only without one does the verdict decide.
    """
    if "pass" in gate:
        return gate["pass"] is True
    verdict = str(gate.get("verdict", "")).strip().upper()
    return verdict in GREEN_VERDICTS


def at_chance(gate: dict) -> list[dict]:
    """Reasoning cuts the gate publishes that do NOT clear their chance rate."""
    out = []
    for ckpt, cuts in sorted(gate["evidence"]["reasoning_evals"].items()):
        for name, m in sorted(cuts.items()):
            if not beats_chance(m):
                out.append({"checkpoint": ckpt, "eval": name,
                            "accuracy_options_only": m["accuracy_options_only"],
                            "ci95": m["accuracy_options_only_ci95"],
                            "chance_options_only": m["chance_options_only"]})
    return out


def violations(gate: dict) -> list[dict]:
    """Every way this gate could contradict the numbers it publishes.

    Returns a list of findings; empty means the decision and the evidence
    agree. These are errors, never warnings.
    """
    found = []
    green = is_green(gate)
    evidence = gate["evidence"]
    stale = at_chance(gate)
    clears = [c for c in _reasoning_rows(gate) if beats_chance(c["metrics"])]

    if green and not clears:
        found.append({
            "rule": "S1",
            "why": "the gate claims GO while no reasoning eval clears its "
                   "own chance rate — more rows cannot be the fix for a "
                   "curve that never left chance",
            "at_chance": stale})

    deltas = evidence["scaling_250k_to_1m"]
    if green and not any(d["delta"] > 0 for d in deltas.values()):
        found.append({
            "rule": "S2",
            "why": "the gate claims GO while no 250 k -> 1 M delta on the "
                   "primary metric is positive",
            "deltas": deltas})

    for run, d in sorted(deltas.items()):
        recomputed = round(d["at_1m"] - d["at_250k"], 6)
        if abs(recomputed - d["delta"]) > 1e-6:
            found.append({
                "rule": "S3", "run": run,
                "why": "the published delta is not the difference of the "
                       "published endpoints",
                "recorded": d["delta"], "recomputed": recomputed})

    authorised = gate["decision"]["authorises"]
    if not green and any(authorised.values()):
        found.append({
            "rule": "S4",
            "why": "a NO-GO gate may not authorise work",
            "authorises": authorised})
    if green and not all(authorised.values()):
        found.append({
            "rule": "S5",
            "why": "a GO gate that authorises nothing is not a GO",
            "authorises": authorised})
    return found


def _reasoning_rows(gate: dict) -> list[dict]:
    return [{"checkpoint": ckpt, "eval": name, "metrics": m}
            for ckpt, cuts in sorted(
                gate["evidence"]["reasoning_evals"].items())
            for name, m in sorted(cuts.items())]


def curve_disagreements(gate: dict, curve_path: str = CURVE) -> list[dict]:
    """Does `curve.jsonl` still say what the gate quotes it as saying?

    The job log is the raw record: one `run-end` per run, carrying the tail
    of the trainer's own summary. Every in-corpus accuracy the gate
    publishes has to be findable verbatim in the tail of the run it is
    attributed to, and every cited run has to have exited 0.
    """
    ends = {}
    with open(curve_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("t") == "run-end":
                ends[rec["run_id"]] = rec

    found = []
    for run, datasets in sorted(gate["evidence"]["in_corpus_rescore"].items()):
        rec = ends.get(run)
        if rec is None:
            found.append({"run": run, "why": "no run-end in curve.jsonl"})
            continue
        if rec.get("rc") != 0:
            found.append({"run": run, "why": "run did not exit 0",
                          "rc": rec.get("rc")})
        tail = rec.get("tail", "")
        for name, m in sorted(datasets.items()):
            if name not in REASONING_EVALS + IN_CORPUS_REASONING:
                continue
            if str(m["accuracy"]) not in tail:
                found.append({"run": run, "dataset": name,
                              "why": "accuracy not in the run's own tail",
                              "gate": m["accuracy"]})
    return found


# -------------------------------------------------------------- the evidence

def _eval_only(doc: dict) -> dict:
    out = {}
    for name in REASONING_EVALS:
        raw = doc.get("eval_only_results", {}).get(name, {}).get("raw")
        if raw is None:
            continue
        out[name] = {k: raw[k] for k in (
            "n", "accuracy", "accuracy_ci95", "chance",
            "accuracy_options_only", "accuracy_options_only_ci95",
            "chance_options_only", "ece", "abstain_rate", "mean_k")}
    return out


def build_gate(root: str = ROOT) -> dict:
    """Assemble the gate from what the 1 M job left on disk."""
    upstream = load(UPSTREAM_GATE)
    backbones = upstream["training"]["backbones"]

    reasoning, unseen_headline = {}, {}
    for name in sorted(os.listdir(BY_CHECKPOINT)):
        if not name.endswith(".json"):
            continue
        doc = load(os.path.join(BY_CHECKPOINT, name))
        key = name[:-len(".json")]
        reasoning[key] = _eval_only(doc)
        unseen_headline[key] = {
            "model_version": doc["model_version"],
            "pass": doc["pass"],
            "failed_criteria": doc["failed_criteria"],
            "accuracy_seen": doc["headline"]["overall"]["accuracy_seen"],
            "accuracy_unseen": doc["headline"]["overall"]["accuracy_unseen"],
            "accuracy_drop": doc["headline"]["overall"]["accuracy_drop"],
            "chance_unseen":
                doc["headline"]["overall"]["comparability"]["chance_unseen"]}

    #: #T-bakeoff-real measured the trainable head against the frozen
    #: backbone on this exact architecture. Quoted, not recounted.
    head_params = {r["id"]: r["cost"]["head_params"]
                   for r in load(BAKEOFF)["pareto"]["rows"]
                   if "head_params" in r.get("cost", {})}

    scaling, rescore, curves, cost = {}, {}, {}, {}
    for bid, b in sorted(backbones.items()):
        run = b["run_id"]
        s0, s1 = b["stages"]["stage0"], b["stages"]["stage1"]
        scaling[run] = {
            "backbone": bid,
            "metric": s1["primary_metric"],
            "chance": s1["unseen_chance"],
            "at_250k": s0["unseen"]["accuracy"],
            "at_1m": s1["unseen"]["accuracy"],
            "delta": round(s1["unseen"]["accuracy"] - s0["unseen"]["accuracy"], 6),
            "seen_at_250k": s0["seen"]["accuracy"],
            "seen_at_1m": s1["seen"]["accuracy"],
            "ece_at_250k": s0["unseen"]["ece"],
            "ece_at_1m": s1["unseen"]["ece"],
            "abstain_at_250k": s0["unseen_abstain_rate"],
            "abstain_at_1m": s1["unseen_abstain_rate"],
            "n": s1["unseen_n"]}
        summary_path = os.path.join(root, "artifacts", "runs", run, "summary.json")
        run_path = os.path.join(root, "artifacts", "runs", run, "run.json")
        if os.path.exists(summary_path):
            summary = load(summary_path)
            rescore[run] = {
                name: {"accuracy": m["accuracy"],
                       "accuracy_options_only": m["accuracy_options_only"],
                       "n": m["n"]}
                for name, m in sorted(summary["train"]["per_dataset"].items())}
            curves[run] = summary["scale_curve"]
        if os.path.exists(run_path):
            manifest = load(run_path)
            frozen = manifest["backbone"]["params"]
            head = head_params.get(manifest["backbone"]["id"])
            cost[run] = {
                "backbone": manifest["backbone"]["id"],
                "backbone_params": frozen,
                "backbone_frozen": manifest["backbone"]["frozen"],
                "head_params": head,
                "head_params_source":
                    "artifacts/gates/T-bakeoff/report.json (same head "
                    "architecture, measured)",
                "trainable_fraction":
                    None if head is None else round(head / (frozen + head), 6),
                "architecture": manifest["architecture"],
                "lr": manifest["lr"],
                "max_length": manifest["max_length"]}

    gate = {
        "format": 1,
        "task": "T-mix-5m",
        "pass": False,
        "verdict": "NO-GO",
        "generated_utc": utcnow(),
        "decision": {
            "authorises": {"mix_5m": False,
                           "distillation_pilot_100k": False,
                           "curve_to_10m": False},
            "criterion": "§§89, 136 — T-mix-5m escala a 5 M SÓLO si 1 M "
                         "demuestra generalización; si 1 M no mejora, esta "
                         "task registra NO-GO y no genera 5 M",
            "why": "1 M did not demonstrate generalisation. Across 250 k -> "
                   "1 M on the same corpus, the same trainer and the same "
                   "fixed evals, unseen-label accuracy FELL on both "
                   "backbones and no reasoning eval cleared its own chance "
                   "rate at either stage. A curve that is flat at chance "
                   "cannot be lifted by multiplying the rows on it: 5 M "
                   "would buy ~5x the compute for the same measurement.",
            "not_a_claim": "this is a NO-GO on SCALING, not on the corpus. "
                           "T-mix-1m passed as a corpus build; what is "
                           "unmet is the 'ganancia demostrada' half of the "
                           "§136 precondition."},
        "upstream": {
            "T-mix-1m": {
                "gate": "artifacts/gates/T-mix-1m/gate.json",
                "pass": upstream["pass"],
                "corpus": upstream["corpus"],
                "rows": upstream["mix"]["rows"],
                "scope": "the passing checks are construction checks (rows, "
                         "fence, guardrails, diversity, both top-2 trained); "
                         "none of them is a generalisation gain"},
            "T-unseen-labels": {
                "artifacts": "artifacts/gates/T-unseen-labels/by-checkpoint/",
                "by_checkpoint": unseen_headline}},
        "evidence": {
            "reasoning_evals": reasoning,
            "reasoning_evals_note":
                "eval-only cuts (#T-halt-contam): logiqa/reclor are never "
                "training rows. Read accuracy_options_only against "
                "chance_options_only — the raw accuracy of an abstaining "
                "head says more about abstention than about reasoning.",
            "in_corpus_rescore": rescore,
            "in_corpus_rescore_note":
                "summary.json['train'] — the trained head re-scored over the "
                "1 000 000 rows it was TRAINED on. swag and synth-v1 sitting "
                "at chance here is the stronger form of the finding: the "
                "model cannot fit these families, let alone generalise them.",
            "scaling_250k_to_1m": scaling,
            "within_run_scale_curve": curves,
            "trainable_surface": cost},
        "hypothesis": {
            "name": "frozen backbone",
            "claim": "the curve is flat because the only thing the rows can "
                     "move is a small pointer head on top of an encoder that "
                     "never updates. The head learns to match option text to "
                     "state text — which is why the text-matching families "
                     "are near-ceiling and the reasoning families are at "
                     "chance — and no quantity of rows adds a capability the "
                     "frozen representation does not already carry.",
            "supported_by": [
                "run.json backbone.frozen is true for every run on the curve, "
                "and the trainable surface is under 4 % of the parameters "
                "(see evidence.trainable_surface)",
                "text-matching families are near-ceiling on the same "
                "checkpoint that is at chance on swag and synth-v1",
                "seen-label accuracy also fell 250 k -> 1 M while ECE rose: "
                "the extra rows moved calibration, not capability"],
            "not_yet_tested": "no run on this curve unfroze the backbone, so "
                              "the hypothesis is what the curve SUPPORTS, "
                              "not what it proves"},
        "flips_to_go_when": [
            "an unfrozen-backbone (or LoRA / partially-unfrozen) run at 250 k "
            "on this same corpus clears chance on at least one eval-only "
            "reasoning cut by CI95 lower bound, AND",
            "the 250 k -> 1 M delta on unseen-label accuracy is positive for "
            "at least one backbone under that configuration, AND",
            "the gain survives the #T-unseen-labels protocol unchanged — "
            "same holdout, same calibration fitted on seen labels only"],
        "rule": {
            "id": "S1",
            "statement": "a gate may not claim GO while the reasoning evals "
                         "are indistinguishable from chance",
            "test": "accuracy_options_only_ci95[0] > chance_options_only, the "
                    "same test as eval/unseen.py's unseen_ranking_beats_chance",
            "enforced_by": "tools.mix_5m.nogo.violations (data/test_mix_5m.py)"},
        "honesty": "published as measured. The task's own gate says a NO-GO "
                   "IS the deliverable when 1 M does not improve (§§89, 136); "
                   "recording it is the work, not a failure to do the work.",
        "report": "artifacts/gates/T-mix-5m/SCALING_NOGO.md",
    }
    return gate


def write_gate(path: str = GATE) -> dict:
    gate = build_gate()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(gate, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return gate


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(prog="tools.mix_5m.nogo")
    ap.add_argument("action", choices=("write", "check"))
    args = ap.parse_args(argv)

    if args.action == "write":
        gate = write_gate()
        print(f"{GATE}: pass={gate['pass']} verdict={gate['verdict']}")
        return 0

    gate = load(GATE)
    bad = violations(gate)
    if os.path.exists(CURVE):
        bad += curve_disagreements(gate)
    else:
        print(f"note: {CURVE} absent (gitignored) — curve cross-check skipped")
    for finding in bad:
        print(json.dumps(finding, sort_keys=True), file=sys.stderr)
    print(f"{GATE}: {len(bad)} violation(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
