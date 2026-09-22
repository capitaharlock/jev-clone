#!/bin/bash
# #T-labelspace-div curve (62k/250k/1M unseen on the many-spaces mix).
# Queued BEHIND the gen-objective sweep: waits for its last summary.json,
# then trains 2x 1M episodic runs (ettin-68m + modernbert-base, one run each:
# DEFAULT_STAGES already emits 62.5k/250k/1M checkpoints) and scores those
# stages with eval.unseen run(write=False) into T-labelspace-div/curve.json.
# Never touches MPS before the sweep is done; never rewrites T-unseen-labels.
set -u
cd /Users/ricartjuncadella/Documents/Prj/asimovia/jev-clone
while [ ! -f artifacts/runs/genobj-short-prior-s20260922/summary.json ]; do sleep 600; done
PY=.venv-train/bin/python
$PY -m training.python.train_decision train --max-samples 1000000 --batch-size 64 --seed 20260922 --device mps --run-id labeldiv-1m-ettin-s20260922 --fence-clean --only-dataset episodic-div --dataset-cap 1.0 --family-cap 1.0 --no-gate || exit 1
$PY -m training.python.train_decision train --max-samples 1000000 --batch-size 64 --seed 20260922 --device mps --run-id labeldiv-1m-modernbert-s20260922 --backbone modernbert-base --fence-clean --only-dataset episodic-div --dataset-cap 1.0 --family-cap 1.0 --no-gate || exit 1
$PY - <<'PYEOF' || exit 1
import json
from eval.unseen import run as unseen_run
curve = {"task": "T-labelspace-div", "mix": "decision-mix-labeldiv-1m", "cuts": {}}
for run_id in ("labeldiv-1m-ettin-s20260922", "labeldiv-1m-modernbert-s20260922"):
    for stage in (62500, 250000, 1000000):
        ckpt = f"artifacts/checkpoints/decision/{run_id}/stage-{stage:09d}"
        g = unseen_run(ckpt, "mps", write=False)
        curve["cuts"][f"{run_id}@{stage}"] = {
            "pass": g["pass"], "failed_criteria": g["failed_criteria"],
            "verdict": g["verdict"], "headline": g.get("headline")}
with open("artifacts/gates/T-labelspace-div/curve.json", "w") as fh:
    json.dump(curve, fh, indent=2, sort_keys=True)
    fh.write("\n")
print("curve written")
PYEOF
