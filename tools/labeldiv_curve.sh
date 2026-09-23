#!/bin/bash
# #T-labelspace-div curve (62k/250k/1M unseen on the many-spaces mix).
#
# One variable against #T-lever-stack: the MIXTURE. Same backbone (ettin-68m),
# same head (d512), same objective (prior-penalty 1.0), same seed, same stages.
# The control is artifacts/runs/leverstack-d512-prior-ettin-68m-s20260922.
#
# Sentinel: waits for the control run's own summary.json, so the job can be
# registered while the GPU is still busy and starts by itself. (The previous
# sentinel waited on `genobj-prior-ettin-68m-s20260922`, a run-id that never
# existed — the real one is `genobj-prior-1m-...` — and polled forever.)
set -u
cd /Users/ricartjuncadella/Documents/Prj/asimovia/jev-clone
CONTROL=artifacts/runs/leverstack-d512-prior-ettin-68m-s20260922/summary.json
while [ ! -f "$CONTROL" ]; do sleep 120; done
sleep 60  # let the control's own eval.unseen gate release the device
PY=.venv-train/bin/python
RUN=labeldiv-d512-prior-ettin-s20260922
$PY -m training.python.train_decision train --max-samples 1000000 --batch-size 64 \
  --seed 20260922 --mix-seed 20260922 --device mps --run-id "$RUN" \
  --backbone ettin-68m --fence-clean --only-dataset episodic-div \
  --dataset-cap 1.0 --family-cap 1.0 --d-model 512 --prior-penalty 1.0 --no-gate || exit 1
$PY - <<'PYEOF' || exit 1
import json, os
from eval.unseen import run as unseen_run
run_id = "labeldiv-d512-prior-ettin-s20260922"
curve = {"task": "T-labelspace-div", "mix": "decision-mix-labeldiv-1m",
         "control_run": "leverstack-d512-prior-ettin-68m-s20260922",
         "arm": "ettin-68m d512 prior-penalty 1.0, only-dataset episodic-div",
         "cuts": {}}
for stage in (62500, 250000, 1000000):
    ckpt = f"artifacts/checkpoints/decision/{run_id}/stage-{stage:09d}"
    g = unseen_run(ckpt, "mps", write=False)
    curve["cuts"][f"{run_id}@{stage}"] = {
        "pass": g["pass"], "failed_criteria": g["failed_criteria"],
        "verdict": g["verdict"], "headline": g.get("headline")}
os.makedirs("artifacts/gates/T-labelspace-div", exist_ok=True)
with open("artifacts/gates/T-labelspace-div/curve.json", "w") as fh:
    json.dump(curve, fh, indent=2, sort_keys=True)
    fh.write("\n")
print("curve written")
PYEOF
