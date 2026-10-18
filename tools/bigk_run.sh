#!/usr/bin/env bash
# The full-space arm end to end — #T-bigk-optsets.
#
# One job, four steps, so the verdict falls without anyone deciding
# anything after seeing a number:
#
#   1. train the arm: same corpus, same seed, same objective as the
#      control `leverstack-d512-prior`; the ONLY variable is that every
#      row offers its whole label space (`--full-space`).
#   2. `eval.unseen` at K<=8 — the phase-1 diagnostic, so this arm is
#      comparable with the ones before it. It exits 1 on NO-GO, which is
#      a verdict and not a failure, hence the `|| true`.
#   3. `tools.bigk_arm report` — the PRIMARY: full cardinality on the
#      frozen DEVELOPMENT cut, plus the abstention curve, the supervision
#      block and the pre-registered decision rule.
#   4. the RESERVED cut, ONLY if step 3 said GO (rule R7: arms are chosen
#      on dev; the reserved cut is read once, with a reason, and the read
#      is logged). A NO-GO ends here on purpose.
set -u
cd "$(dirname "$0")/.."
PY=.venv-train/bin/python
RUN=bigk-fullspace-d512-prior-ettin-68m-s20260922
ARM=artifacts/checkpoints/decision/$RUN/stage-000250000
CTL=artifacts/checkpoints/decision/leverstack-d512-prior-ettin-68m-s20260922/stage-000250000
GATE=artifacts/gates/T-bigk-optsets/gate.json

set -e
$PY -m training.python.train_decision train \
    --max-samples 250000 --batch-size 64 \
    --seed 20260922 --mix-seed 20260922 --device mps \
    --run-id "$RUN" --backbone ettin-68m --fence-clean --no-gate \
    --prior-penalty 1.0 --d-model 512 --full-space
set +e
$PY -m eval.unseen gate --checkpoint "$ARM" --device mps
set -e
$PY -m tools.bigk_arm report --arm "$ARM" --control "$CTL" --device mps

if $PY -c "import json,sys; sys.exit(0 if json.load(open('$GATE'))['verdict']['go'] else 1)"; then
    echo "[bigk] development primary beats chance and the control — reading the reserved cut ONCE (R7)"
    $PY -m eval.fullspace gate --checkpoint "$ARM" --device mps \
        --reason "#T-bigk-optsets: the full-space arm passed its pre-registered development rule; one read of the reserved cut to publish the 77-way number beside the old 0.0123"
else
    echo "[bigk] NO-GO on the development cut — the reserved cut is NOT read (R7). Verdict in $GATE"
fi
