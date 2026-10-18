#!/usr/bin/env bash
# El brazo muestreado de punta a punta — #T-fullspace-objective.
#
# Un job, cuatro pasos, para que el veredicto caiga sin que nadie decida
# nada después de ver una cifra:
#
#   1. entrenar el brazo a 62 500 filas (R5): mismo corpus, misma seed y
#      mismo objetivo que el control `bigk-fullspace` (la vía exacta); la
#      ÚNICA variable declarada es que el denominador sale del espacio de
#      la fila — negativos in-batch entre espacios con `log pi`, lo que
#      exige batches mixtos y así queda dicho.
#   2. `eval.unseen` a K<=8 — el diagnóstico de fase 1, que hace este brazo
#      comparable con los anteriores. Sale con 1 en NO-GO, que es un
#      veredicto y no un fallo: de ahí el `|| true`.
#   3. `tools.fullspace_arm report` — la PRIMARIA: cardinalidad completa
#      sobre el corte de desarrollo congelado, con abstención en los dos
#      regímenes y la regla pre-registrada aplicada.
#   4. el corte RESERVADO, SÓLO si el paso 3 dijo GO (R7). Un NO-GO
#      termina aquí a propósito.
set -u
cd "$(dirname "$0")/.."
PY=.venv-train/bin/python
export PYTHONPATH=.
RUN=fullspace-sampled-d512-prior-ettin-68m-s20260922
ARM=artifacts/checkpoints/decision/$RUN/stage-000062500
CTL=artifacts/checkpoints/decision/bigk-fullspace-d512-prior-ettin-68m-s20260922/stage-000062500
GATE=artifacts/gates/T-fullspace-objective/gate.json

set -e
$PY -m training.python.train_decision train \
    --max-samples 62500 --batch-size 64 \
    --seed 20260922 --mix-seed 20260922 --device mps \
    --run-id "$RUN" --backbone ettin-68m --fence-clean --no-gate \
    --prior-penalty 1.0 --d-model 512 --full-space --in-batch-negatives
set +e
$PY -m eval.unseen gate --checkpoint "$ARM" --device mps
set -e
$PY -m tools.fullspace_arm report --arm "$ARM" --control "$CTL" --device mps

if $PY -c "import json,sys; sys.exit(0 if json.load(open('$GATE'))['verdict']['go'] else 1)"; then
    echo "[fullspace] la primaria de desarrollo bate el azar y al control — se lee el corte reservado UNA vez (R7)"
    $PY -m eval.fullspace gate --checkpoint "$ARM" --device mps \
        --reason "#T-fullspace-objective: el brazo muestreado pasó su regla pre-registrada de desarrollo; una lectura del corte reservado para publicar la cifra a 77 vías"
else
    echo "[fullspace] NO-GO en el corte de desarrollo — el corte reservado NO se lee (R7). Veredicto en $GATE"
fi
