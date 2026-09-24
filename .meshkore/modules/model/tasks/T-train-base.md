---
id: T-train-base
title: Baseline training runs on converted sets
status: done
priority: high
owner: unassigned
category: model
initiative: decision-model
depends_on:
  - T-prefetch
created: 2026-09-20
updated: 2026-09-20
resolved_by: A009
resolved_by_conv: work-decision-model-T-train-base-1789904581
---
# Baseline training runs on converted sets

Primer entrenamiento real sobre los sets ya convertidos a schema
universal (`artifacts/data-prefetch/`): TF-IDF + regresión logística por
familia (boolean: boolq, civil-comments; choice: helpsteer2), train en
`split==train`, eval en calibration/test. Corre en `.venv-train`
(sklearn), logs en `artifacts/runs-baseline-v1.log`, métricas en
`artifacts/runs/<ts>/metrics.json`.

v2 absorbe `artifacts/data-qwen/*.aug.jsonl` cuando el conversor Qwen
tenga masa suficiente; v3 sube a backbones reales (bakeoff top-2).

## Done when

- Métricas accuracy/logloss publicadas por set en `artifacts/runs/`.
- El run es re-ejecutable con un solo comando y seed fija.
- v2 incorpora datos Qwen-aug sin reentrenar desde cero a mano.

## Resolution


Failed to authenticate: OAuth session expired and could not be refreshed

## Resolution (2026-09-20, general-09192230)

Re-ran directly: `.venv-train/bin/python data/train_baseline.py` → run
`20260920T154511Z`, 8/8 jobs with accuracy/logloss in
`artifacts/runs/20260920T154511Z/metrics.json`. Trainer now also persists
`models/<task>.pkl` per job; testable via
`.venv-train/bin/python tools/predict_v0.py --run 20260920T154511Z
--task massive "<text>"` (smoke-tested: `alarm_set` 0.80 on an alarm query).
The prior `blocked` was an agent-auth failure, no code cause.
