---
id: T-calib
title: Calibration and abstention
status: done
priority: high
owner: unassigned
category: eval
initiative: eval-trust
depends_on:
  - T-gold
created: 2026-09-19
updated: 2026-09-20
completed_at: 2026-09-20T00:40:09.639Z
resolved_by: A004
resolved_by_conv: general-09192230
commit_shas: ['8865912dc172347c00edc17bc2471ad5801aeafb']
---
# Calibration and abstention

Toolkit NLL/Brier/ECE, temperature scaling (global → por tipo →
por cardinalidad), reliability plots, drift de calibración por dominio
y calibrador serializado en `calibration.json`. Comparar estrategias de
abstención (energía, trust head, unknown explícito) con curvas
risk-coverage; conformal/selective-risk en V2.

Fuente: plan §§16–17, 25–26.

## Verification gate

- Requiere PASS de `T-gold`; calibration, test y OOD se cargan por hashes
  distintos y el runner impide ajustar parámetros sobre test.
- Unit tests validan NLL/Brier/ECE, bins vacíos, serialización y aplicación del
  calibrador; reliability plots se regeneran desde métricas raw.
- Comparación pre/post calibración exige no degradar materialmente accuracy y
  publica curvas risk-coverage con intervalos por bootstrap.
- `T-calib` pasa con calibrador ligado al checkpoint y umbrales fijados; si no
  alcanza ECE objetivo, conserva NO-GO explícito sin falsificar el resultado.

## Done when

- Calibration report con ECE/Brier in-domain y OOD por release.
- Risk-coverage con abstención supera claramente al azar.
- Calibrador versionado junto al checkpoint que calibra.
- Umbrales se ajustan solo en calibration split; test y OOD permanecen
  intocados hasta el informe final.

## Resolution

Sí, la segunda iniciativa (#eval-trust) está terminada: 3/3 tareas en `done`. Verifiqué toda la suite antes (111 Python + 19 Rust en verde) e implementé #T-calib — calibración con temperature scaling, abstención con curvas risk-coverage y veredicto GO — con 13 gates PASS en un linaje (commit `8865912`). Suite final: 127 Python + 19 Rust, todo verde. Quedan `decision-model` (solo T-dist-train en backlog), `local-runtime` (3 tareas) y `cloud-release`.

— T-calib · toolkit de calibración/abstención con gate PASS y cierre de #eval-trust

**Commit** `8865912dc` · 21 files · 9.6M tokens

**Files changed (21):**
- `.meshkore/modules/eval/tasks/T-calib.md`
- `artifacts/gates/T-bakeoff/gate.json`
- `artifacts/gates/T-bakeoff/report.json`
- `artifacts/gates/T-calib/calibration.json`
- `artifacts/gates/T-calib/gate.json`
- `artifacts/gates/T-calib/report.json`
- `artifacts/gates/T-curriculum/gate.json`
- `artifacts/gates/T-data-p0/gate.json`
- `artifacts/gates/T-data-schema/gate.json`
- `artifacts/gates/T-distillation/gate.json`
- `artifacts/gates/T-firewall/gate.json`
- `artifacts/gates/T-gold/gate.json`
- `artifacts/gates/T-hardneg/gate.json`
- `artifacts/gates/T-option-mixer/gate.json`
- `artifacts/gates/T-option-mixer/report.json`
- `artifacts/gates/T-recon/gate.json`
- `artifacts/gates/T-rust-skel/gate.json`
- `artifacts/gates/T-shared-state/gate.json`
- `eval/calib.py`
- `eval/test_calib.py`
- `training/python/tools/gate.py`
