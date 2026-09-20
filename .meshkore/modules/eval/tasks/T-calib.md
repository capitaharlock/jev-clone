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

#T-calib con gate PASS (linaje 3d8343ca9a78, cadena de 13 gates
revalidada): NLL/Brier/ECE con bins vacíos conservados, temperature
scaling global → por locale → por cardinalidad con fallback registrado,
calibrador en `artifacts/gates/T-calib/calibration.json` ligado al
predictor y al hash del split; fit rechaza splits no-calibration.
Veredicto GO (test ECE 0.021 ≤ 0.05, accuracy intacta, gap EN-ES 0.037);
risk-coverage en test vacua (predictor perfecto en el piloto sintético)
y demostrada en OOD MASSIVE (0.19 vs 0.50 azar) con CIs bootstrap.
16 tests verdes.
