---
id: T-data-eval
title: OOD, calibration split y reportes públicos
status: active
priority: medium
owner: unassigned
category: data
initiative: data-training
depends_on:
  - T-mix-5m
created: 2026-09-20
updated: 2026-09-20
---

# OOD, calibration split y reportes públicos

Cierra el work-stream con lo que lo hace creíble (§§76–83, 138–140):
set OOD 100 k train + 10 k held-out sin overlap (correcta removida,
ambiguity con targets 0.5/0.5, §§54–55, 138); calibration split
sellado 20–100 k labels reales multi-dominio/multi-K/multi-idioma —
NUNCA gradient-train sobre él (§§76, 139); clean room Jevals con IDs
congelados y leak detector triple (hash exacto, normalizado,
semántico, §77); `JevClone-General-Eval` 10–50 k held-out propio,
jamás entrenado (§78); benchmarks de dynamic labels (labels B
no vistas, §79), option shuffle (flip rate, JS divergence, §80),
candidate insertion (§81), OOD/abstención (§82) y multi-question
1→50 Q con latencia/memoria (§83). Publica `JEVALS_REPORT.md`,
`GENERALIZATION_REPORT.md`, `CALIBRATION_REPORT.md`,
`LATENCY_REPORT.md` (§140). Métricas: accuracy, NLL, Brier, ECE,
risk/coverage, high-confidence errors, OOD, estabilidad (§113);
primer objetivo: batir baselines encoder + POC + zero-shot dinámico
+ calibración antes de mirar a Jev (§115).

Fuentes: data-training §§54–55, 76–83, 113–115, 138–140.

## Verification gate

- Requiere PASS de `T-mix-5m`; el harness falla si el calibration
  split aparece en train o si un ID Jevals está en train (§§77, 139).
- Tests: permutation stability y candidate-insertion en verde con
  umbrales escritos; curva risk-coverage mejor que azar con abstención
  real (§82); multi-Q muestra ganancia shared-state en latencia;
  dynamic-labels B supera memorización de IDs (§79); los 4 reportes
  existen y son reproducibles por seed+manifest.
- El gate escribe `artifacts/gates/T-data-eval/gate.json` con
  `pass: true` y referencias a los 4 reportes; es el gate de cierre
  de `#data-training`.

## Done when

- OOD + calibration split sellados; clean room Jevals congelado.
- 4 reportes publicados; objetivo §115 (baselines + zero-shot +
  calibración) cumplido antes de comparar con Jev.
