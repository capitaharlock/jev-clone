---
id: T-mix-10m
title: decision-mix v2/v3 (10M/20M, FLAN amplio, Dolma)
status: backlog
priority: low
owner: unassigned
category: data
initiative: train-scaleout
depends_on:
  - T-mix-5m
created: 2026-09-20
updated: 2026-09-20
---

# decision-mix v2/v3 (10M/20M, FLAN amplio, Dolma)

Rama futura en `backlog`: `decision-mix-v2` (10 M) y v3 (20 M, §§87–88)
SÓLO si 5 M → 10 M sigue mejorando y el modelo no satura; 50 M ni se
plantea si 20 M no mejora — más filas no arreglan arquitectura (§89).
Amplía con subsets FLAN filtrados (sólo closed/boolean/ordinal,
streaming, sin descargar cientos de GB, §21), stream Dolma como
segunda distribución de states (§34), Natural Instructions v2 para
diversidad de instrucciones (§22) y preference pairs con auditoría de
provenance (§§50–51). Puerta de activación: veredicto "continuar" del
gate `#T-mix-5m` + presión de la curva de scaling (§§73, 111).

Fuentes: data-training §§21–22, 34, 50–51, 73, 87–89, 111.

## Verification gate

- Requiere PASS de `T-mix-5m` con veredicto de continuar; ninguna task
  obligatoria depende de esta.
- Tests: mismos que `#T-mix-5m` sobre v2 (curva extendida, ablations,
  guardrails, benchmark-clean fence); si 10 M no mejora, NO-GO y se
  archiva sin tocar V1.
- El gate escribe `artifacts/gates/T-mix-10m/gate.json` con `pass: true`
  y la curva 5 M → 10 M (→ 20 M).

## Done when

- Criterio de activación (curva abierta) verificado antes de generar
  nada; v2/v3 sólo existen si los números los piden.
- Ninguna dependencia obligatoria apunta aquí; V1 queda completo sin
  esta task.
