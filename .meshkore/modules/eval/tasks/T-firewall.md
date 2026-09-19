---
id: T-firewall
title: Eval firewall and stress suite
status: done
priority: high
owner: unassigned
category: eval
initiative: eval-trust
depends_on:
  - T-data-schema
created: 2026-09-19
updated: 2026-09-20
---

# Eval firewall and stress suite

Registry de benchmarks versionados e inmutables (MMLU-Pro, GPQA,
ARC, MuSR, RewardBench…) con scanner de contaminación en CI: entrenar
con ellos está prohibido. Stress suite propia: label rename, opaque-ID
trap, shuffle, hard siblings, state irrelevante, evidencia al final,
contradicciones, mismatch multilingüe, typos. Criterios GO/NO-GO del
§128 y release criteria del §170.

CLINC150/OOS queda como holdout de transferencia semántica y OOD, no como
train, hasta resolver su licencia. El scanner cubre hashes exactos y similitud
contra prompts/outputs de teachers para detectar paráfrasis contaminadas.

Fuente: plan §§42, 64–65, 120, 128, 147, 167–170.

## Verification gate

- Requiere PASS de `T-data-schema`.
- Fixtures canarios de exact match, texto normalizado y paráfrasis aproximada
  deben ser detectados; fixtures limpios no pueden generar falsos positivos por
  encima del umbral documentado.
- Cada benchmark queda fijado por revisión/hash y un test prueba que ni sus
  inputs ni outputs pueden entrar en train, teachers o hard negatives.
- La stress suite tiene semillas y umbrales versionados; el gate conserva raw
  results y PASS/NO-GO en `artifacts/gates/T-firewall/gate.json`.

## Done when

- Contamination scanner bloquea datasets que tocan benchmarks.
- Stress suite corre por release con umbrales GO/NO-GO.
- Ningún modelo se llama "calibrated" sin métricas held-out + OOD.
- Un dataset completo no visto demuestra semantic label transfer o dispara
  NO-GO para la hipótesis de decision foundation model.
