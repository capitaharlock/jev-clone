---
id: T-hardneg
title: Hard-negative and unknown engine
status: done
priority: high
owner: unassigned
category: data
initiative: data-foundation
depends_on:
  - T-option-mixer
created: 2026-09-19
updated: 2026-09-20
completed_at: 2026-09-20T00:21:58Z
resolved_by: A004
resolved_by_conv: general-09192230
---

# Hard-negative and unknown engine

Índice de embeddings de labels para nearest-label sampler, hard
negatives in-domain y cross-dataset, distractores adversariales de
teacher con difficulty score. Y la otra mitad: generadores de
missing-answer, preguntas no relacionadas, states corruptos y
contradictorios para entrenar el `unknown` explícito y el trust head.

Fuente: plan §§16–17, 44.2, I8–I9 del plan.

## Verification gate

- Requiere PASS de `T-option-mixer`; genera ejemplos contra la arquitectura y
  schema ya fijados.
- Tests por seed cubren nearest-label, cross-dataset, missing-answer,
  unrelated, corrupto y contradictorio, con dificultad monotónica comprobada.
- El firewall vuelve a escanear cada artefacto generado y un canario eval-only
  debe ser rechazado antes de persistirlo.
- El gate `T-hardneg` mide recall/precision de `unknown`, errores de alta
  confianza y distribución de dificultad sobre un smoke entrenable.

## Done when

- Sampler de hard negatives con difficulty score reproducible.
- Set OOD (`unknown`/corrupto/contradictorio) versionado en registry.
- Missing Correct Answer Detection medido en baseline.

## Resolution

#T-hardneg con gate PASS (linaje 5bba14f9b1c4): sampler nearest-label in-domain + cross-dataset con difficulty reproducible, 24 OOD en 4 clases re-escaneados por el firewall, canario eval-only real rechazado antes de persistir, unknown recall 0.417 / precision 1.0 / 0 errores de alta confianza; 10 tests verdes.
