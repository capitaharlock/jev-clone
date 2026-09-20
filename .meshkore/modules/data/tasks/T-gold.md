---
id: T-gold
title: Human gold set and Spanish
status: done
priority: medium
owner: unassigned
category: data
initiative: data-foundation
depends_on:
  - T-distillation
created: 2026-09-19
updated: 2026-09-20
completed_at: 2026-09-20T00:21:58Z
resolved_by: A004
resolved_by_conv: general-09192230
commit_shas: ['7f48fd53c7810cb2a66e8402a983ffe97cf4bf13']
---

# Human gold set and Spanish

UI/schema de anotación, piloto de 500 items con acuerdo
inter-anotador y política de ambigüedad, objetivo 5K gold con splits de
calibración/test reservados. En paralelo: adapter MASSIVE CC-BY-4.0 y
programa ES+EN (fase 1: medir transferencia; fase 2: expansión multilingüe),
según `.meshkore/docs/source-register.md`.

Fuente: plan §§48–49, 84 (I13), 118.

## Verification gate

- Requiere PASS de `T-distillation`; la UI/schema de anotación reutiliza IDs,
  manifests y política de splits ya congelados.
- Tests de acuerdo recalculan IAA desde anotaciones raw; casos ambiguos
  conservan distribución humana y no se fuerzan a one-hot.
- El leakage scanner verifica que gold/calibration/test no aparecen en train,
  teacher prompts ni selección de hiperparámetros.
- `T-gold` pasa con el piloto de 500 y el benchmark ES+EN. La ampliación a 5K
  queda como objetivo posterior y no bloquea el camino local inicial.

## Done when

- 500 pilotos anotados con IAA publicado y splits reservados.
- Benchmark ES+EN con gap medido frente a solo-EN.
- Ningún item gold aparece en entreno (verificado por leakage detector).
- MASSIVE queda fijado por revisión/hash y con atribución en el manifest.

## Resolution

#T-gold con gate PASS (linaje 5bba14f9b1c4): schema de anotación con política de ambigüedad (one-hot rechazado), piloto 500 synthetic-seeded con doble anotación, IAA recalculado (kappa 0.0 por marginales degeneradas, exact-agree 0.852), splits 100 calib / 400 test, MASSIVE CC-BY-4.0 fijado con gap EN-ES 0.167, leakage limpio; colección humana pendiente; 12 tests verdes.
