---
id: T-curriculum
title: Training curriculum and losses
status: done
priority: high
owner: unassigned
category: model
initiative: decision-model
depends_on:
  - T-hardneg
created: 2026-09-19
updated: 2026-09-20
completed_at: 2026-09-20T00:21:58Z
resolved_by: A004
resolved_by_conv: general-09192230
commit_shas: ['7f48fd53c7810cb2a66e8402a983ffe97cf4bf13']
---

# Training curriculum and losses

Stages 0–3 (head warmup → multi-dataset supervised → semantic mix →
hard negatives), sampling
entre datasets, curriculum de cardinalidad de opciones y de
multi-question. Cola reproducible `experiments/*.yaml` → `results/<run_id>`
con successive halving para backbone, LR, fusion depth y pesos de loss.
Total loss inicial: CE + Brier + KL + permutation
consistency (+ ordinal para `score`). Sin RL complejo en fase 1.

Distillation, calibration y compression son gates posteriores separados
(`#T-distillation`, `#T-calib`, `#T-quant-onnx`) para que cada transición tenga
una evidencia independiente y no se declare completada dentro de esta task.

Fuente: plan §§23–24, 50–53, 127.

> scope V1: loss inicial CE + Brier + KL + permutation consistency;
> término ordinal de `score` diferido a V2 con el tipo `score`.

## Verification gate

- Requiere PASS de `T-hardneg`; cada stage tiene contrato de entrada, métrica
  de promoción y rollback al último checkpoint válido.
- Tests deterministas cubren sampling, acumulación de loss, máscaras, resume y
  recuperación tras interrupción; mismo seed+manifest reproduce batches y
  métricas dentro de tolerancia.
- Ablations CE/CE+Brier y permutation on/off corren en un presupuesto smoke;
  no se promociona una loss que empeore el fixed regression benchmark.
- `T-curriculum` pasa con stages 0–3 reproducibles y una cola single-machine;
  no exige ni activa dispositivos extra.

## Done when

- Cada stage tiene entrada/salida y métricas de promoción definidas.
- Ablations obligatorias (Brier, permutation loss, distillation…)
  ejecutadas al menos una vez.
- ECE in-domain < 0.05 como objetivo V1 registrado por stage.
- Cada run conserva config, seed, revisiones de datos/código, entorno,
  checkpoint y métricas; reanudar no altera la muestra.

## Resolution

#T-curriculum con gate PASS (linaje 5bba14f9b1c4): stages 0-3 reproducibles con loss decreciente y ECE por stage, mismo seed+manifest reproduce batches y métricas, resume idéntico al ininterrumpido, ablación CE+Brier gana con medida (NLL 0.0233 vs 0.0262) y se promociona, cola experiments/*.yaml → results/; 12 tests verdes.
