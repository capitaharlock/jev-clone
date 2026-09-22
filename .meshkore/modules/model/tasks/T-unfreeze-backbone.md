---
id: T-unfreeze-backbone
title: Descongelar el backbone — capacidad entrenable donde está el lenguaje
status: next
priority: high
owner: unassigned
category: model
initiative: generalization-fix
depends_on:
  - T-antiscale-diag
created: 2026-09-22
updated: 2026-09-22
---

# Descongelar el backbone — capacidad entrenable donde está el lenguaje

Todos los runs hasta hoy entrenan ~2,9 M de cabeza pointer sobre un
ModernBERT-base de 149 M **congelado** (`backbone.frozen: true` en
`artifacts/runs/*/run.json`). La comparación pregunta↔opción tiene que ocurrir
en las representaciones, no en dos capas de 256 dimensiones por encima de
ellas.

Trabajo: hacer el congelado un parámetro de entreno con tres regímenes
comparables sobre el **mismo** corpus `decision-mix-clean-1m` y los mismos
evals fijos:

- `frozen` (baseline actual, para que la comparación sea honesta),
- `last-n` (últimas 2-4 capas descongeladas, LR discriminativo),
- `full` (todo, LR bajo + warmup).

Cada régimen publica su curva unseen en los mismos cortes (62 k / 250 k /
500 k / 1 M) para que la pendiente sea comparable, no sólo el punto final.
Registrar coste: tiempo por 1 M y memoria pico en MPS — descongelar el backbone
multiplica ambos y el operador debe ver el precio.

## Verification gate

- Los tres regímenes corren con la misma seed, el mismo mix y el mismo eval.
- Test: el manifest de cada checkpoint declara `backbone.frozen` y el número de
  parámetros entrenables reales, verificado contra el `state_dict`.
- El gate escribe `artifacts/gates/T-unfreeze-backbone/gate.json` con las tres
  curvas y el delta de pendiente frente al baseline congelado.

## Done when

- Existe la curva unseen de los tres regímenes sobre el mismo corpus.
- Al menos un régimen tiene pendiente **no decreciente** de 250 k a 1 M.
- Está registrado el coste (min/1 M, GB pico) de cada régimen.
