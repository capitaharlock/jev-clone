---
id: T-unfreeze-backbone
title: Descongelar el backbone — capacidad entrenable donde está el lenguaje
status: active
priority: high
owner: unassigned
category: model
initiative: generalization-fix
depends_on:
  - T-antiscale-diag
created: 2026-09-22
updated: 2026-09-23
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

## Prioridad (#T-antiscale-diag)

**Segunda de las dos.** No porque la sospecha del backbone congelado esté
descartada — el eje 3 de `#T-antiscale-diag` es justo lo que la mide y sigue
`measured: false` a la espera del job `antiscale-wide` — sino porque la misma
representación congelada sostiene accuracy unseen muy por encima del azar
antes del punto en que la curva gira: lo que cambia en el tramo es sólo lo que
se le enseñó a la cabeza. Descongelar bajo un objetivo que ya empuja en la
dirección equivocada compra más capacidad para la misma lección. **Vuelve a
ser la primera** si la cabeza ×2 o ×4 aplana la pendiente 250 k → 1 M.

## Estado (2026-09-23) — primera, y ya en ejecución

La condición de la sección anterior se cumplió: `#T-antiscale-diag` midió el
eje 3 y la cabeza ×2 aplana la pendiente 250 k → 1 M de −0,1665 a −0,0549. La
capacidad entrenable **sí** mueve la pendiente, así que esta task pasa a ser la
primera y a `active`.

Implementado (commit `aa52692`): `--unfreeze {none,last-n,full}`,
`--unfreeze-layers`, `--backbone-lr`, grupo de optimizador propio con LR
discriminativo, forward con grafo para estados y textos, bypass del memo de
textos, `backbone.safetensors` + sha256 en el checkpoint con verificación en
`load_checkpoint`, sha del backbone dentro de `model_version`, y
`summary.cost` (min/1 M + memoria pico). 46/46 tests verdes.

**El baseline `none` no se re-entrena**: es
`genobj-prior-1m-ettin-68m-s20260922` (congelado, prior-penalty 1.0, 1 M,
seed 20260922, mismo mix `decision-mix-v3` fence-clean).

- brazo `last-n` (2 capas + final_norm, backbone-lr 1e-5): job `unfreeze-lastn`,
  lanzado 2026-09-22T22:57Z, ~11 h en MPS (22,4 filas/s, 3,34 GB pico medidos
  en el probe), encadena `eval.unseen gate` sobre el checkpoint de 1 M.
- brazo `full`: **no lanzado**. Al ritmo medido cuesta 30 h+ de GPU y compite
  con el mismo MPS; se decide con la pendiente del `last-n` en la mano.
