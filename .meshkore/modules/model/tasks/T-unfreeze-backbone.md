---
id: T-unfreeze-backbone
title: Descongelar el backbone — capacidad entrenable donde está el lenguaje
status: done
priority: high
owner: unassigned
category: model
initiative: generalization-fix
depends_on:
  - T-antiscale-diag
created: 2026-09-22
updated: 2026-09-24
completed_at: 2026-09-23T07:56:46.372Z
resolved_by: A035
resolved_by_conv: general-09222211
commit_shas: ['c57cb5562b3c8cb25095c5a1e720b96b4e889b61', 'd6461d3f6b04a844e03645756a58bbfda7b363db']
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
se le enseñó a la cabeza. Descongelar bajo un objetivo que ya premia el mapa cerrado compra más
capacidad para la misma lección. **Vuelve a
ser la primera** si la cabeza ×2 o ×4 aplana la pendiente 250 k → 1 M.

## Estado (2026-09-23) — primera, y ya en ejecución

La condición de la sección anterior se cumplió: `#T-antiscale-diag` midió el
eje 3 y la cabeza ×2 aplana la pendiente 250 k → 1 M de −0,1665 a −0,0549. La
capacidad entrenable **sí** mueve la pendiente, así que esta task pasa a ser la
primera y a `active`.

Implementado (commit `1dea217`): `--unfreeze {none,last-n,full}`,
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

## Resolution

Dos cierres y un relevo en marcha.

**GPU Metal (`#T-metal-throughput`, commit `d6461d3`).** Estaba dispatch-bound, no lenta: el bucle puntuaba fila a fila. Ahora una llamada por batch → **×2,7** en la config real, ×4,5 congelado, GPU al **97,4 %** a B=128. El run de 1M bajó de ~11 h a **128 min**.

**Backbone descongelado: NO-GO.** El brazo `last-n=2` terminó y no aplana nada — peor en el gate independiente (0,2422 vs 0,2879 unseen). El brazo `full` no se lanza.

**Relevo**: job `lever-stack-d512-prior` (~2 h) — apilar las dos palancas que sí funcionaron, nunca medidas juntas.

<details><summary>#T-unfreeze-backbone — NO-GO, gate en disco</summary>

Mismo corpus, seed, objetivo y evals que el brazo congelado; única variable, el régimen del backbone.

| rows | none (congelado) | last-n=2 @1e-5 |
|---|---|---|
| 250 k | 0,3215 | 0,3281 |
| 1 M | 0,2347 | **0,2354** |
| pendiente | −0,0868 | **−0,0927** |

Gate independiente `eval.unseen` (n=5624, otro corte): unseen **0,2422 vs 0,2879**, seen 0,4864 vs 0,4753. Mejor en seen, peor en unseen = las dos capas extra se gastaron en memorizar el espacio de etiquetas de entreno.

Coste medido: 128,1 min/1M, 4,1 GB de pico. `artifacts/gates/T-unfreeze-backbone/gate.json`.</details>

<details><summary>#T-metal-throughput — el A/B, y por qué no toca la métrica</summary>

`PointerDecisionHead.forward_batch`: misma matemática sobre un eje de opciones con padding a K_max, con máscaras en la set-attention, en el resumen permutación-invariante y en los scores; `unknown` en la columna compartida K_max y gold remapeado. Migrados entreno, `evaluate` y `eval.calib.predict`.

| B | régimen | ms/step fila | ms/step batch | speedup | GPU % |
|---|---|---|---|---|---|
| 64 | last-n | 1175 | 441 | ×2,67 | 56→58 |
| 128 | last-n | 2304 | 701 | ×3,29 | 58→**97** |
| 256 | last-n | 4981 | 1566 | ×3,18 | 60→99 |
| 64 | congelado | 1017 | 227 | ×4,48 | 51→86 |

B=128 es el óptimo; B=256 regresa por presión de memoria. **Los runs de la curva siguen a B=64 a propósito** — cambiar la batch cambiaría la optimización y toda la familia 250k–1M se midió a 64.

Tests de equivalencia fila a fila con K=1..7 desiguales y estado con padding, tolerancia 1e-5. 159/159 verdes.
</details>

<details><summary>#T-lever-stack — lo que está corriendo y qué decide</summary>

Tres hipótesis medidas contra la anti-monotonía, dos sobreviven:

| palanca | pendiente 250k→1M |
|---|---|
| nada (congelado d256) | −0,1899 |
| cabeza ×2 (d512) | **−0,0549** |
| objetivo prior | **−0,0868** |
| backbone last-n | −0,0927 ✗ |

Nunca se han apilado, y son mecanismos distintos (capacidad vs quitar la recompensa de memorizar), así que a priori no compiten. Brazo: ettin-68m + d512 + prior a 1M, todo lo demás idéntico al brazo prior d256 que hace de control.

Regla escrita antes de mirar: si el apilado no mejora −0,0549, el eje capacidad/objetivo se declara agotado y el relevo pasa al espacio de etiquetas (`#T-labelspace-div`) — ahí es donde entran los datasets generados y la referencia externa.
</details>



**Commit** `c57cb551a` (+1) · 12 files · 4.7M tokens

**Files changed (12):**
- `.meshkore/docs/coverage.md`
- `.meshkore/modules/model/tasks/T-lever-stack.md`
- `.meshkore/modules/model/tasks/T-metal-throughput.md`
- `.meshkore/modules/model/tasks/T-unfreeze-backbone.md`
- `.meshkore/public/jobs.yaml`
- `artifacts/gates/T-metal-throughput/gate.json`
- `artifacts/gates/T-unfreeze-backbone/gate.json`
- `eval/calib.py`
- `model/decision_head.py`
- `model/test_decision_head.py`
- `training/python/test_train_decision.py`
- `training/python/train_decision.py`

## Alcance del veredicto — fase 1 (2026-09-24)

El NO-GO de esta task se midió con la pérdida sobre K≤8 opciones muestreadas,
el objetivo de fase 1. En ese régimen descongelar el encoder no tiene por qué
ayudar: con 8 candidatos la representación congelada ya basta para acertar, así
que la pérdida no tenía presión que transmitir al backbone.

Por la regla R4 (`.meshkore/docs/fase-2-espacio-completo.md`) el veredicto vale
dentro de su régimen y **no se transfiere**: la pregunta se rehace en
`#T-encoder-finetune` bajo el objetivo de espacio completo, donde la presión sí
cae sobre la representación. Los números de aquí se conservan como registro de
fase 1.
