---
id: T-loop-nightly
title: El bucle sin fin — orquestador de ciclos, escalera de volumen y de brazos, hitos
status: next
priority: high
owner: unassigned
category: model
initiative: daily-learning-loop
depends_on:
  - T-loop-trainer
  - T-loop-scoreboard
  - T-numeric-gen
created: 2026-09-27
updated: 2026-09-27
---

> **Reescrita el 2026-09-27 (decisión del operador: escalar sin parar).** Ya no es «una noche»: es `training/python/loop.py` (`run --forever`, `status`), el **proceso B** de `docs/bucle-infinito.md`. Implementa §1–§5 del manual **literalmente**: ciclo reanudable por etapa, `artifacts/loop/state.json`, `artifacts/loop/cycle-XXXX/`, escalera de volumen (§2), la regla de promoción de abajo (§3, **sin cambios**), escalera de brazos por racha (§4) e hitos (§5, que llaman a `#T-ce-confirm`). La etapa `datagen` sale de este proceso: la hace el productor (`#T-episode-scale` → `data.stream`), y el bucle consume lo publicado. Tests obligatorios: la regla de promoción (media sube + familia con caída significativa → no promueve; todo sube → promueve), la escalera (promueve → sube escalón; racha 1..5 → brazo correcto), la reanudación (matar en TRAIN y relanzar → retoma sin repetir ciclo) y la inanición (sin datos → `starved`, no inventa). El «Done when» de abajo pasa a ser: 10 ciclos reales seguidos en `history.jsonl` con `decision.json`, y al menos un cambio de escalón y uno de brazo ejercitados.

# La noche encadenada — un comando, siete etapas, y la regla de promoción que no se negocia

## Contexto

Con trainer que continúa, marcador diario y generación por día, sólo falta
**encadenarlo** y decidir automáticamente si el checkpoint de hoy sustituye al
de ayer. La regla se escribe aquí, antes de correr, y se implementa con test.

## Las siete etapas (un script, `training/python/nightly.py`, cada etapa con su log y su código de salida)

1. `datagen` — `data.episode_gen run --seed <YYYYMMDD> --n <cuota>` (+`--resume`).
2. `verify` — verificador separado y contrafactuales (`#T-episode-verify`,
   `#T-counterfactuals`); cuarentena a su fichero.
3. `splits` — se AÑADE al train por `variant_group` (`#T-episode-splits`);
   dev y sellado intactos (el script comprueba sus shas antes y después).
4. `mix` + `trainer` — `mix_manifest build` + `ce_finetune train --init current`.
5. `evalgate` — `ce_finetune eval` sobre dev; typed-decisions test; BANKING77.
6. `scoreboard` — `eval.scoreboard daily` anexa la fila.
7. `promote` — regla de abajo; escribe `artifacts/checkpoints/ce/current.json`
   y `promoted`/`reason` en la fila.

Se ejecuta como **un solo `command` del job canónico `trainer`** (los demás
jobs quedan libres); un fallo en cualquier etapa aborta las siguientes y
**no promueve**. Máximo 8 h; si se pasa, aborta y lo dice.

## Regla de promoción (escrita el 2026-09-27, antes de la primera noche)

Se promueve el checkpoint de hoy si y sólo si, en dev, contra el `current`:

- la forzada **y** el contrafactual conjunto suben con IC95 % de la
  diferencia (pareada por fila) que no cruza 0; **y**
- ninguna familia ni idioma baja más de lo que su IC95 % permite
  (diferencia pareada cuyo IC superior sea < 0 = caída = no promueve).

Si no se promueve, `current` no cambia y la fila lleva `reason` con la
familia/idioma/métrica culpable. Tres noches seguidas sin promoción → el
script lo escribe en el diario y para el bucle hasta que un humano mire
(`#T-loop-error-mining` es lo que hay que mirar primero).

## Done when

- El script existe, cada etapa es reanudable, y una etapa rota no promueve
  (test con etapa simulada).
- La regla de promoción tiene test: media que sube con una familia que cae
  → no promueve; todo sube → promueve.
- Siete noches reales consecutivas en `history.jsonl` con `promoted`/`reason`.
