---
id: T-loop-nightly
title: La noche encadenada — un comando, siete etapas, y la regla de promoción que no se negocia
status: next
priority: high
owner: unassigned
category: model
initiative: daily-learning-loop
depends_on:
  - T-loop-trainer
  - T-loop-scoreboard
  - T-episode-scale
created: 2026-09-27
updated: 2026-09-27
---

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
