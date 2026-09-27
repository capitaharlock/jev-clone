---
id: T-loop-scoreboard
title: El marcador diario — una fila por día, contra Jev y Laya, en el dashboard
status: next
priority: high
owner: unassigned
category: eval
initiative: daily-learning-loop
depends_on:
  - T-ce-finetune
  - T-ingest-laya
created: 2026-09-27
updated: 2026-09-27
---

# El marcador diario — una fila por día, contra Jev y Laya, en el dashboard

## Contexto

Hoy el estado de un checkpoint vive en gates sueltos. Para «mejorar cada día»
hace falta **una fila por día, append-only, con las mismas columnas siempre**,
y una curva que el operador pueda mirar en 5 segundos. `eval/scoreboard.py`
(«ONE command, ONE checkpoint, ONE table») y `tools/training_monitor.py`
(job `dashboard`, :8794) son la base; no se crean otros.

## Columnas (fijas; cambiar una es una versión nueva del marcador)

- `date`, `checkpoint_sha`, `mix_sha`, `train_rows_consumed`
- batería de desarrollo (`data/battery_dev.jsonl`, sha): forzada, con
  abstención, macro por familia, contrafactual conjunto — cada una con n,
  azar, IC95 %; desglose por idioma y por K.
- **typed-decisions test** (`#T-ingest-laya`, 2 000 decisiones): accuracy con
  IC95 %, y las líneas de referencia **Jev 0,727** y **Laya 0,766**
  (constantes citadas con fuente, no medidas por nosotros hasta que
  `#T-teacher-probe` las reproduzca).
- BANKING77 K=77 (`eval/fullspace.py`, 3 080 filas): accuracy; referencia Jev
  0,870 publicada, Laya 0,425.
- `promoted: true|false` y `reason` (lo escribe `#T-loop-nightly`).

## Qué hacer, paso a paso

1. `eval/scoreboard.py daily --checkpoint <dir> --out
   artifacts/scoreboard/history.jsonl`: calcula todo por comando, **anexa**
   (nunca reescribe), y falla si el sha de la batería cambió respecto a la
   fila anterior sin `--battery-rotated <motivo>`.
2. Reglas de coherencia (`eval/gate_rules.py`): una fila sin n/IC/azar es
   inválida; una accuracy por encima de 1 − abstención es inválida; una fila
   que cite Jev/Laya como «medidas» sin `#T-teacher-probe` done es inválida.
3. Dashboard: `tools/training_monitor.py` lee `history.jsonl` y pinta la
   curva de dev (forzada y contrafactual) y typed-decisions con las líneas de
   Jev y Laya. Sin librerías nuevas.
4. Gate `artifacts/gates/T-loop-scoreboard/gate.json`: ≥ 3 filas reales de
   checkpoints distintos (el sin ajustar, el del smoke, el de 5 000).

## Done when

- `history.jsonl` existe con ≥ 3 filas reales, coherentes, y el dashboard
  las pinta con las referencias.
- Ningún número del marcador se puede escribir sin el comando (test).
