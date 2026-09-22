---
id: T-gen-loop
title: Continuous decision-data generator loop
status: done
owner: rjj-mac-claude
category: data
initiative: data-training
depends_on:
  - T-halt-contam
  - T-gen-schemas
created: 2026-09-20
updated: 2026-09-21
created_by: live-anchor-loop
created_by_conv: general-09192230
---
# Continuous decision-data generator loop

**Bloqueada por la auditoría `.meshkore/docs/audit-2026-09-21.md`
(hallazgos C y D).** El loop tal y como está produce 258 700 filas desde 190
esqueletos y 5 plantillas frescas de Q-W-E-N en 11 h: volumen sin
información. Se para en `#T-halt-contam` y se rehace como generador de
esquemas de decisión completos en `#T-gen-schemas`.

No se cierra como `done` ni se borra: queda bloqueada como el registro de por
qué el generador cambió de contrato.

## 2026-09-21 architect — desbloqueada

Sus dos dependencias cerraron: `#T-halt-contam` mató el generador viejo y
puso sus 360 700 filas en cuarentena (commit `88ad4e6`), y `#T-gen-schemas`
entregó el contrato nuevo — esquemas de decisión completos, 1 152 esquemas
desde 1 152 esqueletos únicos (commit `39e6322`). El trabajo que queda es el
que la propia task anunciaba: relanzar el loop continuo **sobre el contrato
nuevo**, con el gate de diversidad como criterio de parada.

## 2026-09-21 A025 — relanzada sobre el contrato nuevo

`tools/gen_schemas/loop.py` + `tools/data_gen_loop.py --loop`. El loop genera
por rondas sobre `gen-schemas/v1`, **mide cada lote contra el corpus que ya
existe** y lo rechaza entero si lo degradaría; los lotes rechazados van a
`rejected/` y nunca entran en `corpus/`. `run.json` + un índice append-only de
textos normalizados lo hacen reanudable: al reiniciar se reconstruye la presión
de dedup en lugar de regenerar lo que ya hay.

### Criterio de parada (fijado antes de medir)

* **Primario** — diversidad marginal < **250 esqueletos nuevos por hora de
  reloj** (intervalo de pacing incluido), promediada sobre las últimas 4
  rondas. Por debajo, 24 h de ejecución añaden < 6 000 esqueletos: menos de
  cinco segundos del run acotado de `#T-gen-schemas`. Es ~14x el ritmo de
  información del loop que mató `#T-halt-contam` (190 esqueletos en 11 h ≈ 17/h).
* **Secundario** — 3 lotes consecutivos rechazados por el gate por lote.
* Backstops (no son el criterio): `max_rows`, `max_rounds`, `max_hours`.

### Gate por lote

`min_rows 48 · min_novelty 0.30 · max_domain_share 0.40 ·
max_language_share 0.55 · min_k_values 3 · min_hard_negative_rate 0.80 ·
unknown_rate ∈ [0.05, 0.35]`.

### Medido (no asumido)

Run de saturación sin pacing (`--interval 0`, seed 20260921), guardado en
`artifacts/gates/T-gen-loop/saturation-run.json`: **el loop paró solo en la
ronda 185**, por el criterio secundario ("3 lotes consecutivos rechazados").

| | loop viejo (`#T-halt-contam`) | loop nuevo |
|---|---:|---:|
| filas | 258 700 | 37 216 |
| esqueletos únicos | 190 | 37 216 |
| **filas por esqueleto** | **1 361,6** | **1,0** |
| novelty primera → última ronda | no se medía | 1,000 → 0,276 |
| lotes rechazados | no había gate | 12 (1 520 filas, todas en `rejected/`) |
| razones de rechazo | — | `novelty` x10, `unknown_rate` x3 |
| parada | la mató un humano a las 11 h | criterio de diversidad, ronda 185 |

La diversidad marginal nunca bajó de 250/h antes de que el gate por lote
cortara: **para este generador local el criterio que ata es el suelo de
novelty (0,30), no el suelo por hora**, que queda como cota exterior. Ambos
van escritos en `gate.json`.

Job del daemon: **`gen-loop`** (`--interval 30`). Reanudable verificado en
producción: dos reinicios reconstruyeron el índice desde disco (1 273 y 2 466
filas) y siguieron en la ronda siguiente sin regenerar nada.

## Done when

1. `python3 tools/data_gen_loop.py --loop` corre como job del daemon, sobrevive
   al turno y **termina solo**, con la razón escrita en `run.json`.
2. Todo lote que falle el gate queda en `rejected/`; `corpus/` tiene
   exactamente las filas aceptadas y **1.0 filas por esqueleto**.
3. `run.json` permite matar y relanzar sin regenerar: el índice se reconstruye
   y las filas previas siguen contando como duplicados.
4. `artifacts/gates/T-gen-loop/gate.json` publica pass/fail, el criterio de
   parada, la curva de diversidad observada y filas aceptadas vs rechazadas.
5. `python3 -m unittest tools.test_gen_loop` en verde.
