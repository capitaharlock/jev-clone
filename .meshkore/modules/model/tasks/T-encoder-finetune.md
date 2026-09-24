---
id: T-encoder-finetune
title: Reabrir el fine-tune del encoder bajo el objetivo nuevo
status: next
priority: high
owner: unassigned
category: model
initiative: full-space-training
depends_on:
  - T-fullspace-objective
created: 2026-09-24
updated: 2026-09-24
---

# Reabrir el fine-tune del encoder bajo el objetivo nuevo

`#T-unfreeze-backbone` cerró en **NO-GO** el 2026-09-23: descongelar las 2
últimas capas a lr 1e-5 dio 0,2422 unseen contra 0,2879 del brazo congelado.
Ese veredicto se midió **con K≤8**, el objetivo de fase 1, y por eso no se
transfiere (regla R4): con 8 candidatos la representación congelada ya basta
para acertar, así que la pérdida no tenía presión que transmitir al encoder.

Bajo la pérdida sobre el espacio entero la situación se invierte: rankear
cientos de etiquetas por contenido es exactamente lo que exige que la
representación cambie, y es donde el profesor pone toda su capacidad (reentrena
el encoder entero). Esta task rehace la pregunta en el régimen de fase 2.

## Brazos

| brazo | entrenable | lr backbone |
|---|---|---|
| congelado (control) | ~11 M de cabeza | — |
| `last-n=2` | + 2 capas | 1e-5 con decay por capa |
| `last-n=6` | + 6 capas | 1e-5 con decay por capa |
| `full` | encoder entero (68 M) | 5e-6 → 2e-5, warmup + cosine |

El brazo `full` es el que reproduce la receta del profesor y es el que nunca se
ha ejecutado en este repo. Ettin-68M cabe entero en MPS a B=128 por la ruta
batched de `#T-metal-throughput`; si no cabe con el batch que pide el objetivo
nuevo, se acumula gradiente en vez de bajar el batch (el batch **es** el
denominador de la pérdida, bajarlo cambia la variable medida).

## Done when

- Los cuatro brazos corridos a 62 k con el objetivo de `#T-fullspace-objective`,
  misma seed y mismo corpus, y sus `eval.fullspace` a 77 vías en una tabla.
- El brazo ganador repetido a 250 k para confirmar que la pendiente no se
  invierte, antes de comprometer ningún run de 1 M.
- `#T-unfreeze-backbone` queda marcada como **superseded** por esta task, con
  la razón escrita en su cuerpo (medida en el régimen de fase 1).
- Veredicto: cuánta capacidad entrenable hace falta, o NO-GO con la siguiente
  hipótesis nombrada (tamaño de backbone, no entrenabilidad).
