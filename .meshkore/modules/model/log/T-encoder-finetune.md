---
superseded_by: cross-encoder-pilot / T-ce-finetune
id: T-encoder-finetune
title: Reabrir el fine-tune del encoder bajo el objetivo nuevo
status: superseded
priority: high
owner: unassigned
category: model
initiative: full-space-training
depends_on:
  - T-bigk-optsets
  - T-fullspace-objective
created: 2026-09-24
updated: 2026-09-25
---

# Reabrir el fine-tune del encoder bajo el objetivo nuevo

> **Reformulada el 2026-09-25 (plan de recuperación §8).** Sus cuatro brazos de
> descongelado estaban condicionados a «el objetivo ganador» de
> `#full-space-training`, y ese objetivo cerró en NO-GO: no hay ganador de
> calidad que transmitir al encoder. Entrenar otra cabeza aleatoria sobre
> representaciones congeladas repite justo la limitación que hay que someter a
> prueba. Se sustituye por `#T-ce-finetune`: ajustar un scorer semántico **ya
> preentrenado para relacionar textos**, con su propia medición sin entrenar como
> control previo.

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
ha ejecutado en este repo.

## Dos correcciones de la revisión externa

**La acumulación de gradiente NO conserva los negativos in-batch** (hallazgo I).
Acumular cuatro pérdidas calculadas por separado con B=32 actualiza cada 128
filas, pero cada fila sigue viendo sólo los candidatos de su microbatch: el
promedio de cuatro log-softmax no es el log-softmax sobre la unión. Si no cabe
el batch que pide el objetivo muestreado, o se comparten las representaciones y
se calcula la pérdida conjunta preservando gradientes, o **se declara que se ha
cambiado el objetivo**. Cuando cada fila lleva ya su espacio exacto completo
(la vía de `#T-bigk-optsets`) esta limitación no aplica igual: los dos modos se
distinguen en el gate.

**La memoria no está medida para esta configuración.** `T-metal-throughput/gate.json`
mide encoder congelado y `last-n (2 capas)` con cabeza d256: no acredita full
fine-tuning con d512 y K grande. Antes de prometer el run, una medición breve de
memoria y de pasos completos en la configuración real de cada brazo.

## Done when

- Medidas memoria y velocidad de paso de los cuatro brazos en su configuración
  real (d512, K del objetivo ganador) antes de lanzar ninguno.
- Los cuatro brazos corridos a 62 k con el objetivo que haya ganado entre la vía
  exacta y la muestreada, misma seed y mismo corpus, y sus `eval.fullspace` a 77
  vías en una tabla con azar, K e IC.
- El brazo ganador repetido a 250 k **y en otra seed** para confirmar que la
  pendiente no se invierte, antes de comprometer ningún run de 1 M (R9).
- `#T-unfreeze-backbone` queda marcada como **superseded** por esta task, con
  la razón escrita en su cuerpo (medida en el régimen de fase 1).
- Veredicto: cuánta capacidad entrenable hace falta, o NO-GO con la siguiente
  hipótesis nombrada (tamaño de backbone, no entrenabilidad).
