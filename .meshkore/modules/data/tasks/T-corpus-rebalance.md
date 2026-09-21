---
id: T-corpus-rebalance
title: Rebalanceo del corpus — civil-comments deja de ser el 74,6 %
status: next
priority: medium
owner: unassigned
category: data
initiative: data-training
depends_on:
  - T-optset-sampler
created: 2026-09-21
updated: 2026-09-21
---

# Rebalanceo del corpus — civil-comments deja de ser el 74,6 %

Hallazgo E: de 2 681 195 ejemplos totales, `civil-comments` (toxicidad
binaria) aporta 1 999 514 = **74,6 %** del corpus. No enseña a elegir entre
opciones dinámicas: es una tarea binaria de etiqueta fija. El corpus útil
para el producto es una fracción minoritaria de lo que se está entrenando.

La estrategia de `data-training` ya fijaba guardrails (≤15 % por dataset,
≤30 % por familia) y el propio plan decía que "nunca 89 % sin que el pipeline
lo marque como error". Esta task los hace cumplir de verdad y los aplica a la
mezcla que alimenta a `#T-train-real`.

Trabajo:

1. Bajar `civil-comments` al nivel de los demás por muestreo (el `sample`
   de `JOBS` ya existe: pasa a ser la regla, no la excepción) y recontar la
   mezcla resultante.
2. Guardrails como **error duro** en el loader: un dataset > 15 % o una
   familia > 30 % aborta la construcción de la mezcla con un mensaje que dice
   cuál y cuánto.
3. Manifest de mezcla versionado: composición exacta, seed, shas por shard, y
   diversity index por dominio/familia/tipo/K/idioma.
4. Reponderar hacia lo que enseña la propiedad del producto: filas con
   opciones dinámicas y K variable por encima de las binarias de etiqueta
   fija.

## Verification gate

- Test: una mezcla construida con civil-comments al 74 % falla con error
  explícito.
- Test: el manifest reproduce la mezcla bit a bit desde seed + shas.
- El gate escribe `artifacts/gates/T-corpus-rebalance/gate.json` con
  `pass: true` y la composición final por dataset y familia.

## Done when

- Ningún dataset supera el 15 % de la mezcla ni ninguna familia el 30 %.
- El manifest versionado reproduce la mezcla exactamente.
- La proporción de filas con opciones dinámicas está publicada y es mayoría.
