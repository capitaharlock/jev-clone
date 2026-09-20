---
id: T-option-mixer
title: Option interaction and order invariance
status: done
priority: medium
owner: unassigned
category: model
initiative: decision-model
depends_on:
  - T-shared-state
created: 2026-09-19
updated: 2026-09-20
completed_at: 2026-09-20T00:21:58Z
resolved_by: A004
resolved_by_conv: general-09192230
---

# Option interaction and order invariance

Las opciones de una pregunta interactúan (listwise), no se puntúan
aisladas. Baseline de scorer independiente → mixer DeepSets → Set
Transformer de 1–2 capas; invariancia al orden con permutation loss y
benchmarks de shuffle/inserción.

Fuente: plan §§13–14, 23.4, 64–65.

## Verification gate

- Requiere PASS de `T-shared-state`.
- Property tests permutan opciones y verifican la permutación inversa de las
  probabilidades; insertion tests comprueban estabilidad frente a distractores
  no competitivos.
- La misma matriz compara scorer independiente, DeepSets y Set Transformer con
  seeds y presupuesto iguales, incluyendo hard semantic siblings.
- `T-option-mixer` pasa únicamente si la mejora de robustez y el overhead de
  latencia quedan medidos; si ningún mixer gana, se selecciona explícitamente
  el baseline simple en vez de bloquear la cadena.

## Done when

- Mixer supera al scorer independiente en hard semantic siblings.
- Option Permutation Stability e Insertion Stability por encima del umbral.
- Coste del mixer medido en el breakdown de latencia.

## Resolution

#T-option-mixer con gate PASS (linaje 5bba14f9b1c4): misma matriz para independent/DeepSets/Set-Transformer; estabilidad a permutación 1.0 en los tres (equivarianza por construcción), inserción 0.924 gana el baseline → selección explícita del baseline simple con coste de latencia medido; 6 tests verdes.
