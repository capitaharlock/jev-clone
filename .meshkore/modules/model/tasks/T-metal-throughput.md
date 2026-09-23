---
id: T-metal-throughput
title: Saturar la GPU Metal — una llamada de cabeza por batch, no por fila
status: active
priority: high
owner: unassigned
category: model
initiative: train-scaleout
created: 2026-09-23
updated: 2026-09-23
---

# Saturar la GPU Metal — una llamada de cabeza por batch, no por fila

Antes de repartir runs entre tres máquinas hay que usar bien la que ya
existe. Medido el 2026-09-23 con `ioreg IOAccelerator`, la utilización del
dispositivo durante un run de entreno oscila entre 0 % y 87 % en ráfagas:
la GPU está esperando, no calculando.

La causa es estructural, no de configuración. El bucle de entreno puntuaba
la batch **fila por fila** (`for i, sample in enumerate(batch)` →
`row_logits`), así que un step de 64 filas encolaba 64 cadenas de kernels
diminutos — dos bloques de cross-attention sobre 3-9 opciones cada uno — y
el backward arrastraba las 64 en el grafo. Perfilado del step (B=64,
ettin-68m, last-2 descongeladas): backward 37 %, `encode_states` 29 %,
bucle de cabeza 23 %, embeddings 8 %.

Trabajo: `PointerDecisionHead.forward_batch` — la misma matemática sobre un
eje de opciones con padding a `K_max`, con máscaras en la set-attention, en
el resumen permutación-invariante y en los scores, de modo que la fila i
reciba exactamente lo que le daba el camino fila a fila. `unknown` pasa a
vivir en la columna compartida `K_max` y el gold se remapea. Los tres
consumidores del bucle (entreno, `evaluate`, `eval.calib.predict`) pasan a
la ruta batched, y las métricas bajan del dispositivo en una transferencia
por step en vez de una por fila.

Lo que NO entra: cambiar la matemática de la cabeza, el corpus, la seed o
el objetivo. Es una task de coste, y su gate lo trata como tal — si mueve
la métrica, está rota.

## Verification gate

- Test de equivalencia fila a fila entre `forward` y `forward_batch` con
  conjuntos de opciones desiguales (K=1..7) y estado corto con padding.
- Test de que el contenido de las ranuras de padding no llega a ninguna
  fila real (`pack_options` recolecta, no rellena con ceros).
- Test de que `pack_options` / `batch_gold` / `batch_prior_penalty`
  reproducen el layout y los números del camino fila a fila.
- A/B medido en el mismo proceso y la misma batch: ms/step y samples/s de
  las dos rutas, publicado en `artifacts/gates/T-metal-throughput/gate.json`.

## Done when

- `forward_batch` y el camino fila a fila coinciden a tolerancia float en
  los tests, con K desigual y padding de estado.
- El A/B publica un speedup ≥ 1,5× en ms/step sobre la config real de
  entreno (ettin-68m, B=64, last-n descongelado, MPS).
- Entreno, `evaluate` y `eval.calib.predict` usan la ruta batched y la
  suite de `model/` y `training/python/` sigue verde.
- El gate registra la utilización de GPU antes y después, medida con
  `ioreg IOAccelerator`, no estimada.
