---
id: T-bakeoff-real
title: Bake-off con pesos reales y retirada de los proxies del Pareto
status: next
priority: medium
owner: unassigned
category: model
initiative: decision-rebuild
depends_on:
  - T-train-real
created: 2026-09-21
updated: 2026-09-21
---

# Bake-off con pesos reales y retirada de los proxies del Pareto

`artifacts/gates/T-bakeoff/report.json` compara hoy encoders hash char-ngram
con pesos aleatorios (dim 256/1024/4096) y lista Ettin-68M, ModernBERT-base,
LFM2.5-230M y NeoBERT-250M como `pending_weights`. La honestidad del
artefacto es de agradecer —nunca se inventaron números— pero la decisión de
backbone sigue sin tomarse.

Con `#T-torch-stack` (pesos descargados) y `#T-train-real` (entrenamiento que
funciona), el bake-off se rehace midiendo lo que importa:

- Métrica primaria: accuracy + ECE **sobre etiquetas no vistas**
  (`#T-unseen-labels`), no accuracy de fila.
- Latencia p50/p95 por decisión con K=4 y estado realista, en MPS y CPU,
  dentro del presupuesto 70-500 ms que define el producto.
- Coste: parámetros, memoria en inferencia, tiempo de entreno hasta el corte.
- Pareto limpio: **ningún proxy aleatorio en la gráfica**. Los proxies se
  mueven a una sección "sanity checks" claramente separada o desaparecen.

Alcance mínimo real: Ettin-68M y ModernBERT-base entrenados de verdad.
LFM2.5-230M y NeoBERT-250M entran solo si sus pesos y licencia lo permiten;
si no, se declaran `not_available` con el motivo, como ya se hace.

## Verification gate

- Cada fila del Pareto tiene un `weights_sha256` y un `run_id` de
  `#T-train-real` detrás; una fila sin linaje hace fallar el gate.
- Test que rechaza la publicación si algún backbone del report tiene
  `pending_weights` o `random_init`.
- El gate reescribe `artifacts/gates/T-bakeoff/report.json` con `pass: true`
  y el veredicto top-2 escrito en prosa.

## Done when

- Top-2 de backbone elegido con números medidos, no proxies.
- El report publica accuracy/ECE unseen + latencia p95 + coste por candidato.
- La decisión y su porqué están en `.meshkore/docs/model-card.md`.
