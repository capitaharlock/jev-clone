---
id: T-benchmark-v3
title: Benchmark v3 con fuentes reales y margen medido frente a JEv
status: active
priority: high
owner: architect-master
category: eval
initiative: decision-sota
depends_on:
  - T-real-sources
created: 2026-10-15
updated: 2026-10-15
---

# Benchmark v3

Lo construye la fábrica (`decision-dataset-factory`); TinyJev solo lo lee a
través de un adaptador, como hace con v2.

## Arenas

Productos con fichas, reseñas, corrección de código por ejecución, velocidad
de código medida, respuestas técnicas, conocimiento, ciencia, matemáticas,
literatura, inferencia y hechos, preferencia multicriterio y abstención.

## Criterios

- Split por entidad: problema, producto, pregunta, autor.
- El gold sale de la etiqueta de la fuente, de un programa o de la ejecución.
  Ningún modelo decide el gold.
- Base, campeón, Qwen y JEv se miden sobre las mismas filas antes de fijarlo.
- El corte sellado se abre una vez, al final de la iniciativa.

## Done when

- 30.000 casos de desarrollo y 10.000 sellados, fijados por hash.
- Informe por arena con intervalo de Wilson para los cuatro modelos.
- La comparativa de Evaluation muestra v3 en lugar de la batería de 400 filas.
