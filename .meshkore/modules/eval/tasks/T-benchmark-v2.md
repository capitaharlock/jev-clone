---
id: T-benchmark-v2
title: Benchmark v2 por capacidad y dominio
status: active
priority: critical
owner: unassigned
category: eval
initiative: honest-eval
depends_on:
  - T-battery-dev
  - T-dev-rotation
created: 2026-10-13
updated: 2026-10-13
---

# Benchmark v2

La fábrica independiente declara y construye una matriz de 10 capacidades por
10 dominios, en español e inglés. Son 10.000 decisiones: 1.000 por capacidad y
1.000 por dominio.

## Capas

- Borrador estructural: gold programático, distribución y esquema auditables.
- Prosa: Qwen reescribe, pero debe conservar entidades y números; no toca gold.
- Fuentes públicas: revisión, hash, licencia y decisión de redistribución.
- Sellado: origen independiente, invisible para train y desarrollo.

## Integración

El gate antiguo de 400 casos sigue decidiendo cada promoción. Benchmark v2 es
diagnóstico: 1.000 filas cada ciclo y 10.000 cada cinco ciclos. No alimenta el
selector de brazos. Una candidatura de release pasa además ocho permutaciones y
el corte sellado.

## Éxito

- Cada capacidad supera 95%, no solo la media.
- El límite inferior Wilson al 95% supera 95% para sostener la afirmación.
- Se publica el peor cruce capacidad por dominio.
- El gold no procede de una respuesta del modelo generador.
- Ninguna fila de evaluación entra en entrenamiento.

## Estado

La fábrica ya produce 10.000 filas válidas en 200 celdas. El refinado de prosa
es incremental entre ciclos. La capa de fuentes y el corte sellado siguen
abiertos; mientras tanto el resultado se etiqueta como diagnóstico estructural.
