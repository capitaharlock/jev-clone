---
id: T-runtime-parity-v3
title: Paridad del runtime con MLX sobre la sonda v3
status: next
priority: medium
owner: architect-master
category: runtime
initiative: champion-runtime
depends_on:
  - T-candle-pointer
created: 2026-10-15
updated: 2026-10-15
---

# Paridad del runtime con MLX sobre la sonda v3

Puntuar los 1.350 casos de la sonda v3 con el runtime y con MLX, comparar
elecciones y puntuaciones, y medir latencias.

## Criterios

- Elección idéntica en al menos el 99,5 % de los casos.
- Diferencia máxima de puntuación dentro de la tolerancia del stack §30.
- p50 y p95 publicados en el panel junto al informe v3 del campeón.
