---
id: T-candle-pointer
title: Backbone Qwen y cabeza de puntero en Candle
status: next
priority: high
owner: architect-master
category: runtime
initiative: champion-runtime
depends_on:
  - T-champion-export
created: 2026-10-15
updated: 2026-10-15
---

# Backbone Qwen y cabeza de puntero en Candle

Cargar el backbone exportado con los modelos Qwen de `candle-transformers` y
portar la cabeza de puntero a `jev-model`, detrás del trait `Backend`
(`encode_state` y `decide`) que ya usa `jevclone serve`.

## Criterios

- Mismo API y CLI que el runtime de septiembre.
- Backends Metal y CPU; CUDA cuando haya máquina.
- Golden vectors de la exportación en los tests de `jev-model`.
