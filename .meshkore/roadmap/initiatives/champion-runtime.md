---
id: champion-runtime
title: Servir el campeón actual desde el runtime Rust
status: active
owner: architect-master
modules:
  - runtime
created: 2026-10-15
updated: 2026-10-15
---

# Servir el campeón actual desde el runtime Rust

`#local-runtime` cerró en septiembre con el cross-encoder de entonces. Desde
el ciclo 5 el campeón es TinyJev-4B: un backbone Qwen de 4B con LoRA r16 y una
cabeza de puntero, entrenado y evaluado en MLX. El runtime Rust no sabe
cargarlo, así que el producto sirve un modelo que ya no es el mejor.

Contexto de los tres frentes: `.meshkore/context/architecture.md`.

## Tareas

- `#T-champion-export`: fusionar el LoRA, exportar backbone y cabeza a
  safetensors con config, tokenizer y hash.
- `#T-candle-pointer`: cargar el backbone en Candle y portar la cabeza de
  puntero detrás del trait `Backend`.
- `#T-runtime-parity-v3`: paridad frente a MLX sobre la sonda v3 y latencias
  p50 y p95 en Metal y CPU.

## Done when

- `jevclone serve` responde con el campeón promovido más reciente.
- La elección del runtime coincide con MLX en al menos el 99,5 % de la sonda
  v3 y la diferencia de puntuaciones queda dentro de la tolerancia del stack.
- Cada promoción del autoloop produce una exportación nueva sin intervención.
