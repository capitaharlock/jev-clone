---
id: T-candle-infer
title: Inferencia local real — Candle Metal/CUDA en jev-model con fallback Python
status: backlog
priority: medium
owner: unassigned
category: runtime
initiative: oss-release
depends_on:
  - T-train-real
created: 2026-09-21
updated: 2026-09-27
---

> **Archivada el 2026-09-27** por decisión del operador: el único objetivo es un modelo que aprende y mejora cada día (`docs/guia-un-solo-objetivo.md`). No se despacha. Se reactiva sólo si el operador lo pide.
# Inferencia local real — Candle Metal/CUDA en jev-model con fallback Python

> **Actualización 2026-09-24.** El punto de partida de abajo está desfasado:
> `crates/jev-model/src/{engine,modernbert,pointer}.rs` ya tiene implementación
> Candle real del encoder y del pointer head. Lo que falta es la integración de
> producto —servidor conectado al `Engine`, cargador que lea el encoder
> afinado— y vive en `#T-serve-engine`. Lo que queda aquí es la paridad
> numérica y las latencias por backend.

Punto de partida original: **no había Candle en `Cargo.toml`**. El workspace
Rust no tenía backend de inferencia neuronal. Lo que sí existía y es sólido es
`crates/jev-runtime`: cache de estado con clave compuesta
(`model_version`, `state_hash`, `tokenizer_hash`), LRU, batching dinámico con
ventana 0-2 ms, backpressure y cancelación. Se reutiliza tal cual — le falta
el motor debajo.

Trabajo:

1. `candle-core` + `candle-nn` en `crates/jev-model`, con features `metal` y
   `cuda`, y CPU como fallback siempre compilable.
2. Cargar el checkpoint **safetensors** + `tokenizer.json` de `#T-train-real`
   e implementar el forward del pointer head: `encode_state` una vez →
   cross-attention → score por opción. La clave compuesta del cache de
   `jev-runtime` ya contempla `model_version` y `tokenizer_hash`: se respeta.
3. **Paridad numérica** contra el runner Python de referencia: mismo prompt →
   mismo argmax, y diferencia de probabilidades por debajo de tolerancia
   declarada. El runner Python se conserva como fallback y como oráculo de
   paridad, no se tira.
4. Verificación real en hardware: M-series (Metal) y NVIDIA (CUDA), con
   latencia p50/p95 medida en cada uno dentro del presupuesto 70-500 ms.
5. `jevclone serve` expone la decisión sobre el modelo real, no sobre un
   scorer.

## Verification gate

- Test de paridad sobre 1 000 filas: 100 % de coincidencia de argmax con el
  runner Python, y ECE que no se degrada al cambiar de backend.
- Benchmark warm y cold publicado por backend (Metal, CUDA, CPU).
- El gate escribe `artifacts/gates/T-candle-infer/gate.json` con `pass: true`,
  paridad, latencias por backend y `model_version`.

## Done when

- `crates/jev-model` carga safetensors y ejecuta el pointer head en Metal,
  CUDA y CPU.
- La paridad con el runner Python está verificada sobre 1 000 filas.
- Hay latencias p95 medidas en M-series y en NVIDIA, no estimadas.

## Resolution

**Cancelado, no fallido (2026-09-21).** El agente fue detenido con SIGTERM (exit 143) por orden del operador: la prioridad pasó a entrenar el modelo (#data-training), no a empaquetar la release. No hubo error de ejecución.

Dejó en el worktree, sin commit: `.gitignore`, `.github/workflows/ci.yml`, `Cargo.toml`/`Cargo.lock`. El índice de git quedó con 9 168 `git rm --cached` huérfanos; el architect los desestageó (ficheros intactos en disco).
Reanudar desde ahí cuando la release vuelva a estar en scope.
