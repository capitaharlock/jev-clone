---
id: T-state-cache
title: State cache and batching server
status: done
priority: high
owner: unassigned
category: runtime
initiative: local-runtime
depends_on:
  - T-local-infer
created: 2026-09-19
updated: 2026-09-20
---

# State cache and batching server

Servidor Axum+Tokio: handlers → cola → batch scheduler → GPU worker →
fan-out, con batching dinámico opcional (ventana 0–2 ms, bypass para
peticiones críticas) y state cache L1 GPU / L2 RAM con clave
(model_version, state_hash, tokenizer_hash). Benchmark §177: encode
once + 100 packs.

Fuente: stack §§17–20; plan §177.

## Verification gate

- Requiere PASS de `T-local-infer`.
- Tests concurrentes cubren hit/miss, invalidación por model/tokenizer/state,
  LRU/eviction, cancelación, backpressure y deduplicación de trabajos iguales.
- Cache vs uncached debe conservar golden vectors dentro de tolerancia; tests
  de carga prueban un límite explícito de trabajo GPU y memoria.
- `T-state-cache` pasa con benchmark cold/warm `encode once + 100 packs`, raw
  samples, queue latency y hit-rate; una meta de rendimiento fallida produce
  NO-GO medido, no una omisión del gate.

## Done when

- p50 warm-state de pocos ms por pack en el benchmark §177.
- Hit-rate de cache y latencia de cola en métricas.
- Peticiones concurrentes no lanzan trabajo GPU descontrolado.
