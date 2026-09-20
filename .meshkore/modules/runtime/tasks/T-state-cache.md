---
status: done
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
completed_at: 2026-09-20T01:35:08.547Z
resolved_by: A004
resolved_by_conv: general-09192230
commit_shas: ['99f509f082ff9cee1d0ad3a30e7c766a50f88da7']
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

## Resolution

#T-state-cache terminada con gate PASS. Toda la suite en verde por construcción (el gate corre fmt, clippy, tests Rust y los 17 grupos Python en cada eslabón) y la cadena completa de 15 gates queda en PASS bajo un solo linaje, commit `99f509f`. Siguiente en #local-runtime: #T-quant-onnx.

— T-state-cache · cache LRU con invalidación, batching/dedup/cancelación y bench encode-once + 100 packs con gate PASS

**Commit** `99f509f08` · 19 files · 16.4M tokens

**Files changed (19):**
- `.meshkore/modules/runtime/tasks/T-state-cache.md`
- `artifacts/gates/T-bakeoff/gate.json`
- `artifacts/gates/T-bakeoff/report.json`
- `artifacts/gates/T-calib/gate.json`
- `artifacts/gates/T-curriculum/gate.json`
- `artifacts/gates/T-data-p0/gate.json`
- `artifacts/gates/T-data-schema/gate.json`
- `artifacts/gates/T-distillation/gate.json`
- `artifacts/gates/T-firewall/gate.json`
- `artifacts/gates/T-gold/gate.json`
- `artifacts/gates/T-hardneg/gate.json`
- `artifacts/gates/T-local-infer/gate.json`
- `artifacts/gates/T-option-mixer/gate.json`
- `artifacts/gates/T-option-mixer/report.json`
- `artifacts/gates/T-recon/gate.json`
- `artifacts/gates/T-rust-skel/gate.json`
- `artifacts/gates/T-shared-state/gate.json`
- `artifacts/gates/T-state-cache/bench.json`
- `artifacts/gates/T-state-cache/gate.json`
