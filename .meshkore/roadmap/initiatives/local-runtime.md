---
id: local-runtime
title: Local runtime and artifact parity
status: done
owner: architect-master
modules:
  - runtime
created: 2026-09-19
updated: 2026-09-19
completed_at: 2026-09-20T01:41:37.439Z
commit_sha: 99f509f082ff9cee1d0ad3a30e7c766a50f88da7
---
# Local runtime and artifact parity

Rust owns the product: workspace de crates, backends Candle
(CPU/Metal/CUDA), CLI `jevclone serve`, servidor Axum con state cache y
batching, paridad numérica PyTorch↔Candle, y derivados optimizados
(INT8, ONNX, TensorRT/CoreML) solo tras profiling.

Abre la cadena V1 con el arnés y vuelve a cerrarla tras calibración con
inferencia, cache y export. Se selecciona con las otras cuatro iniciativas:
los cuatro gates están ordenados por dependencias, no por el orden visual.

Cubre stack §§3–20, 28–31, 35–40 y plan §§57–62, 88–90.

## Done when

- El mismo `model.safetensors + config + tokenizer` corre en CPU y el
  acelerador de la máquina primaria sin reentrenar; M4/M5/Windows RTX se
  validan cuando estén disponibles, sin bloquear V1.
- Paridad PyTorch↔Candle dentro de tolerancia en la matriz del stack §30.
- `encode_state` una vez + 100 packs de preguntas demuestra p50 warm de
  pocos ms (§177).
- API/CLI idénticas en los backends disponibles y cubiertas por los mismos
  golden vectors para incorporar después los restantes.
- Cada artefacto y derivado pasa paridad, calibración y hash validation.
