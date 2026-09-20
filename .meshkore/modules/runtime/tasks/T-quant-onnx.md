---
id: T-quant-onnx
title: Quantization and export backends
status: done
priority: medium
owner: general-09192230
category: runtime
initiative: local-runtime
depends_on:
  - T-state-cache
created: 2026-09-19
updated: 2026-09-20
---

# Quantization and export backends

Orden estricto: FP16/BF16 referencia → INT8 dinámico → INT8 estático
(con recalibración) → INT4 experimental. ONNX como derivado (shapes
dinámicas K/Q o buckets), EPs CUDA/CoreML, TensorRT si el benchmark lo
justifica, comparador MLX en Apple. Kernels custom solo tras profiling.

Fuente: stack §§10–12, 14, 40; plan §§59, 88–90.

## Verification gate

- Requiere PASS de `T-state-cache`; FP16/BF16 es la referencia congelada antes
  de generar cualquier derivado.
- Golden vectors y stress suite comparan cada derivado con Candle canónico;
  INT8 repite calibración y ONNX cubre buckets K/Q, shapes inválidas y export
  round-trip.
- Cada backend se conserva sólo si mejora latencia/memoria en el hardware donde
  existe y respeta umbrales de accuracy, NLL, Brier y ECE predefinidos.
- `T-quant-onnx` puede pasar seleccionando FP16 y rechazando INT8/ONNX si no
  aportan; la decisión y los benchmarks quedan en el gate, sin bloquear cloud.

## Done when

- INT8 sin degradación material de calidad ni calibración.
- Benchmark Candle vs ONNX vs TensorRT por plataforma publicado.
- Ningún kernel custom sin perfil que lo justifique.
- Cada derivado referencia el checkpoint canónico y repite paridad,
  calibración y stress suite antes de ser releaseable.
