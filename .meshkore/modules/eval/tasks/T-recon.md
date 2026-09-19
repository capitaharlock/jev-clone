---
id: T-recon
title: Reconnaissance and benchmark harness
status: done
priority: high
owner: unassigned
category: eval
initiative: eval-trust
depends_on:
  - T-data-p0
created: 2026-09-19
updated: 2026-09-20
completed_at: 2026-09-20T00:00:00.000Z
resolved_by: A004
resolved_by_conv: general-09192230
---

# Reconnaissance and benchmark harness

I0 del plan: inventario de hardware (M4 Max, M5, RTX), ejecución de
GLiClass, Laya, Verdict, `kev`, `jevlike` y baseline LLM pequeño, con
informe de ideas robables y límites medidos. Más harness de latencia
correcto: buckets de state/Q/opciones, cold vs warm, p50/p95/p99,
memoria y throughput por máquina, y descomposición tokenize/encode/
fusión/head.

El inventario fija CPU/GPU/VRAM/RAM, SO, drivers y versiones de PyTorch/MPS/
CUDA/Candle. La máquina local es el único gate obligatorio; otros equipos
aportan resultados opcionales y nunca bloquean el camino principal. Ningún
número externo se extrapola como promesa para el hardware local.

Fuente: plan §§60–61, 66–67, 71, 129.

## Verification gate

- Requiere PASS de `T-data-p0` y ejecuta todos los baselines sobre el mismo
  snapshot de datos y protocolo del arnés.
- Un test de consistencia verifica warm-up, sincronización del dispositivo,
  aislamiento de tokenización y cálculo correcto de p50/p95/p99 sobre raw
  samples; no acepta sólo tiempos agregados.
- La máquina local ejecuta el smoke completo. Backends ausentes se registran
  como `not_available` con motivo, no como fallo ni como resultado inventado.
- `T-recon` queda PASS sólo si config, entorno, raw samples y resumen pueden
  regenerar la misma tabla de referencia.

## Done when

- Informe I0 con números propios por baseline y máquina.
- Harness de latencia corre en la máquina primaria con protocolo portable; los
  dos dispositivos extra se incorporan sólo cuando estén disponibles.
- Baselines 0–7 registradas como referencia fija del proyecto.
- Resultados distinguen cold/warm, state length, Q, K, batch, precisión y
  backend, con config y raw samples conservados.

## Resolution

#T-recon ejecutada con gate PASS en el linaje `7db4fba` (misma tanda revalida T-rust-skel → T-data-schema → T-firewall → T-data-p0, todo PASS): harness I0 con inventario portable, 8 baselines sobre el snapshot P0 (4 locales medidos, 4 pesados como `not_available` con motivo) y protocolo de latencia cold/warm con p50/p95/p99 sobre raw samples. Informe en `artifacts/gates/T-recon/recon-report.json`.
<details><summary>eval/recon.py + tests — qué incluye</summary>

- Inventario CPU/OS/Python/Torch/CUDA/MPS; snapshot P0 fijado por hash de fixtures + revisiones (misma semilla que T-data-p0).
- Baselines 0 random / 1 majority / 2 char-ngram-centroid / 3 tfidf-ponderado; 4–7 (GLiClass, Laya, Verdict/kev/jevlike, LLM) como `not_available` offline.
- `measure()` con warmup descartado, cold separado y raw samples; hash del informe solo sobre lo determinista.
- 9 tests nuevos; gate extendido con paso `python-recon` y `eval/` en el linaje.
</details>

— T-recon · harness I0, 8 baselines y gate PASS
