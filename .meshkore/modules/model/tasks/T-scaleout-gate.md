---
id: T-scaleout-gate
title: Criterio de activación e inventario x3
status: backlog
priority: low
owner: unassigned
category: model
initiative: train-scaleout
depends_on:
  - T-curriculum
created: 2026-09-20
updated: 2026-09-20
---

# Criterio de activación e inventario x3

Puerta de entrada de `#train-scaleout`: nada de dispositivos extra hasta
que esté escrito **cuándo** se activan y **con qué hierro**. Fija el
umbral de activación (volumen de datos de train+verificación, presión de
calendario de la cola de experimentos) y refresca el inventario hardware
de las tres máquinas (I0-T1, `benchmarks/hardware.json`): Mac M4 Max
48 GB, Mac M5 48 GB, Windows NVIDIA (GPU exacta y VRAM detectadas, stack
§9). Deja además el baseline single-machine (tiempo/época en workload de
referencia, seed+manifest de paridad) contra el que se medirá el A/B de
`#T-dist-train`.

Fuente: plan §§123–125, I0-T1; stack §§8–9, §34.

## Verification gate

- Requiere PASS de `T-curriculum`; ninguna task obligatoria depende de
  esta.
- Tests: el script de inventario detecta las 3 máquinas y su backend
  (MPS/MPS/CUDA) de forma determinista; la evaluación del umbral sobre
  fixtures (volumen por debajo/encima) da no-activar/activar sin
  ambigüedad; el baseline reproduce tiempo/época dentro de tolerancia
  con mismo seed+manifest.
- El gate escribe `artifacts/gates/T-scaleout-gate/gate.json` con
  `pass: true`, hash del inventario, umbral aplicado y referencia al
  baseline; sin ese artifact no arranca `#T-dist-train`.

## Done when

- Umbral de activación escrito (volumen + calendario) con valores, no
  intenciones.
- `benchmarks/hardware.json` refrescado para las 3 máquinas con
  GPU/VRAM detectadas.
- Baseline single-machine publicado y reproducible por seed+manifest.
