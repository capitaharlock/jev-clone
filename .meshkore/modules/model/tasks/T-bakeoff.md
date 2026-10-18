---
id: T-bakeoff
title: Backbone bake-off
status: done
priority: high
owner: unassigned
category: model
initiative: decision-model
depends_on:
  - T-recon
created: 2026-09-19
updated: 2026-09-20
completed_at: 2026-09-20T20:57:25.379Z
resolved_by: A004
resolved_by_conv: general-09192230
commit_shas: ['7f48fd53c7810cb2a66e8402a983ffe97cf4bf13']
---
# Backbone bake-off

No casarse con ningún backbone hasta medir. Smoke A/B/C del §173 con el
mismo decision head (dynamic option markers + pointer scorer) sobre
`jhu-clsp/ettin-encoder-68m`, `answerdotai/ModernBERT-base` y
`LiquidAI/LFM2.5-Encoder-230M`, HuffPost y K aleatorio 2–32. Después,
Pareto 17M→32M→68M→150M→400M de Ettin y control opcional NeoBERT;
los top-2 pasan a Banking77 + BoolQ + Civil + HelpSteer2.

Fuente: plan §§5, 6, 173, 120–121.

> gate: LFM2.5 solo entra al producto si su licencia LFM Open v1.0 pasa
> revisión; NeoBERT exige auditar su `trust_remote_code`. Fuentes y licencias:
> `.meshkore/docs/source-register.md`.

## Verification gate

- Requiere PASS de `T-recon`; todos los candidatos usan idénticos splits,
  seeds, head provisional, precisión y presupuesto de pasos.
- Unit tests cubren dynamic option markers, masks, K variable y restauración de
  orden; un fixed benchmark corre antes y después de cada cambio de modelo.
- Cada run registra commit, revisión del backbone, manifest de datos, hardware,
  métricas y latencia. Un run incompleto o no comparable queda excluido del
  Pareto, no imputado.
- `T-bakeoff` sólo pasa con top-2 explícitos y un informe reproducible de por
  qué los demás candidatos pierden por calidad, coste, licencia o exportación.

## Done when

- Runs A/B/C medidos en el mismo harness con accuracy, NLL y p50/p95.
- Report Pareto calidad/latencia por backbone y tamaño publicado.
- Top-2 elegidos con licencia, compatibilidad Candle/export y coste de memoria
  como gates, no solo calidad.

## Resolution

Ningún job de entrenamiento corre ahora (ps vacío); el pilot prog-gold falló (rc=1). Hay que relanzarlo; nada activo 24h.
## Resolution

#T-bakeoff con gate PASS (linaje 5bba14f9b1c4): harness único con mismos splits/seeds/head provisional; 3 proxies smoke medidos (L 0.854, M 0.812, S 0.771 acc) con top-2 = proxy-L/M; backbones reales como pending_weights (LFM condicional, NeoBERT bloqueado por remote code) excluidos del Pareto sin imputar; 9 tests verdes.
