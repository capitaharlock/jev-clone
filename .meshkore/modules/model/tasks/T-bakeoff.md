---
id: T-bakeoff
title: Backbone bake-off
status: next
priority: high
owner: unassigned
category: model
initiative: decision-model
depends_on:
  - T-recon
created: 2026-09-19
updated: 2026-09-20
resolved_by: A001
resolved_by_conv: _onboarding_v1
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

**Failed — exit 1.**

[codex error] You've hit your usage limit. Upgrade to Pro (https://chatgpt.com/explore/pro), visit https://chatgpt.com/codex/settings/usage to purchase more credits or try again at 3:06 AM.

5.8M tokens

Unblocked 2026-09-20: el intento previo cayó por límite de uso del agente
externo, no por causa técnica; su dependencia #T-recon ya está en PASS, así
que vuelve a `next` como siguiente de la cadena.
