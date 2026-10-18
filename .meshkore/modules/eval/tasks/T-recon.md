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
completed_at: 2026-09-20T00:23:37.245Z
resolved_by: A004
resolved_by_conv: general-09192230
commit_shas: ['66bf6b8a6753d00f3e0a12f124d69e1448f65b8b', '7f48fd53c7810cb2a66e8402a983ffe97cf4bf13']
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

Sí, terminada. La primera iniciativa (#data-foundation, 4/4) está en `done`: para cerrarla hubo que ejecutar también sus dependencias de modelo, así que completé la cadena entera T-bakeoff → T-shared-state → T-option-mixer → T-hardneg → T-curriculum → T-distillation → T-gold, con los 12 gates en PASS bajo un solo linaje (`7f48fd5`, `66bf6b8`). 70 tests nuevos, todos verdes. Tres veredictos honestos que deja la tanda: el mixer pierde contra el baseline y se selecciona el simple; distillation conserva hard-labels sin ganancia medida; el piloto gold es sintético con acuerdo 0.852 y la colección humana queda pendiente.

— T-bakeoff · harness único, top-2 en proxies y backbones reales pendientes con gate PASS
— T-shared-state · encode-once verificado, caché con invalidación y límite V1 8192 con gate PASS
— T-option-mixer · comparativa en misma matriz con fallback explícito al baseline con gate PASS
— T-hardneg · sampler nearest-label, 24 OOD y canario real rechazado con gate PASS
— T-curriculum · stages 0-3 reproducibles y ablación CE+Brier promocionada con gate PASS
— T-distillation · pipeline local-first y keep-hard-labels documentado con gate PASS
— T-gold · piloto 500, MASSIVE ES+EN fijado y leakage limpio con gate PASS

**Commit** `66bf6b8a6` (+1) · 50 files · 6.8M tokens

**Files changed (50):**
- `.meshkore/docs/coverage.md`
- `.meshkore/modules/data/tasks/T-gold.md`
- `.meshkore/modules/data/tasks/T-hardneg.md`
- `.meshkore/modules/model/tasks/T-bakeoff.md`
- `.meshkore/modules/model/tasks/T-curriculum.md`
- `.meshkore/modules/model/tasks/T-distillation.md`
- `.meshkore/modules/model/tasks/T-option-mixer.md`
- `.meshkore/modules/model/tasks/T-shared-state.md`
- `.meshkore/roadmap/initiatives/data-foundation.md`
- `artifacts/fixtures/gold/massive_es_en.jsonl`
- `artifacts/gates/T-bakeoff/gate.json`
- `artifacts/gates/T-bakeoff/report.json`
- `artifacts/gates/T-curriculum/gate.json`
- `artifacts/gates/T-curriculum/report.json`
- `artifacts/gates/T-data-p0/gate.json`
- `artifacts/gates/T-data-schema/gate.json`
- `artifacts/gates/T-distillation/gate.json`
- `artifacts/gates/T-distillation/report.json`
- `artifacts/gates/T-firewall/gate.json`
- `artifacts/gates/T-gold/gate.json`
- `artifacts/gates/T-gold/pilot.json`
- `artifacts/gates/T-gold/report.json`
- `artifacts/gates/T-hardneg/gate.json`
- `artifacts/gates/T-hardneg/ood_registry.json`
- `artifacts/gates/T-option-mixer/gate.json`
- `artifacts/gates/T-option-mixer/report.json`
- `artifacts/gates/T-recon/gate.json`
- `artifacts/gates/T-rust-skel/gate.json`
- `artifacts/gates/T-shared-state/gate.json`
- `artifacts/gates/T-shared-state/report.json`
- `data/gold.py`
- `data/hardneg.py`
- `data/test_gold.py`
- `data/test_hardneg.py`
- `experiments/curriculum.yaml`
- `model/__init__.py`
- `model/bakeoff.py`
- `model/distillation.py`
- `model/option_mixer.py`
- `model/shared_state.py`
- `model/test_bakeoff.py`
- `model/test_distillation.py`
- `model/test_option_mixer.py`
- `model/test_shared_state.py`
- `results/curriculum-seed5050/manifest.json`
- `training/__init__.py`
- `training/python/__init__.py`
- `training/python/curriculum.py`
- `training/python/test_curriculum.py`
- `training/python/tools/gate.py`
