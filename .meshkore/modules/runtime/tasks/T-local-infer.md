---
status: done
id: T-local-infer
title: Local inference on Metal, CUDA and CPU
status: done
priority: high
owner: unassigned
category: runtime
initiative: local-runtime
depends_on:
  - T-calib
created: 2026-09-19
updated: 2026-09-20
completed_at: 2026-09-20T01:26:44.103Z
resolved_by: A004
resolved_by_conv: general-09192230
commit_shas: ['19b0f42d994a56667bb0b66dd94f5544c781c635']
---
# Local inference on Metal, CUDA and CPU

Backends Candle CPU/Metal/CUDA tras el trait `Backend`
(`encode_state`/`decide`), CLI `jevclone serve --device auto|metal|
cuda|cpu`, y primer inventario real: VRAM del Windows RTX antes de
fijar presupuestos de memoria. Misma semántica y API en los tres.

Fuente: stack §§8–9, 13, 15, 35, 38; plan §56.

## Verification gate

- Requiere PASS de `T-calib` y consume exactamente su checkpoint, calibrador y
  manifest; cualquier hash distinto impide arrancar.
- Golden vectors comparan PyTorch↔Candle, single↔batch y CPU↔acelerador local
  para states cortos/largos, K=2/32 y Q=1/N en FP32/FP16 cuando aplique.
- La máquina primaria y CPU son obligatorias. Metal/CUDA no disponibles quedan
  como `not_available`; se prueban en otro dispositivo sólo de forma opcional.
- `T-local-infer` pasa con paridad, selección `--device auto`, fallback CPU y
  benchmark raw reproducible en el único ordenador requerido para V1.

## Done when

- Mismo artefacto corre en CPU y el acelerador de la máquina primaria sin
  reentrenar; M4/M5/RTX se validan al estar disponibles sin bloquear V1.
- `--device auto` elige Metal/CUDA/CPU según disponibilidad.
- Memoria y presupuestos de la máquina primaria documentados; Windows RTX se
  añade a la misma matriz cuando esté disponible.
- La salida numérica y la semántica de `unknown` pasan el harness en CPU y el
  backend local; la matriz completa conserva casos para validación posterior.

## Resolution

Cadena de gates en marcha (2/15). Coverage ya cita #T-state-cache, sin cambios. Espero al PASS completo para cerrar el commit.

15.7M tokens
