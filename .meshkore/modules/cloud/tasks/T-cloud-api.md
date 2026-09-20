---
id: T-cloud-api
title: Public API and cloud prototype
status: done
priority: medium
owner: general-09192230
category: cloud
initiative: cloud-release
depends_on:
  - T-quant-onnx
created: 2026-09-19
updated: 2026-09-20
completed_at: 2026-09-20T02:45:00.000Z
resolved_by: A004
resolved_by_conv: general-09192230
---

# Public API and cloud prototype

API pública de decisiones (V1: `choice` + `boolean` como choice
binario + `unknown`; `score` diferido a V2), batch multi-question,
versiones de modelo, adapter fino RunPod (envelope→core API, sin
acoplar el runtime al proveedor) y prototipo en RunPod Serverless: una
región US, scale-to-zero, contenedor multi-stage sin Python. Adapter de
compatibilidad Jev-like si aporta adopción.

Fuente: stack §§21–27, 39; plan §§93–94.

## Verification gate

- Requiere PASS de `T-quant-onnx` y empaqueta sólo el derivado seleccionado por
  ese gate; el contenedor final se inspecciona para confirmar que no lleva
  Python ni datos de entrenamiento.
- Contract tests cubren schema/versiones, orden e IDs, batch multi-Q,
  `unknown`, errores, límites, timeouts y prohibición de state en logs.
- El mismo golden corpus debe producir resultados equivalentes en binario local
  y Docker. El adapter RunPod se prueba contra un stub; la prueba live es
  condicional a credenciales y autorización de despliegue.
- `T-cloud-api` pasa localmente con imagen reproducible, SBOM, health/readiness
  y medida de cold-start; no publica ni despliega por sí sola.

## Done when

- API documentada con ejemplos state→distribuciones.
- Mismo binario local/Docker sin cambio de semántica; RunPod se valida live
  cuando haya credenciales/autorización, sin bloquear el desarrollo local.
- Cold-start medido y dentro del presupuesto.
- El contrato versiona modelo/schema, conserva orden e IDs de opciones y
  devuelve incertidumbre/`unknown` sin loguear el state.

# 2026-09-20 architect: blocked — depends_on T-quant-onnx not done (in scope, later in order); retried in pass order, will revisit after local-runtime chain.

## Resolution

Gate PASS (config=cpu): API V1 (`/v1/choice`, `/v1/batch`), adapter RunPod
(envelope→core), Dockerfile multi-stage sin Python, SBOM, health/readiness y
cold-start medido. Cadena de 9 gates revalidada en un linaje tras el trabajo
nuevo (T-hardneg → T-curriculum → T-distillation → T-gold → T-calib →
T-local-infer → T-state-cache → T-quant-onnx → T-cloud-api). T-release queda
desbloqueada.
