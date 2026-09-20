---
status: done
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
completed_at: 2026-09-20T08:13:19.285Z
resolved_by: A004
resolved_by_conv: general-09192230
commit_shas: ['220f7e2d6258509ee982fbf413229bf909a01404']
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

#T-cloud-api terminada con gate PASS (commit `220f7e2`): API V1, adapter RunPod, Docker sin Python, SBOM y cold-start. El trabajo nuevo había roto el linaje, así que revalidé 9 gates en orden, todos PASS bajo un solo árbol. Queda una sola tarea en #cloud-release: #T-release (ya desbloqueada, en `next`).
<details><summary>Cadena revalidada — 9 gates PASS</summary>

- T-hardneg → T-curriculum → T-distillation → T-gold → T-calib → T-local-infer → T-state-cache → T-quant-onnx → T-cloud-api, config=cpu, árbol `9460986a`.
- Cada gate corre la regresión acumulada (fmt, clippy, tests Rust + 18 grupos Python); un run rojo nunca deja PASS.
</details>
<details><summary>T-cloud-api — qué entrega</summary>

- `POST /v1/choice` y `/v1/batch` (choice + boolean binario + `unknown`, orden e IDs preservados, sin state en logs), `POST /runpod` (envelope→core API).
- `crates/jev-server/src/v1.rs` + `runpod.rs`, `Dockerfile` multi-stage sin Python, `sbom.json`, `.meshkore/docs/api-v1.md`, cold-start en `artifacts/gates/T-cloud-api/`.
</details>

— T-cloud-api · API pública V1, adapter RunPod, Docker reproducible y SBOM con gate PASS

**Commit** `220f7e2d6` · 21 files · 17.5M tokens

**Files changed (21):**
- `.meshkore/docs/api-v1.md`
- `.meshkore/modules/cloud/tasks/T-cloud-api.md`
- `.meshkore/modules/cloud/tasks/T-release.md`
- `Dockerfile`
- `artifacts/gates/T-calib/gate.json`
- `artifacts/gates/T-cloud-api/coldstart.json`
- `artifacts/gates/T-cloud-api/gate.json`
- `artifacts/gates/T-curriculum/gate.json`
- `artifacts/gates/T-distillation/gate.json`
- `artifacts/gates/T-gold/gate.json`
- `artifacts/gates/T-hardneg/gate.json`
- `artifacts/gates/T-local-infer/gate.json`
- `artifacts/gates/T-quant-onnx/gate.json`
- `artifacts/gates/T-state-cache/gate.json`
- `crates/jev-server/Cargo.toml`
- `crates/jev-server/src/lib.rs`
- `crates/jev-server/src/runpod.rs`
- `crates/jev-server/src/v1.rs`
- `sbom.json`
- `training/python/test_cloud_api.py`
- `training/python/tools/gate.py`
