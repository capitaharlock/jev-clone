---
status: done
id: T-release
title: Research release and observability
status: done
priority: medium
owner: general-09192230
category: cloud
initiative: cloud-release
depends_on:
  - T-cloud-api
created: 2026-09-19
updated: 2026-09-20
completed_at: 2026-09-20T10:27:23.526Z
resolved_by: A004
resolved_by_conv: general-09192230
commit_shas: ['ea2978e8c22eb81733f34caac3d2794f89c6bead', '4492b30f6655e72222f4e3fd480a3938af4e3668']
---
# Research release and observability

M5 del plan: checkpoint + inference library + API + model card
(metodología de calibración pública) + benchmark suite + manifests +
hardware results + limitaciones. Observabilidad tracing/OpenTelemetry/
Prometheus (latencias por fase, GPU, cache hits, errores) sin loguear
contenido de states. Entrenamiento cloud en RunPod Pods solo si el
hierro local no basta.

Fuente: stack §§24, 28; plan §§134, 170.

## Verification gate

- Requiere PASS de `T-cloud-api`; el runner recorre transitivamente los 17
  gates anteriores y rechaza huecos, hashes incompatibles o NO-GO no resueltos.
- Tests de instalación offline cargan la release desde cero, ejecutan golden
  decisions y verifican firmas/hashes, licencias, SBOM y ausencia de secretos o
  contenido de states en logs.
- Model card, benchmark, calibración, dataset manifests y limitaciones se
  contrastan automáticamente contra los artefactos raw; ningún claim puede
  existir sin evidencia enlazada.
- `T-release` produce un bundle candidato local. Publicar, desplegar o subir a
  git requiere una acción posterior explícita del operador y rol privilegiado.

## Done when

- Research release completa con model card y limitaciones.
- Métricas de producción emitiéndose sin contenido de usuario.
- Flujo train-en-Pod documentado y probado contra stub; si llega a usarse live,
  el gate verifica que el Pod se destruye tras cada run.
- Pesos, licencias, dataset manifests, hashes y calibration report forman una
  release atómica; cualquier gate pendiente la bloquea.

# 2026-09-20 architect: blocked — chains on T-cloud-api (blocked); revisit after it ships.
# 2026-09-20 general: T-cloud-api PASS — unblocked, dispatchable.

## Resolution

#T-release cerrada con gate PASS (commits `4492b30` + `ea2978e`): bundle con 11 evidencias verificadas, model card, métricas sin contenido y Pod flow con destroy garantizado. El trabajo nuevo había roto el linaje, así que revalidé los 18 gates en orden, todos PASS bajo un solo árbol. #cloud-release queda 2/2; solo resta #decision-model en `active` con su opcional en `backlog`.
<details><summary>Cadena — 18 gates PASS (config=cpu, árbol cb1a1a35)</summary>

- rust-skel → data-schema → firewall → data-p0 → recon → bakeoff → shared-state → option-mixer → hardneg → curriculum → distillation → gold → calib → local-infer → state-cache → quant-onnx → cloud-api → release.
- Cada gate corre la regresión acumulada; un run rojo nunca deja PASS.
</details>

— T-release · bundle research release verificable, model card, métricas y Pod flow con gate PASS

**Commit** `ea2978e8c` (+1) · 60 files · 23M tokens

**Files changed (60):**
- `.meshkore/docs/model-card.md`
- `.meshkore/docs/pod-train.md`
- `.meshkore/modules/cloud/tasks/T-cloud-api.md`
- `.meshkore/modules/cloud/tasks/T-release.md`
- `.meshkore/modules/runtime/tasks/T-quant-onnx.md`
- `.meshkore/roadmap/initiatives/cloud-release.md`
- `Cargo.lock`
- `artifacts/gates/T-bakeoff/gate.json`
- `artifacts/gates/T-bakeoff/report.json`
- `artifacts/gates/T-calib/gate.json`
- `artifacts/gates/T-cloud-api/coldstart.json`
- `artifacts/gates/T-cloud-api/gate.json`
- `artifacts/gates/T-curriculum/gate.json`
- `artifacts/gates/T-data-p0/gate.json`
- `artifacts/gates/T-data-schema/gate.json`
- `artifacts/gates/T-distillation/gate.json`
- `artifacts/gates/T-firewall/gate.json`
- `artifacts/gates/T-gold/gate.json`
- `artifacts/gates/T-hardneg/gate.json`
- `artifacts/gates/T-local-infer/gate.json`
- `artifacts/gates/T-option-mixer/gate.json`
- `artifacts/gates/T-option-mixer/report.json`
- `artifacts/gates/T-quant-onnx/gate.json`
- `artifacts/gates/T-recon/gate.json`
- `artifacts/gates/T-release/gate.json`
- `artifacts/gates/T-release/release.json`
- `artifacts/gates/T-rust-skel/gate.json`
- `artifacts/gates/T-shared-state/gate.json`
- `artifacts/gates/T-state-cache/bench.json`
- `artifacts/gates/T-state-cache/gate.json`
- `crates/jev-bench/Cargo.toml`
- `crates/jev-cli/Cargo.toml`
- `crates/jev-core/Cargo.toml`
- `crates/jev-format/Cargo.toml`
- `crates/jev-model/Cargo.toml`
- `crates/jev-quant/Cargo.toml`
- `crates/jev-runtime/Cargo.toml`
- `crates/jev-server/Cargo.toml`
- `crates/jev-server/src/lib.rs`
- `crates/jev-server/src/metrics.rs`
- `data/__pycache__/__init__.cpython-314.pyc`
- `data/__pycache__/adapters.cpython-314.pyc`
- `data/__pycache__/firewall.cpython-314.pyc`
- `data/__pycache__/gold.cpython-314.pyc`
- `data/__pycache__/hardneg.cpython-314.pyc`
- `data/__pycache__/leakage.cpython-314.pyc`
- `data/__pycache__/registry.cpython-314.pyc`
- `data/__pycache__/schema.cpython-314.pyc`
- `data/__pycache__/seed.cpython-314.pyc`
- `data/__pycache__/test_adapters.cpython-314.pyc`
- `data/__pycache__/test_data_schema.cpython-314.pyc`
- `data/__pycache__/test_firewall.cpython-314.pyc`
- `data/__pycache__/test_gold.cpython-314.pyc`
- `data/__pycache__/test_hardneg.cpython-314.pyc`
- `eval/__pycache__/__init__.cpython-314.pyc`
- `eval/__pycache__/calib.cpython-314.pyc`
- `eval/__pycache__/recon.cpython-314.pyc`
- `eval/__pycache__/test_calib.cpython-314.pyc`
- `eval/__pycache__/test_recon.cpython-314.pyc`
- `model/__pycache__/__init__.cpython-314.pyc`
