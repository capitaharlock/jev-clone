---
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
completed_at: 2026-09-20T12:30:00Z
resolved_by: A004
resolved_by_conv: general-09192230
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

#T-release terminada con gate PASS: bundle atómico en `artifacts/gates/T-release/release.json`
(11 evidencias con sha256 re-verificado, ningún claim sin evidencia), model card con
limitaciones en `.meshkore/docs/model-card.md`, `GET /metrics` con contadores Prometheus
sin contenido de usuario, flujo train-en-Pod contra stub con destroy garantizado
(`.meshkore/docs/pod-train.md`). El trabajo nuevo (SPDX en manifests, `metrics.rs`,
gate `python-release`) rompió el linaje, así que revalidé la cadena entera de 18 gates
en orden, todos PASS bajo un solo árbol (`cb1a1a35`, config=cpu). Con esto #cloud-release
queda 2/2 y se cierra.

— T-release · bundle research release verificable, model card, métricas sin contenido y Pod flow con gate PASS
