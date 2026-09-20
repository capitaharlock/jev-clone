---
id: T-release
title: Research release and observability
status: blocked
priority: medium
owner: unassigned
category: cloud
initiative: cloud-release
depends_on:
  - T-cloud-api
created: 2026-09-19
updated: 2026-09-20
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
