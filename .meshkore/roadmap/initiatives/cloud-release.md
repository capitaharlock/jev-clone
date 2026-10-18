---
reactivated: 2026-09-20
id: cloud-release
title: Cloud service and research release
status: done
owner: architect-master
modules:
  - cloud
created: 2026-09-19
updated: 2026-09-20
completed_at: 2026-09-20T10:27:23.300Z
commit_sha: ea2978e8c22eb81733f34caac3d2794f89c6bead
---
# Cloud service and research release

Del binario local al servicio: API pública de decisiones, contenedor de
inferencia sin Python, prototipo en RunPod Serverless (una región,
scale-to-zero) y research release con model card y métricas.

Tramo final de la ejecución V1: queda visible para expresar el resultado,
pero sus dos tasks sólo arrancan después de los gates locales anteriores.
Seleccionar las cinco iniciativas en **Run All** conserva esa clausura.

Cubre stack §§21–27 y plan §§93–94, 134.

## Done when

- El mismo binario corre local y en Docker sin cambiar semántica; el adapter de
  RunPod pasa contract tests y la prueba live se hace sólo con credenciales y
  autorización (criterio stack §39).
- API pública documentada con tipos V1 `choice`/`boolean` + `unknown`.
- Bundle de research release local con checkpoint, manifests y calibration
  report; publicar o desplegar es una operación posterior explícita.
- Model card enlaza fuentes, licencias, hashes, métricas y limitaciones; no se
  publica ningún peso con un gate legal o de contaminación abierto.
