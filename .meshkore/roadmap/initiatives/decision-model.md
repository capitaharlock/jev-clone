---
reactivated: 2026-09-20
id: decision-model
title: Decision model and training
status: done
owner: architect-master
modules:
  - model
created: 2026-09-19
updated: 2026-09-20
completed_at: 2026-09-20T18:39:54.361Z
commit_sha: 4c2f504ab6bca9cab77b7f8d63fd20dbf1d38584
---
# Decision model and training

Tercer tramo de la cadena V1: bake-off, shared state, invariancia de opciones,
curriculum y distillation. Se muestra como iniciativa propia, pero **Run All**
debe incluir las cinco iniciativas para que todas las dependencias estén dentro
del alcance.

Scope V1: `choice` y `boolean` binario, `unknown`, una máquina y un terminal.
`score`, `extract`, `multiselect`, multi-región y publicación real quedan
fuera. Cada task ejecuta la regresión acumulada y deja un gate PASS con hashes
antes de desbloquear la siguiente.

## Done when

- `Run All` completa los ocho gates obligatorios sin dependencias externas a
  esta iniciativa.
- El modelo shared-state V1, calibrado y con `unknown`, pasa calidad,
  robustez, paridad y presupuesto medido en la máquina local.
- El bundle local verificable incluye API, artefactos, model card, licencias,
  hashes y evidencia de cada claim; publicar o desplegar requiere otra orden.
