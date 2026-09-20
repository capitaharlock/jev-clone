---
id: decision-model
title: Decision model and training
status: active
owner: architect-master
modules:
  - model
created: 2026-09-19
updated: 2026-09-19
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
