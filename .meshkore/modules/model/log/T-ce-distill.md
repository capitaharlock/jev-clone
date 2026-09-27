---
id: T-ce-distill
title: Destilar el scorer a estado compartido y medir cuánto pierde
status: backlog
priority: medium
owner: unassigned
category: model
initiative: shared-state-distill
depends_on:
  - T-ce-confirm
created: 2026-09-25
updated: 2026-09-27
---

> **Archivada el 2026-09-27** por decisión del operador: el único objetivo es un modelo que aprende y mejora cada día (`docs/guia-un-solo-objetivo.md`). No se despacha. Se reactiva sólo si el operador lo pide.

# Destilar el scorer a estado compartido y medir cuánto pierde

Sólo tras una mejora confirmada en `#T-ce-confirm`. El cross-encoder es la
referencia de calidad, no la arquitectura final: relee el estado por cada opción y
no puede cumplir el contrato de una pasada de estado con caché compartida.

Destilar sus puntuaciones a un modelo con **estado compartido** —bi-encoder o
interacción tardía— y publicar la pérdida de calidad, no estimarla. Antes de
destilar, generar 100 k ejemplos **dirigidos a los errores** del modelo confirmado,
no más volumen indiscriminado.

Regla del plan de recuperación que esta task respeta: no elegir la arquitectura
definitiva por latencia antes de demostrar competencia.

## Verification gate

- Test: el modelo destilado se mide en la misma batería, con el mismo protocolo, y
  la diferencia contra el maestro se publica por familia e idioma.
- Test: la calibración del destilado se verifica aparte; heredar la temperatura del
  maestro no cuenta como calibrado.

## Done when

- La pérdida de calidad y de calibración respecto al maestro está medida y
  publicada con IC95 %.
- Los 100 k dirigidos a errores existen con su manifest y su criterio de selección
  escrito.
- Si la pérdida excede el margen escrito antes de medir, queda documentado y el
  camino pasa a `#T-ce-cascade`.
