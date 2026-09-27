---
id: shared-state-distill
title: Recuperar la latencia — destilar a estado compartido (futuro)
status: backlog
owner: architect-master
modules:
  - model
  - runtime
created: 2026-09-25
updated: 2026-09-27
---

> **Archivada el 2026-09-27** por decisión del operador: el único objetivo es un modelo que aprende y mejora cada día (`docs/guia-un-solo-objetivo.md`). No se despacha. Se reactiva sólo si el operador lo pide.

# Recuperar la latencia — destilar a estado compartido (futuro)

Paso 6 del plan de recuperación: **sólo tras una mejora confirmada** en
`#cross-encoder-pilot`. Está en `backlog` a propósito y ninguna task obligatoria
depende de ella.

El cross-encoder relee el estado por cada opción. Eso es aceptable para
demostrar que la tarea es aprendible, y no lo es para el contrato de producto
(una pasada de estado, caché compartida, 20–40 ms). Cuando exista un modelo
competente, se destilan sus puntuaciones a una arquitectura con estado
compartido —bi-encoder o interacción tardía— y **se mide cuánto pierde**. Si
pierde demasiado, la referencia se mantiene en cascada para los casos difíciles.

Regla del plan que esta iniciativa respeta: **no elegir la arquitectura
definitiva por latencia antes de demostrar competencia.**

## Done when

- El modelo destilado conserva calidad y calibración dentro de un margen escrito
  antes de medir, o queda documentada la pérdida y la cascada que la compensa.
- Latencia y memoria medidas en hardware real, no estimadas.
