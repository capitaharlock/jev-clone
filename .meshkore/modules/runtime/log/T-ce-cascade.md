---
id: T-ce-cascade
title: Latencia real — cascada para casos difíciles, cuantización y paridad Rust
status: backlog
priority: medium
owner: unassigned
category: runtime
initiative: shared-state-distill
depends_on:
  - T-ce-distill
created: 2026-09-25
updated: 2026-09-27
---

> **Archivada el 2026-09-27** por decisión del operador: el único objetivo es un modelo que aprende y mejora cada día (`docs/guia-un-solo-objetivo.md`). No se despacha. Se reactiva sólo si el operador lo pide.

# Latencia real — cascada para casos difíciles, cuantización y paridad Rust

Cierre del camino a producto, y sólo cuando haya calidad que servir. Si el modelo
de estado compartido de `#T-ce-distill` pierde demasiado, la referencia se mantiene
en **cascada**: el modelo rápido decide, y los casos difíciles —baja confianza, K
grande, familia conflictiva— suben al scorer completo.

Incluye lo que el proyecto ya tiene a medias y no puede darse por resuelto:
integración del motor en el servidor, carga del encoder afinado, y la clave de
caché V1 que la auditoría dejó señalada. No lo garantiza que el runtime sea
rápido: se verifica **sobre el checkpoint finalmente elegido**, antes de cualquier
release.

## Verification gate

- Test: paridad numérica PyTorch↔Rust sobre el checkpoint elegido, con tolerancia
  escrita.
- Test: la clave de caché de estado distingue estados distintos (el fallo señalado
  en la auditoría tiene test de regresión).
- Test: la cascada mide su tasa de escalado y su latencia p95 **con** el escalado
  incluido, no sólo la ruta rápida.

## Done when

- Latencia p95 y memoria medidas en hardware real, con la cascada activa.
- Cuantización aplicada sin pérdida de calidad fuera del margen escrito.
- Paridad Rust/Python verificada sobre el checkpoint elegido y el encoder afinado
  cargándose de verdad en el servidor.
