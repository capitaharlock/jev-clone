---
id: T-shared-state
title: Shared-state V1 architecture
status: done
priority: high
owner: unassigned
category: model
initiative: decision-model
depends_on:
  - T-bakeoff
created: 2026-09-19
updated: 2026-09-20
completed_at: 2026-09-20T00:21:58Z
resolved_by: A004
resolved_by_conv: general-09192230
commit_shas: ['7f48fd53c7810cb2a66e8402a983ffe97cf4bf13']
---

# Shared-state V1 architecture

El experimento decisivo (§174): state encoder una vez + 2 capas de
fusión por pregunta, frente a full-cross-encoder `[state+question+
options]×Q`. Incluye state encoder API, question/option encoder ligero,
1 capa de cross-attention, pooling de opciones, pointer head, branch
batching y cache real con invalidación. El benchmark de estado largo usa
512/2K/4K/8K tokens, evidencia al principio/medio/final y decide el límite V1;
no se promete 8K con la misma latencia del camino común.

> scope V1: solo `choice` (`boolean` = choice binario). `score`,
> `extract` y `multiselect` quedan diferidos a V2 para no complicar el
> head ni el runtime. La ganancia de velocidad V1 viene de aquí: cero
> decoding autoregresivo, state encode-once, 1–3 capas de fusión.

Fuente: plan §§7–12, 22, 101–102, 174.

## Verification gate

- Requiere PASS de `T-bakeoff` y usa los top-2 sin cambiar splits ni protocolo.
- Unit tests cubren masks, option indexing, padding, 1–N preguntas, 2–32
  opciones, evidencia por posición e invalidación por versión/hash.
- Tests numéricos comparan single vs batch, cache vs uncached y full-cross vs
  shared-state; tolerancias y margen de calidad se fijan antes de mirar test.
- El gate `T-shared-state` conserva curvas raw de Q y longitud, y sólo pasa si
  state se codifica exactamente una vez y no aparece regresión silenciosa.

## Done when

- Shared-state conserva calidad del full-cross-encoder en P0 (±margen).
- Escalado sublineal al crecer Q demostrado en benchmark multi-Q.
- Cache de state con invalidación correcta y test de paridad.
- Curva calidad/latencia/memoria por longitud de state publicada y límite V1
  fijado con evidence-position stress.

## Resolution

#T-shared-state con gate PASS (linaje 5bba14f9b1c4): state encode-once verificado (1 vs Q encodes), paridad single/batch y cache/uncached a 1e-9, invalidación por versión/hash, límite V1 fijado en 8192 tokens con curvas raw por Q y longitud, calidad dentro del margen del full-cross; 10 tests verdes.
