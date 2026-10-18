---
id: T-data-schema
title: Universal data schema and registry
status: done
priority: high
owner: unassigned
category: data
initiative: data-foundation
depends_on:
  - T-rust-skel
created: 2026-09-19
updated: 2026-09-20
---

# Universal data schema and registry

Un schema para todos los tipos (V1: `choice` + `boolean` como choice
binario; `score`/`extract`/`multiselect` diferidos a V2): state,
preguntas, opciones dinámicas, respuesta, confianza teacher, split.
Registry con hashes, dataset cards internas, license fence (qué se
puede mezclar y distribuir) y detector de leakage contra benchmarks.
Cada dataset registra fuente original, mirror operativo, licencia, uso
`train`/`eval-only`, revisión inmutable, SHA-256 y transformación en su card,
siguiendo `.meshkore/docs/source-register.md`.

Fuente: plan §§27–28, 121–122.

## Verification gate

- Requiere PASS de `T-rust-skel` y reutiliza su runner; ningún test se ejecuta
  fuera del arnés acumulativo.
- Property tests cubren JSONL/Arrow round-trip, IDs únicos, splits, cardinalidad
  2–42, `choice`/`boolean` y rechazo de registros incompletos.
- Tests negativos prueban que licencia, revisión o hash ausentes, y cualquier
  mezcla `eval-only`/`research-only` en train, bloquean la carga.
- Dos builds con el mismo manifest producen exactamente los mismos hashes y
  `python training/python/tools/gate.py --task T-data-schema` deja PASS.

## Done when

- Schema valida los 5 adapters P0 y rechaza `train` sin licencia/procedencia
  aprobada o sin revisión inmutable.
- Todo artefacto de entreno referencia un manifest con hash.
- Leakage detector corre en CI sobre cada dataset registrado.
- El registry separa material `train`, `research-only` y `eval-only` y bloquea
  mezclas o redistribución no autorizadas.
