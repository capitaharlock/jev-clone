---
id: data-foundation
title: Data foundation and governance
status: done
owner: architect-master
modules:
  - data
created: 2026-09-19
updated: 2026-09-20
completed_at: 2026-09-20T00:22:29.384Z
commit_sha: c2d3306be47fed31363004d27a94d21d3b642fbb
---
# Data foundation and governance

Todo lo que el modelo come: schema universal, registry con licencias y
hashes, adapters P0, hard negatives, datos sintéticos/teachers y gold
set humano con español. Sin esto, el resto memoriza artefactos.
El registro de fuentes y licencias vigente vive en
`.meshkore/docs/source-register.md`.

Primer tramo de la única ejecución V1. Esta iniciativa debe seleccionarse
junto con las otras cuatro en **Run All**: sus cuatro gates alimentan la
cadena serial completa, no son una cola aislada de cloud.

Cubre plan §§27–49, 116–118.

## Done when

- Todo dataset de entreno pasa por schema + registry + license fence.
- Adapters P0 (HuffPost, Banking77, BoolQ, Civil, HelpSteer2) verdes.
- Hard negatives y ejemplos `unknown` generados de forma reproducible.
- Pilot gold de 500 items con acuerdo inter-anotador medido.
- Cada run referencia revisiones y hashes; eval-only no puede cruzar al train.
