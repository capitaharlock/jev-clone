---
id: T-massive-huff
title: MASSIVE completo, HuffPost y cierre P0
status: done
priority: high
owner: unassigned
category: data
initiative: data-training
depends_on:
  - T-bigsrc
created: 2026-09-20
updated: 2026-09-20
---

# MASSIVE completo, HuffPost y cierre P0

Cierra el frente "real labels" (§§26–31, 84, 122–126): MASSIVE completo
(>1 M utterances, 52 lenguas; primera expansión EN+ES+FR+DE+PT+IT, el
resto sólo por decisión, §26), HuffPost completo (~210 k; investigar
por qué el local trae 66.510 — subset, split, adapter o mirror —
y corregir o documentar, §§31, 125), y terminar adapters LogiQA 2.0
y ReClor pendientes (§124; enseñar decisión sin generar explicación).
BoolQ se mantiene con decisión de licencia Share-Alike; Civil Comments
se mantiene pero con cap 100–300 k por ciclo (§§27, 84, 126);
NLI/zero-shot-label y DPO pairs entran sólo como research hasta
auditoría de provenance (§§23–24).

Fuentes: data-training §§23–31, 84–85, 122–126.

## Verification gate

- Requiere PASS de `T-bigsrc`; cada fuente ampliada trae manifiesto
  con rows antes/después del filtro y justificación del delta
  (caso HuffPost 66 k vs 210 k resuelto por escrito).
- Tests: adapter MASSIVE conserva intent + descripciones + idioma y
  pasa round-trip ES↔EN + code-switching (§59); LogiQA/ReClor
  convierten state/question/options/gold sin almacenar explicaciones;
  el cap de Civil se verifica contando su % en un mix de prueba
  (falla si > 15 %, §§65–66); BoolQ marcado con su obligación
  Share-Alike en la rama commercial (§99).
- El gate escribe `artifacts/gates/T-massive-huff/gate.json` con
  `pass: true` y matriz fuente×licencia×split; sin ese artifact no
  arranca `#T-synth-factory`.

## Done when

- MASSIVE completo + HuffPost completo + LogiQA/ReClor cerrados en
  parquet canónico con manifests.
- Informe HuffPost 66 k vs 210 k escrito; Civil con cap aplicado y
  verificado; DPO/zero-shot-label en `research-only/` hasta auditoría.
- Layer A (labels humanas/expertas, §61) congelada como base del mix.
