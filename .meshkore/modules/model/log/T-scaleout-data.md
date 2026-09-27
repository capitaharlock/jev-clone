---
id: T-scaleout-data
title: Escala de volumen de datos en x3
status: backlog
priority: low
owner: unassigned
category: model
initiative: train-scaleout
depends_on:
  - T-dist-train
created: 2026-09-20
updated: 2026-09-27
---

> **Archivada el 2026-09-27** por decisión del operador: el único objetivo es un modelo que aprende y mejora cada día (`docs/guia-un-solo-objetivo.md`). No se despacha. Se reactiva sólo si el operador lo pide.

# Escala de volumen de datos en x3

Segunda mitad del motivo de `#train-scaleout`: cuando lleguen muchos más
conjuntos de datos de entrenamiento y verificación, el volumen por run
crece y una sola máquina deja de bastar. Esta task parte del scheduler
de `#T-dist-train` ya verificado y añade el manejo de volumen:
fragmentación y streaming determinista de datasets entre familias A/B/C,
asignación de shards compatibles por máquina, paridad de manifests y
hashes a volumen real, y capacidad del results registry (muchos más runs
y artifacts). El firewall de contaminación (`#T-firewall`) sigue
aplicando: más datos no relaja qué puede ver el entrenamiento.

Fuente: plan §§123–125 (mixed dataset batches, sequence bucketing),
§§42–43 (governance, dataset matrix); stack §34.

## Verification gate

- Requiere PASS de `T-dist-train`; rama opcional, nada obligatorio
  depende de ella.
- Tests: el sharding es determinista (mismo seed+revisión → mismos
  shards en cualquier máquina); un corpus sintético grande verifica
  paridad de manifest+hash entre 1 y 3 workers; el registry ingiere N
  runs sintéticos sin duplicar ni perder artifacts; el fence de
  licencias/leakage se re-ejecuta sobre el corpus ampliado en verde.
- Gate con `artifacts/gates/T-scaleout-data/gate.json` (`pass: true`,
  hashes de config/datos, throughput 1-vs-3 medido); si el volumen
  actual no supera el umbral de `#T-scaleout-gate`, se registra NO-GO y
  la rama queda latente sin romper nada.

## Done when

- Shards deterministas por familia con manifests y hashes verificables.
- Registry probado a volumen sin duplicados ni pérdidas.
- Medición 1-vs-3 publicada; NO-GO documentado si el umbral no se
  alcanza.
