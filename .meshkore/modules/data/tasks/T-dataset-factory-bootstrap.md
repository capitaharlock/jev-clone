---
id: T-dataset-factory-bootstrap
title: Arrancar la fábrica independiente de datasets
status: active
priority: critical
owner: unassigned
category: data
initiative: decision-dataset-factory
depends_on: []
created: 2026-10-03
updated: 2026-10-03
---

# Arrancar la fábrica independiente de datasets

Crear un proyecto hermano que no importe código de `jev.clone`. Su salida es
un dataset de decisión publicable, no un checkpoint.

## Alcance inicial

- Contrato `decision-dataset-v1`.
- Validación de JSONL y generación de manifests.
- CLI instalable sin servicios externos.
- Linaje, licencia, evidencia, gold y grupo de variantes obligatorios.
- Plan de workers distribuidos para Mac y VPS.
- Tests del contrato y del manifest.

## Criterio de cierre

Un tercero puede instalar el paquete, validar un shard, construir su manifest
y entender cómo añadir un conector sin conocer `jev.clone`.
