---
id: T-source-corpus-projections
title: Separar el corpus fuente de los datasets por tarea
status: active
priority: high
owner: architect-master
category: data
initiative: decision-dataset-factory
depends_on:
  - T-world-corpus-engine
created: 2026-10-16
updated: 2026-10-16
---

# Corpus fuente y proyecciones por tarea

La descarga, el parseo, la limpieza y la deduplicación producen un corpus
neutral que se conserva. Los datasets de decisión son proyecciones derivadas,
no el único resultado de la fábrica. Esto permite preparar más adelante datos
para un modelo de lenguaje, recuperación o embeddings sin descargar y limpiar
de nuevo las fuentes.

## Contrato

- `source-document-v1`: texto limpio, ID estable, revisión, taxonomía,
  idioma, licencia, uso permitido, hash y fichero raw de origen.
- `source-corpus-manifest-v1`: hashes de shards y `corpus_id`.
- `dataset-projection-v1`: fija `corpus_id`, receta, tarea, fuentes, salidas y
  hashes.
- `decision-dataset-v1`: primera proyección; añade estado, pregunta,
  candidatos y gold verificable.

## Estado

- Contratos y manifiestos implementados en `decision-dataset-factory`.
- El worker regenera el manifiesto fuente al final de cada ciclo.
- El cambio de esquema invalida de forma intencionada los corpus normalizados
  antiguos para reconstruirlos desde los raw locales, sin descargarlos otra
  vez.
- 41 tests pasan.

## Siguiente

- Hacer que cada arena de decisión conserve `source_document_ids`.
- Versionar recetas de selección, resumen y generación.
- Añadir una proyección para lenguaje solo cuando haya un experimento con
  presupuesto, tokenizer y criterio de admisión definidos.

## Done when

- Toda publicación de decisiones apunta a un `corpus_id` y a una receta.
- Cada episodio derivado de una fuente conserva los IDs de los documentos.
- Se puede regenerar una proyección sin red a partir de raw y corpus local.
