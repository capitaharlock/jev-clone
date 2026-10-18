---
id: T-real-sources
title: Adquirir fuentes reales de productos, código, respuestas técnicas y literatura
status: active
priority: high
owner: architect-master
category: data
initiative: decision-sota
created: 2026-10-15
updated: 2026-10-15
---

# Fuentes reales

Lista declarada en `decision-dataset-factory/config/benchmark-v3-sources.json`.
La descarga queda en `work/sources-v3/` con manifiesto y SHA-256.

## Restricciones

- huggingface.co y TypeSafe responden 503 desde esta red. Las fuentes se bajan
  de su host de origen: GitHub, archive.org, Gutenberg, UCSD, IBM, Open Food
  Facts.
- Las fuentes no comerciales o de solo investigación quedan en evaluación.
- Nada entra en entrenamiento sin licencia revisada, deduplicación y firewall
  contra v3.

## Done when

- Las diez fuentes declaradas están descargadas o tienen motivo de rechazo.
- Cada familia de datos del plan tiene adaptador con pruebas.
- 300.000 decisiones verificadas listas para la mezcla.
