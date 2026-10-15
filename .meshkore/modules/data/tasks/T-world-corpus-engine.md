---
id: T-world-corpus-engine
title: Catalogar y adquirir fuentes mundiales bajo cuotas y licencias
status: active
priority: high
owner: architect-master
category: data
initiative: decision-sota
depends_on:
  - T-real-sources
created: 2026-10-15
updated: 2026-10-15
---

# Motor de corpus mundial

La implementación vive en el repositorio independiente
`decision-dataset-factory`.

## Criterios

- Árbol estable de áreas, ampliable con taxonomías públicas.
- Cada fuente registra licencia, uso, parser, cadencia y fecha de revisión.
- Descarga reanudable con 150 GB de reserva libre, 5 GB por fuente y 250 GB de
  techo para el corpus gestionado.
- Normalización por streaming a JSONL comprimido, sin cargar el corpus en
  memoria.
- Ninguna fuente entra en entrenamiento por haber sido descargada.

## Estado

- 35 fuentes, 51 ramas cubiertas.
- 10 fuentes locales, 6,43 GB inventariados.
- 134.802 registros de texto normalizados en 25,5 MB.
- Pendientes los parsers y las decisiones de licencia de las fuentes marcadas
  `review`.

## Done when

- Todas las fuentes activas tienen snapshot, hash, inventario y parser.
- Cada rama tiene oferta de evaluación y al menos una fuente entrenable.
- La admisión al entrenamiento exige licencia, deduplicación, firewall y
  ganancia medida en v3.
