---
id: T-frontier-data-map
title: Convertir las recetas de datos frontier en experimentos
status: active
priority: high
owner: architect-master
category: data
initiative: decision-dataset-factory
depends_on:
  - T-source-corpus-projections
created: 2026-10-16
updated: 2026-10-16
---

# Recetas de datos frontier

Investigación en
`decision-dataset-factory/docs/frontier-training-data.md`.

## Hallazgos

- Jev: TypeSafe no publica dataset, arquitectura ni recompensa. Su fundador
  afirma que todos los datos son sintéticos y se producen internamente.
- GPT-3: mezcla ponderada de Common Crawl filtrado, WebText2, libros y
  Wikipedia; deduplicación global y control de contaminación.
- GPT-4: web pública, datos licenciados y RLHF; mezcla no publicada.
- DeepSeek-V3: 14,8T tokens con más código, matemáticas y lenguas; cribado,
  deduplicación y preservación de documentos.
- Llama 3: unos 15T tokens públicos. Qwen3: 36T tokens y etapas general, STEM,
  código, razonamiento y contexto largo.
- Hay corpus abiertos de 1T a 30T tokens, pero ocupan varios terabytes o
  centenares de terabytes. Se catalogan; no se descargan enteros en esta
  máquina.

## Cola

1. Señales de calidad, valor educativo, repetición, idioma, PII y licencia.
   Primera pasada hecha: `quality.py` puntúa longitud, proporción alfabética,
   repetición de líneas y densidad de enlaces en un sidecar. No admite filas.
2. Deduplicación exacta y aproximada entre fuentes y splits.
3. Manifiestos de mezcla con pesos por fuente, dominio e idioma.
4. Muestras deterministas y acotadas de FineWeb-Edu, DCLM, Dolma y FineMath.
5. Extracción de PDF con medición de calidad.
6. Laboratorio sintético de estados y muchas preguntas por estado.
7. Cardinalidad variable, permutaciones, abstención y casos ambiguos.
8. Experimentos de calibración con log loss, Brier y label smoothing.
9. Proyecciones separadas para general, código, matemáticas, multilingüe y
   contexto largo.
10. Ablación corta antes de ampliar cualquier fuente o receta.

## Criterio

Cada elemento produce una receta versionada y una comparación en v3 por byte,
token y hora de cómputo. No se atribuye a RLCD ningún método que TypeSafe no
haya publicado.
