---
id: T-parser-queue
title: Cola continua de parsers y fuentes de la fábrica
status: active
priority: high
owner: architect-master
category: data
initiative: decision-dataset-factory
depends_on:
  - T-world-corpus-engine
created: 2026-10-15
updated: 2026-10-15
---

# Cola continua de parsers y fuentes

Una fuente cada vez, sin pausa: parser, test, entrada en el catálogo y paso
por el worker. Cuando la fuente queda normalizada se toma la siguiente.

## Hecho el 15 de octubre

- Worker continuo `com.decisiondata.worker`: descargas en paralelo, una
  normalización por proceso y latido de estado.
- Parser MediaWiki (páginas del espacio principal, sin redirecciones ni
  marcado) y TSV sin cabecera dentro de ZIP.
- Activadas Wikipedia simple, Wikivoyage en inglés y español, Wikilibros en
  español, GeoNames `cities15000` y la taxonomía de productos de Google.
- Parser `uniprot-dat` para el formato plano de UniProt y alta de Swiss-Prot.
- Alta de Wikilibros en inglés, Wikcionario en inglés y español, Wikinoticias,
  Wikiversidad y Wikiquote en español.
- Cada ciclo del worker corre en un intérprete nuevo: un parser nuevo entra
  en servicio sin reiniciar nada.

## Siguientes en la cola

1. Partir las fuentes grandes por fichero de entrada para que Stack Exchange
   y Wikcionario usen todos los núcleos libres y no solo uno.
2. World Bank: indicadores WDI en CSV.
3. Wikidata: subconjunto de clases de producto por SPARQL a ficheros JSONL.
4. OpenAlex: muestra de `works` por concepto, solo metadatos y resúmenes con
   licencia abierta.
5. Wikipedia en español: troceado por ficheros multistream para no superar los
   5 GB por fuente.
6. PubMed baseline: parser `pubmed-xml` de citas y resúmenes.
7. EUR-Lex y CourtListener: texto legal con licencia pública.
8. OpenStreetMap: extractos regionales pequeños con parser `osm-pbf`.

## Done when

No termina. La tarea sigue activa mientras queden fuentes en `review` con
parser pendiente; cada fuente cerrada se apunta en la sección de hechos.
