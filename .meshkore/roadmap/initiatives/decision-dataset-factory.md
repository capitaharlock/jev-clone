---
id: decision-dataset-factory
title: Fábrica independiente de datasets de decisión
status: active
owner: architect-master
modules:
  - data
created: 2026-10-03
updated: 2026-10-16
---

# Fábrica independiente de datasets de decisión

El modelo JEv es un consumidor, no el dueño del sistema de datos. La fábrica
vive como proyecto separado y entrega datasets versionados que pueden usar
otros modelos y equipos.

## Resultado

La fábrica conserva primero un corpus fuente neutral con texto, procedencia,
licencia y hashes. Cada tarea publica después una proyección versionada del
corpus. La primera es para modelos de decisión; en el futuro podrán añadirse
proyecciones para lenguaje, recuperación o embeddings sin repetir la descarga
y limpieza.

Cada publicación contiene manifiesto, `corpus_id`, receta, hashes,
procedencia, licencias, estadísticas, rechazos y una data card. Los workers
pueden ejecutarse en Mac, VPS o nodos con GPU sin compartir el árbol de
trabajo.

## Reglas

- Ninguna fuente entra sin procedencia, revisión, hash y decisión de licencia.
- Internet aporta documentos y evidencia. No convierte una respuesta en gold.
- Qwen propone, redacta y ataca. Una regla, programa, fuente humana o acuerdo
  independiente decide el gold.
- Raw, normalizado y publicado son capas distintas e inmutables.
- Los splits se asignan por grupo antes de generar variantes.
- Los benchmarks reservados no entran en prompts, dedupe, selección ni train.

## Primer hito

Proyecto independiente con contrato estable, CLI de validación y manifiestos,
ejecución local, diseño distribuido y un conector de referencia. Después se
extraen los generadores ya medidos de `jev.clone`.
