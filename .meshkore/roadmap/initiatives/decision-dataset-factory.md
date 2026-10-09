---
id: decision-dataset-factory
title: Fábrica independiente de datasets de decisión
status: active
owner: architect-master
modules:
  - data
created: 2026-10-03
updated: 2026-10-03
---

# Fábrica independiente de datasets de decisión

El modelo JEv es un consumidor, no el dueño del sistema de datos. La fábrica
vive como proyecto separado y entrega datasets versionados que pueden usar
otros modelos y equipos.

## Resultado

Cada publicación contiene episodios, manifiesto, hashes, procedencia, licencias,
estadísticas, rechazos y una data card. Los workers pueden ejecutarse en Mac,
VPS o nodos con GPU sin compartir el árbol de trabajo.

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
