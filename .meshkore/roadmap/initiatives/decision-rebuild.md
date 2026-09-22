---
id: decision-rebuild
title: Motor de decisión real (pointer head sobre el texto de la opción)
status: done
owner: architect-master
modules:
  - model
  - data
created: 2026-09-21
updated: 2026-09-21
completed_at: 2026-09-21T13:57:43.961Z
commit_sha: 855a445667086b2835afc5e8af28a6fe87055e85
---
# Motor de decisión real (pointer head sobre el texto de la opción)

Work-stream de corrección abierto por la auditoría
`.meshkore/docs/audit-2026-09-21.md`. Hallazgos A, B, C: lo que lleva 24 h
entrenando es TF-IDF + LogisticRegression por dataset, con espacio de
etiquetas fijo y global; `train_baseline.py::load()` usa `X = r["state"]`,
`y = q["answer"]` y **tira la pregunta y el conjunto de opciones**. Por eso
ReClor da 0,254 (azar exacto) y LogiQA 0,435 (prior posicional de la clave):
el modelo nunca ve el *texto* de `opt0..opt3`. El 1,0000 del dashboard es
contaminación: 258 700 filas synth-loop salidas de 190 esqueletos con split
`i % 10`.

Esta iniciativa sustituye ese motor por el del plan: `encode_state` una vez →
cross-attention pregunta/opción → memoria del estado → **pointer scorer sobre
el embedding del texto de cada opción** → softmax por pregunta + salida
`unknown`. La propiedad que define el producto —puntuar opciones dinámicas
nunca vistas— es intratable sin esto, y ningún otro frente (más datos, más
gates, más horas de loop) mueve la aguja hasta que exista.

Se conserva intacto lo que la auditoría dio por bueno: `data/schema.py`, los
converters `convert_*.py`, `data/firewall.py` + `data/leakage.py` y
`crates/jev-runtime`.

Orden: `#T-halt-contam` y `#T-torch-stack` son paralelos y desbloquean todo;
`#T-pointer-head` es el núcleo.

## Done when

- El baseline TF-IDF está retirado de la ruta de producto y sustituido por un
  modelo neuronal que puntúa cada opción por su texto.
- Una opción **cuya etiqueta no aparece en entrenamiento** se puntúa por
  encima del azar en held-out (criterio medido en `#I-honest-eval`).
- El bake-off publica números de pesos reales descargados, sin proxies
  aleatorios en el Pareto.
- Ningún split ni dashboard verde se apoya ya en `synth-loop.jsonl`.
