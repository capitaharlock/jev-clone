---
id: generalization-fix
title: La generalización a etiquetas nuevas (el fallo que bloquea el release)
status: active
owner: architect-master
modules:
  - model
created: 2026-09-22
updated: 2026-09-22
---

# La generalización a etiquetas nuevas (el fallo que bloquea el release)

El modelo existe y está entrenado, pero **no hace lo que define al producto**.
`artifacts/gates/T-release-gate/gate.json` da **NO-GO con 10 de 12 criterios
fallados**: accuracy sobre etiquetas no vistas **0,065** frente a un umbral de
0,50 y un azar de **0,165** — es decir, por debajo del azar.

Lo grave no es el nivel, es la **pendiente**. En la curva de
`artifacts/gates/T-mix-5m/gate.json`, con el mismo corpus, el mismo trainer y
los mismos evals, la accuracy unseen **cae** al añadir datos:

| filas | unseen acc (ettin-68m) | unseen acc (modernbert-base) |
|------:|------:|------:|
| 62 k | 0,141 | 0,080 |
| 250 k | 0,239 | 0,194 |
| 500 k | 0,108 | 0,132 |
| 1 M | 0,050 | 0,028 |

Mientras tanto la accuracy *dentro* del corpus es alta (civil-comments 0,94,
dbpedia14 0,98): el sistema **memoriza el espacio de etiquetas visto** y ese
mapa se afila con cada fila nueva, a costa de la propiedad que queremos. Por
eso `#T-mix-5m` registró NO-GO al escalado: más datos empeoran la métrica.

Sospecha principal, a confirmar por medición y no por argumento: el backbone
está **congelado** (`backbone.frozen: true`, ModernBERT-base 149 M) y toda la
capacidad entrenable son ~2,9 M de cabeza pointer (d=256, 2 capas). Una cabeza
pequeña sobre representaciones fijas es exactamente la arquitectura que aprende
un `texto → etiqueta` cerrado en vez de una comparación pregunta↔opción.

Esta iniciativa es el prerrequisito de `#oss-release` y de `#T-release-gate`:
sin ella publicaríamos un modelo por debajo del azar en su propia promesa.

## Done when

- Existe una explicación **medida** (no argumentada) de por qué la accuracy
  unseen cae al escalar, con ablación por eje.
- La curva unseen deja de ser decreciente: a igual corpus, 1 M ≥ 250 k.
- Un checkpoint supera el azar (0,165) en **todos** los cortes unseen y lo
  registra `#T-unseen-labels`.
- `#T-release-gate` puede volver a evaluarse sin que el fallo sea estructural.
