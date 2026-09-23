---
id: T-bigk-optsets
title: Entrenar con el espacio de etiquetas entero, no con 3-8 opciones
status: next
priority: high
owner: unassigned
category: data
initiative: generalization-fix
created: 2026-09-23
updated: 2026-09-23
---

# Entrenar con el espacio de etiquetas entero

`artifacts/gates/T-teacher-probe/fullspace.json` mide lo que ninguna gate de
este repo medía: el head sobre las 3 080 filas de test de BANKING77 con las
**77 etiquetas** presentes, que es el régimen en el que el producto se
promete y el único en el que la cifra publicada del profesor (0,924) es
comparable. Da **0,0123 con azar en 0,0130** — por debajo del azar.

El barrido de cardinalidad del mismo artefacto descarta la explicación
cómoda. Sobre las MISMAS filas, con el gold siempre presente y los
distractores muestreados hacia abajo, la ventaja sobre azar aguanta hasta
K=40 y se disuelve en K=77. No es que el head no haya visto K grande: es que
lo que tiene con K pequeño es una preferencia débil entre un puñado de
candidatos, no un ranking del espacio.

`data/optset.py` fija `k_min=3, k_max=8`. Todo el corpus —1 M de filas— se
generó así, y toda la curva anti-escalado se midió así. Esta task cambia esa
constante y vuelve a medir, porque es la hipótesis más barata que explica a
la vez el suelo de generalización y la caída con más datos: un objetivo de
8 opciones no obliga al modelo a construir nada que escale a 77.

Riesgo declarado antes de medir: subir K encarece cada fila (el coste de
embeber opciones crece con K, aunque la caché de textos de etiqueta lo
amortigua) y puede no mover nada — en cuyo caso el NO-GO se escribe igual.

## Done when

- `data/optset.py` admite un régimen de cardinalidad configurable
  (`k_max` hasta el espacio entero, con muestreo de distractores declarado)
  y su suite cubre el caso `K = |espacio|`.
- Un run comparable al brazo `leverstack-d512-prior` (mismo corpus, misma
  seed, mismo objetivo) entrenado con K grande, y su `eval.unseen` al lado
  del brazo K≤8.
- `eval.fullspace` re-ejecutado sobre el nuevo checkpoint: la cifra de
  BANKING77 a 77 vías publicada junto a la vieja.
- Veredicto escrito: si la cifra a 77 vías sigue en azar, la cardinalidad
  del objetivo queda descartada como causa y se dice cuál es la siguiente.
