---
id: T-bigk-optsets
title: Entrenar con el espacio de etiquetas entero, no con 3-8 opciones
status: active
priority: high
owner: unassigned
category: data
initiative: full-space-training
depends_on:
  - T-option-text
created: 2026-09-23
updated: 2026-09-24
---

# Entrenar con el espacio de etiquetas entero

`artifacts/gates/T-teacher-probe/fullspace.json` mide por primera vez el
régimen del producto: el head sobre las 3 080 filas de test de BANKING77 con
las **77 etiquetas** presentes, que es el régimen en el que el producto se
promete y el único en el que la cifra publicada del profesor (0,924) es
comparable. Da **0,0123 con azar en 0,0130** — 38 aciertos en 3 080.

Con IC 95 % [0,0090, 0,0169] el azar cae dentro: la cifra es **indistinguible
del azar**, que es un fracaso de transferencia demostrado, no un sesgo medido.

El barrido de cardinalidad del mismo artefacto (1 000 filas, gold siempre
presente, distractores muestreados hacia abajo) apoya la hipótesis sin
probarla: K=8 y K=40 dan límite inferior por encima del azar; K=5 y K=20 no.
La lectura es que con K pequeño el head tiene una preferencia débil entre un
puñado de candidatos y no un ranking del espacio — hipótesis de esta task, no
una imposibilidad matemática: nadie ha demostrado que entrenar con K pequeño
sea incapaz de producir un ranking global.

`data/optset.py` fija `k_min=3, k_max=8`. Todo el corpus —1 M de filas— se
generó así, y toda la curva de escalado de fase 1 se midió así. Esta task
cambia esa constante y vuelve a medir, porque es la hipótesis más barata que
explica a la vez el suelo de generalización y la caída con más datos: un
objetivo de 8 opciones no obliga al modelo a construir nada que escale a 77.

Riesgo declarado antes de medir: subir K encarece cada fila —la caché de textos
de etiqueta ahorra forwards del encoder, pero no la atención entre opciones, que
crece ≈ K²— y puede no mover nada — en cuyo caso el NO-GO se escribe igual.

## Done when

- `data/optset.py` admite un régimen de cardinalidad configurable
  (`k_max` hasta el espacio entero, con muestreo de distractores declarado)
  y su suite cubre el caso `K = |espacio|`.
- Un run comparable al brazo `leverstack-d512-prior` (mismo corpus, misma
  seed, mismo objetivo) entrenado con K grande, y su `eval.unseen` al lado
  del brazo K≤8.
- `eval.fullspace` re-ejecutado sobre el nuevo checkpoint: la cifra de
  BANKING77 a 77 vías publicada junto a la vieja.
- El coste publicado: filas/s y memoria a K=8, 41 y 77 con la cabeza actual,
  para que el siguiente run se presupueste con números y no con una promesa.
- `unknown` medido en el régimen nuevo: tasa de abstención a K pequeño y a
  K=|espacio| en el mismo artefacto.
- Veredicto escrito: si la cifra a 77 vías sigue en azar, la cardinalidad
  del objetivo queda descartada como causa y se dice cuál es la siguiente —
  sabiendo que un NO-GO a 62 k limita gasto y no establece causa (R9).

## Re-encuadre 2026-09-24 — ésta es la **vía exacta**, y va primero

Esta task nació como "subir `k_max`" dentro de `#generalization-fix`. La
transición a fase 2 (`.meshkore/docs/fase-2-espacio-completo.md`) la mueve a
`#full-space-training` y le da su sitio real: es el **lado de datos** del
cambio de objetivo — el conjunto de opciones que el corpus entrega, para que el
espacio entero esté disponible cuando el dataset lo permite (banking77 = 77,
huffpost = 41, goemotions = 28) en vez de muestreado a 8.

La redacción anterior decía que «subir K sin cambiar la pérdida sólo encarece
cada fila». **Es falso**, y la revisión externa lo corrigió (hallazgo C):
`cross_entropy(logits, gold)` (`training/python/train_decision.py:1543`)
normaliza sobre todas las columnas que se le den. Si a una fila se le entregan
las 77 etiquetas de su espacio, esa misma llamada **ya normaliza sobre el
espacio entero**, más el logit `unknown`.

De ahí el orden nuevo: esta task es la **primera ablación del objetivo**, más
simple y más barata que `#T-fullspace-objective`, porque no necesita escribir
ninguna fórmula nueva. Lo que sí necesita resolver:

- **`unknown` a cardinalidad completa.** Con el espacio entero presente, una
  fila con gold retirado deja de tener sentido igual que con 8 candidatos: se
  declara qué hace `unknown_fraction` en este régimen y se mide la abstención.
- **El firewall intacto.** Ampliar el conjunto de opciones no puede colar una
  etiqueta de un dataset fenced ni cruzar espacios.
- **El coste real de la cabeza.** La cabeza hace atención entre opciones
  (`model/decision_head.py:105-110`): el coste por fila crece ≈ K² en ese
  bloque, no se amortiza con la caché de textos. Se publica filas/s y memoria a
  K=8/41/77.

`#T-fullspace-objective` es el paso siguiente: la extensión **muestreada** —
negativos in-batch y corrección log-Q— para los espacios que no son enumerables.
Las dos son necesarias; la exacta se mide antes y acota lo que la otra tiene que
demostrar.
