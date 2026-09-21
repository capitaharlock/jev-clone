---
id: T-pointer-head
title: Pointer head — la opción se puntúa por su texto, nunca por su índice
status: active
priority: high
owner: unassigned
category: model
initiative: decision-rebuild
depends_on:
  - T-torch-stack
created: 2026-09-21
updated: 2026-09-21
---

# Pointer head — la opción se puntúa por su texto, nunca por su índice

**El núcleo de toda la corrección.** Hallazgo B: `train_baseline.py::load()`
hace `X = r["state"]`, `y = q["answer"]` con espacio de etiquetas fijo y
global (`clf.classes_`). El conjunto de opciones de cada fila no entra nunca
en el modelo. Por eso ReClor da 0,254 —azar exacto con 4 opciones— y LogiQA
0,435: sus opciones son ids posicionales `opt0..opt3` cuyo texto el modelo no
ve jamás. **No es un fallo de conversión**: los converters preservan
correctamente el texto de cada opción; es el trainer el que lo tira.

Arquitectura a implementar (la del plan maestro §), en
`model/decision_head.py`:

- `encode_state(state) -> H_s` una sola vez por fila, con el backbone de
  `#T-torch-stack`. El coste del estado se paga una vez aunque haya K
  opciones y varias preguntas.
- 1-3 capas de **cross-attention** pregunta/opción contra `H_s` — la memoria
  del estado.
- **Pointer scorer**: `score_k = f(embedding_texto(opción_k), contexto)`.
  Un logit por opción, producido a partir del **texto** de la opción. No
  existe ninguna matriz `num_labels × d`: el espacio de salida es el
  conjunto de opciones de esa fila, no un vocabulario fijo.
- `softmax` por pregunta sobre los K logits + una salida `unknown` explícita
  (logit aprendido, no umbral post-hoc).
- Invariancia al orden: permutar las opciones no cambia el argmax ni la
  distribución más allá de tolerancia numérica. `model/option_mixer.py` ya
  planteaba esto con proyecciones aleatorias en Python puro; aquí se
  implementa de verdad y ese fichero se reemplaza o se reduce a su test.

Contrato con el resto: el head acepta **K variable** por fila y **etiquetas
nunca vistas en entrenamiento**. Esa es la propiedad que define el producto
y la que `#I-honest-eval` va a medir.

## Verification gate

- Test de invariancia: 20 permutaciones de las mismas opciones → mismo
  argmax, y KL entre distribuciones < 1e-4.
- Test de K variable: la misma fila con K=2, 4 y 9 produce salidas válidas
  sin recompilar ni repadear a un máximo fijo.
- Test de etiqueta no vista: con un head inicializado, una opción cuyo texto
  no aparece en ningún ejemplo de entrenamiento recibe un score finito y
  ordenable (la capacidad se mide tras `#T-train-real`).
- Test de `unknown`: una fila cuyo estado no contiene la respuesta eleva el
  logit de `unknown` por encima de todas las opciones tras entrenar.
- El gate escribe `artifacts/gates/T-pointer-head/gate.json` con
  `pass: true`, nº de parámetros del head y latencia por fila con K=4.

## Done when

- `model/decision_head.py` existe, con encode-once + cross-attention +
  pointer scorer + `unknown`, y ninguna capa de tamaño `num_labels`.
- Los cuatro tests (invariancia, K variable, etiqueta no vista, `unknown`)
  pasan en CI.
- Está escrito en el model card qué significa "el modelo puntúa por texto" y
  qué deja de ser posible (ya no hay `clf.classes_` que consultar).
