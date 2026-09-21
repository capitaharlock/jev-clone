---
id: T-unseen-labels
title: Métrica primaria — accuracy y ECE sobre etiquetas no vistas en train
status: active
priority: high
owner: unassigned
category: eval
initiative: honest-eval
depends_on:
  - T-split-domain
created: 2026-09-21
updated: 2026-09-21
---

# Métrica primaria — accuracy y ECE sobre etiquetas no vistas en train

Hallazgo F: **no existe ninguna métrica sobre etiquetas no vistas en
entrenamiento**, que es exactamente lo que el producto promete. Todo lo que
se ha medido hasta hoy mide memorización de un espacio de etiquetas fijo.

Esta task define y publica la única métrica que decide si el proyecto
funciona.

Protocolo (holdouts por etiqueta, no por fila):

- **HuffPost**: entrenar con 30 de las 41 categorías, evaluar sobre las 11
  restantes. El modelo nunca ha visto el texto de esas 11 etiquetas.
- **MASSIVE**: entrenar con 4 locales, evaluar con 2 — generalización
  translingüe además de por etiqueta.
- **Banking77**: holdout de intents hermanos (los pares que `#T-optset-sampler`
  usa como hard negatives), para separar "generaliza" de "acierta lo fácil".
- **LogiQA y ReClor**: eval-only, jamás en train (`#T-halt-contam`). Son el
  termómetro externo: si suben por encima de 0,25 sin haberlos visto, el
  pointer head está haciendo su trabajo.

Métricas publicadas por cada corte: accuracy, **ECE**, Brier, cobertura a
riesgo fijo (curva risk-coverage con la salida `unknown`), y el intervalo de
confianza. Siempre en pares **seen / unseen**: un número unseen sin su seen
al lado no dice si el modelo generaliza o si el corte era fácil.

`eval/calib.py` se reescribe para calibrar **el modelo**: hoy hace
temperature scaling sobre un scorer coseno char-3gram sin parámetros, que no
es el modelo.

## Verification gate

- Test: ninguna etiqueta del conjunto unseen aparece en ninguna fila de
  train (verificado por texto normalizado, no por id).
- Test: la métrica se calcula sobre el checkpoint de `#T-train-real`, no
  sobre un scorer sin parámetros (el gate falla si el artefacto no lleva
  `model_version`).
- El gate escribe `artifacts/gates/T-unseen-labels/gate.json` con
  `pass: true` y la tabla seen/unseen completa por dataset.

## Done when

- La tabla seen/unseen (accuracy, ECE, Brier, cobertura, IC) se publica por
  cada corte y es la que aparece como titular en el dashboard.
- `eval/calib.py` calibra el modelo real y no un scorer coseno.
- LogiQA/ReClor tienen su número eval-only publicado con su fecha y su
  `model_version`.
