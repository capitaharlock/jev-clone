---
id: T-laya-objective
title: RLCD vs nuestro CE listwise — qué aporta una regla de puntuación propia
status: backlog
priority: medium
owner: unassigned
category: model
initiative: laya-teardown
depends_on:
  - T-laya-baseline
created: 2026-09-27
updated: 2026-09-27
---

# RLCD vs nuestro CE listwise — qué aporta una regla de puntuación propia

Sólo tiene sentido si `#T-laya-baseline` demuestra que Laya nos gana de verdad.
Su objetivo de ajuste no es CE a secas:

- **Recompensa por regla estrictamente propia** (`laya.common::proper_reward`):
  log score + 0,5·spherical, y para `score` un RPS sobre las CDF. La parte
  ordinal se penaliza como ordinal, no como categórica.
- **Gradiente de política estilo GRPO**: G=4 muestras de logits con ruido
  gaussiano de media cero proyectada sobre el símplex, ventaja normalizada por
  grupo, σ que decae 0,4 → 0,1 a lo largo del entreno.
- **CE suave con peso 1,0 encima**, contra objetivos de probabilidad del profesor
  — no one-hot.
- **Temperatura por (tipo de pregunta, nº de opciones)**, ajustada por LBFGS
  sobre un 10 % retirado ANTES de entrenar, y recortada a [0,5, 5].

La pieza que más barato podemos probar por separado es la última: nuestro gate de
calibración ajusta una temperatura global (5,39). Laya publica ECE 0,466 → 0,081
sólo con reajustarla por cubo.

## Done when

- Está medido el efecto de la temperatura por (tipo, K) sobre nuestro checkpoint
  actual, contra la global, en ECE y en `acc@50 %` de cobertura.
- Está implementada `proper_reward` equivalente y comparada contra nuestro CE
  listwise en un entreno corto de la misma semilla y las mismas filas.
- El brazo GRPO está medido o descartado por coste, con la cifra del coste.
- Hay un veredicto: se adopta, se adopta parcialmente o NO-GO, con su causa.
