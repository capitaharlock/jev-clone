---
id: T-loop-rl-jev
title: Recompensa del profesor — RLCD estilo Laya con las probabilidades de Jev, sólo si gana en dev
status: backlog
priority: medium
owner: unassigned
category: model
initiative: daily-learning-loop
depends_on:
  - T-laya-objective
  - T-jev-soft-targets
created: 2026-09-27
updated: 2026-09-27
---

# Recompensa del profesor — RLCD estilo Laya con las probabilidades de Jev, sólo si gana en dev

El operador quiere usar la cuenta de Jev para aprendizaje por refuerzo. La
forma concreta y barata es la de Laya (`TMP/laya/laya/common.py::proper_reward`
y su cuaderno): regla de puntuación estrictamente propia (log + 0,5·esférica;
RPS en ordinales) contra la distribución del profesor, gradiente de política
tipo GRPO con G=4 muestras de logits perturbados y σ 0,4 → 0,1, y **CE suave
con peso 1,0 encima**. `#T-laya-objective` lo mide primero en un entreno corto
contra nuestro CE listwise; `#T-jev-soft-targets` aporta la distribución.

Aquí sólo se integra en el bucle **si `#T-laya-objective` da GO**: brazo
`--objective rlcd` en `ce_finetune.py`, mismas filas, misma semilla, y la
regla de promoción de `#T-loop-nightly` decide como con cualquier otro
cambio. Coste del profesor publicado por noche. Si no gana en dev con IC, se
archiva con la cifra.
