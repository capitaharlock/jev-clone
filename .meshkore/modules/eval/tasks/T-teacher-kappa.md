---
id: T-teacher-kappa
title: Kappa contra el profesor — la evidencia que el release gate da por ausente
status: next
priority: high
owner: unassigned
category: eval
initiative: teacher-distill
depends_on:
  - T-teacher-probe
created: 2026-09-23
updated: 2026-09-23
---

# Kappa contra el profesor

`#T-release-gate` falla 9 de 12 criterios. Uno de ellos,
`teacher_cohen_kappa_min = 0,60`, no falla por desacuerdo: falla porque *no
existe la medición*. La regla de §2 del criterio es deliberada —evidencia
ausente = criterio fallado, nunca omitido— y esta task es la que la satisface.

Se mide **kappa**, no `exact_agree_rate`, y el motivo está escrito en el propio
criterio: en `artifacts/gates/T-release/release.json` convivían un acuerdo exacto
de 0,852 con kappa 0,0 — todo el acuerdo era el del desbalanceo de clases, cero
por encima del azar. Un gate que publique acuerdo exacto sin kappa está
repitiendo ese error.

Reutiliza las predicciones del profesor que `#T-teacher-probe` ya dejó en caché:
esta task no debería gastar saldo nuevo salvo que amplíe la muestra.

Lectura esperada, y hay que escribirla antes de medir para no justificarla
después: con nuestra accuracy unseen en 0,24-0,29 y la del profesor
presumiblemente mucho más alta, kappa saldrá **muy por debajo de 0,60**. Eso no
invalida la task — la convierte en la medición que dice si destilar de este
profesor añadiría señal o ruido, que es la decisión que bloquea cualquier plan
de destilación.

## Done when

- `artifacts/gates/T-teacher-kappa/gate.json` publica Cohen's kappa sobre el
  corte unseen, con matriz de confusión agregada, n, IC95 y `model_version`.
- El gate publica también `exact_agree_rate` **al lado** de kappa, con la nota de
  por qué el primero solo no decide.
- `#T-release-gate` deja de listar `teacher_cohen_kappa_min` en
  `absent_evidence`: pasa a tener un número, pase o falle.
- Queda escrito el veredicto de destilación: si kappa < 0,40, destilar de este
  profesor con el modelo actual es añadir ruido y no se abre esa vía.
