---
id: T-teacher-labelspaces
title: Espacios de etiquetas generados por el profesor — solo si el eje de datos es GO
status: backlog
priority: medium
owner: unassigned
category: data
initiative: teacher-distill
depends_on:
  - T-labelspace-div
  - T-teacher-probe
created: 2026-09-23
updated: 2026-09-23
---

# Espacios de etiquetas generados por el profesor

Task **condicionada**: no se ejecuta hasta que `#T-labelspace-div` publique su
curva. Si la mezcla de 20 000 mini-taxonomías aplana la pendiente unseen, el eje
de datos es la palanca buena y conviene alimentarlo con espacios de etiquetas de
mucha más calidad que los que produce la fábrica programática. Si no la aplana,
esta task se archiva sin gastar un céntimo de saldo.

Qué pediría al profesor, y esto es lo que la distingue de `#T-gen-schemas`: no
filas, **taxonomías**. Dominios que no están en el corpus (legal, clínico,
logística, soporte industrial, trámites administrativos), cada uno con su
conjunto de etiquetas plausible, sus distractores duros —etiquetas vecinas pero
incorrectas, que es exactamente lo que `#T-optset-sampler` necesita y hoy
aproxima con 3-gramas de carácter— y sus ejemplos sembradores. La fábrica
existente expande cada taxonomía a filas; el profesor solo inventa el espacio.

Reglas heredadas de `#data-foundation` que no se relajan: firewall de benchmarks
(`data/firewall.py`) sobre todo lo generado, detección de fuga contra los cortes
unseen (`data/leakage.py`), registro de procedencia en `source-register.md` con
el modelo y la fecha, y solape 0 con las etiquetas retenidas de
`#T-unseen-labels` verificado antes de mezclar, no después.

## Done when

- Existe el veredicto de `#T-labelspace-div` y esta task se ejecuta o se archiva
  citándolo.
- (Si GO) Hay ≥ 500 taxonomías nuevas generadas por el profesor, cada una con
  etiquetas, distractores duros y semillas, versionadas con su coste.
- (Si GO) El firewall y el detector de fuga pasan sobre el 100 % de lo generado,
  y el solape con los cortes unseen es 0, medido y publicado.
- (Si GO) La curva 62 k / 250 k / 1 M de la mezcla con taxonomías del profesor
  está medida contra la de `#T-labelspace-div` en el mismo JSON.
