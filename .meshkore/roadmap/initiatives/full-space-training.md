---
id: full-space-training
title: Entrenar la tarea que se mide — espacio de etiquetas completo
status: active
owner: architect-master
modules:
  - model
  - data
created: 2026-09-24
updated: 2026-09-24
---

# Entrenar la tarea que se mide — espacio de etiquetas completo

Nace del postmortem `.meshkore/docs/postmortem-objetivo-de-entreno.md`
(2026-09-24). Durante cuatro días se entrenó una cabeza pointer sobre un
backbone **congelado**, con la pérdida sobre **3–8 opciones muestreadas**,
mientras el producto promete —y el profesor publica— puntuar el **espacio de
etiquetas entero**. Sobre las 3 080 filas de test de BANKING77 a 77 vías damos
**0,0123 con azar 0,0130**: por debajo del azar, 1,3 % del profesor.

Esta iniciativa **no ajusta hiperparámetros**. Cambia el objetivo.

## La teoría, escrita antes de medir

Entrenar con K candidatos enseña a discriminar dentro de una vecindad de
tamaño K, no a rankear un espacio. Con K=8 y distractores muestreados, acertar
sólo exige descartar 7 cosas; el camino más corto hacia eso es memorizar el
mapa cerrado `texto → etiqueta` de las 9 taxonomías que cubren el 83,1 % del
corpus — y ese mapa es justo lo que no transfiere. Por eso la accuracy *dentro*
del corpus es 0,94–0,98 y la *unseen* cae al escalar: cada fila nueva afila el
mapa equivocado.

La receta que sí produce un ranking del espacio es la de recuperación densa
(DPR / E5 / SetFit) y es la que Jev aplica de facto: **normalizar la pérdida
sobre todo el pool de candidatos disponible**, no sobre una muestra de 8. En la
práctica eso son tres cambios, y los tres son necesarios juntos:

1. **Pérdida sobre el espacio entero** — negativos in-batch: los textos de
   etiqueta del batch completo (y del espacio de la fila, si cabe) entran como
   candidatos, con corrección log-Q por el muestreo. Coste asumible porque los
   textos de etiqueta son cortos y se cachean: el encoder de opciones se
   ejecuta una vez por etiqueta única del batch, no una vez por fila.
2. **Encoder entrenable** — con K=8 la representación congelada ya basta para
   acertar, así que descongelar no podía ayudar y `#T-unfreeze-backbone` midió
   NO-GO. Bajo una pérdida sobre el espacio entero la presión cae sobre la
   representación, que es donde Jev pone toda su capacidad. El veredicto
   anterior queda declarado **no transferible** (regla R4).
3. **Espacios de etiquetas diversos** — un objetivo de espacio completo sobre
   9 taxonomías aprende 9 espacios muy bien. La diversidad deja de ser una
   hipótesis lateral (`#T-labelspace-div`) y pasa a ser materia prima del
   objetivo.

Riesgo declarado antes de medir: si con los tres cambios la cifra a 77 vías
sigue pegada al azar, la hipótesis del objetivo queda refutada y el siguiente
sospechoso es la capacidad del backbone (68 M congelado → 149 M+ entrenable),
que se escribe como NO-GO igual de explícito.

## Relación con lo que ya existe

- `#generalization-fix` buscaba *por qué* la curva baja, tocando modelo y
  mezcla **dentro** del objetivo viejo. Sus cuatro brazos medidos se conservan
  como registro; ninguno de sus veredictos se hereda aquí.
- `#honest-eval` recibe la reorganización del testing: la cardinalidad completa
  pasa a métrica primaria y K≤8 baja a diagnóstico (`#T-eval-cardinality`).
- `#teacher-distill` aporta la referencia externa. El endpoint del profesor ya
  está identificado (`api.typesafe.ai`, Bearer); la key aportada devuelve 401.
- El corpus **no se rehace**: 1 M de filas limpias, con licencias y firewall,
  siguen siendo válidas. Lo que cambia es cómo se muestrean las opciones.

## Done when

- La pérdida del trainer se puede calcular sobre el espacio de etiquetas
  entero, con su corrección de muestreo declarada, y el gate de cada run
  publica en qué régimen de cardinalidad se entrenó.
- Existe un checkpoint entrenado con el objetivo nuevo y su cifra en
  `eval.fullspace` a 77 vías, publicada al lado de la vieja (0,0123) y de su
  azar.
- El fine-tune del encoder está medido **bajo el objetivo nuevo**, con
  veredicto propio que sustituye al NO-GO de `#T-unfreeze-backbone`.
- La métrica primaria del repo es la cardinalidad completa: ningún gate puede
  publicar una accuracy sin su azar ni sin declarar su K.
- Hay un veredicto escrito: o la cifra a 77 vías supera el azar con margen y se
  abre la curva de escalado sobre el objetivo correcto, o se declara refutada
  la hipótesis del objetivo y se nombra la siguiente.
