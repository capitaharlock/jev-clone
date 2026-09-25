---
closed_at: 2026-09-25
superseded_by: cross-encoder-pilot
id: full-space-training
title: Entrenar la tarea que se mide — espacio de etiquetas completo
status: superseded
owner: architect-master
modules:
  - model
  - data
created: 2026-09-24
updated: 2026-09-25
---

> **Superada el 2026-09-25 por `#cross-encoder-pilot`.** Su tesis —que el fallo
> venía del denominador (3–8 opciones muestreadas en vez del espacio entero)—
> está falsificada por su propio gate: `#T-bigk-optsets` da 9/1000 en brazo y
> en control, y `#T-fullspace-objective` cierra en 0/1000 con 87,7 % de
> abstención (9/1000 forzando elección) contra un azar de 12,99/1000 a K=77.
> El reanálisis del operador (`.meshkore/docs/plan-recuperacion-2026-09-24.md`,
> §§2, 9) añade que (a) cambiar el denominador no obliga al modelo a usar
> semántica y (b) la insesgadez Horvitz–Thompson que `fullspace_loss.py`
> reclama no está justificada cuando los logits dependen del conjunto.
> Lo que sobrevive: los gates como evidencia histórica y el protocolo de
> cardinalidad completa, que pasa a `#honest-eval`.

# Entrenar la tarea que se mide — espacio de etiquetas completo

Segunda fase del entreno, documentada en
`.meshkore/docs/fase-2-espacio-completo.md` (2026-09-24). La fase 1 validó el
sistema end-to-end con una cabeza pointer sobre backbone **congelado** y la
pérdida sobre **3–8 opciones muestreadas** — un régimen barato, elegido para
que cada iteración cupiera en un Mac, que cumplió su función. La primera
comparación con el protocolo del profesor marca el salto: sobre las 3 080 filas
de test de BANKING77 a 77 vías damos **0,0123 con azar 0,0130** e IC 95 %
[0,0090, 0,0169] — indistinguible del azar. La hipótesis de trabajo es que con
pocas opciones el modelo aprende una preferencia local y no un ranking del
espacio, que es lo que el producto promete.

Esta iniciativa **no ajusta hiperparámetros**. Cambia el objetivo de entreno
para que sea el mismo que la métrica.

## La teoría, escrita antes de medir

Entrenar con K candidatos enseña a discriminar dentro de una vecindad de
tamaño K, no a rankear un espacio. Con K=8 y distractores muestreados, acertar
sólo exige descartar 7 cosas; el camino más corto hacia eso es memorizar el
mapa cerrado `texto → etiqueta` de las 9 taxonomías que cubren el 83,1 % del
corpus — y ese mapa es justo lo que no transfiere. Por eso la accuracy *dentro*
del corpus es 0,94–0,98 y la *unseen* cae al escalar: cada fila nueva afila ese
mapa cerrado.

La receta que sí produce un ranking del espacio es la de recuperación densa
(DPR / E5 / SetFit) y es la que Jev aplica de facto: **normalizar la pérdida
sobre todo el pool de candidatos disponible**, no sobre una muestra de 8. En la
práctica eso son tres cambios, y los tres son necesarios juntos:

1. **Pérdida sobre el espacio entero**, en dos pasos y por este orden: la
   **vía exacta** —entregar el espacio enumerable completo, que la
   `cross_entropy` actual ya normaliza sobre las columnas que recibe
   (`#T-bigk-optsets`)— y después la **vía muestreada** —negativos in-batch con
   corrección log-Q para los espacios que no caben (`#T-fullspace-objective`)—.
   La caché de textos de etiqueta ahorra forwards del encoder, pero no la
   atención entre opciones de la cabeza, que crece ≈ K² por fila: el coste se
   mide antes de comprometer un run.
2. **Encoder entrenable** — con K=8 la representación congelada ya basta para
   acertar, así que la pérdida no tenía presión que transmitir al encoder y
   `#T-unfreeze-backbone` midió NO-GO en ese régimen. Bajo una pérdida sobre el
   espacio entero la presión cae sobre la representación, que es donde el
   profesor pone toda su capacidad. El veredicto de fase 1 es válido en su
   régimen y **no transferible** a éste (regla R4): se vuelve a medir.
3. **Espacios de etiquetas diversos** — un objetivo de espacio completo sobre
   9 taxonomías aprende 9 espacios muy bien. La diversidad deja de ser una
   hipótesis lateral (`#T-labelspace-div`) y pasa a ser materia prima del
   objetivo.

Riesgo declarado antes de medir: si con los tres cambios la cifra a 77 vías
sigue pegada al azar, la hipótesis del objetivo queda refutada y el siguiente
sospechoso es la capacidad del backbone (68 M congelado → 149 M+ entrenable),
que se escribe como NO-GO igual de explícito.

## Revisión externa antes del primer run

El plan pasó por una auditoría externa independiente el 2026-09-24, con sus
hallazgos verificados uno a uno contra el código
(`.meshkore/docs/revision-externa-2026-09-24.md`). Tres cosas cambiaron:

- **La cifra es indistinguible del azar, no inferior**: el IC lo contiene. Lo
  demostrado es el fracaso de transferencia, no un sesgo.
- **El orden**: la vía exacta va antes que la muestreada, y antes que ambas va
  `#T-option-text` — al profesor se le dan definiciones de las 77 categorías y
  24 ejemplos etiquetados, a nuestro modelo identificadores crudos
  (`card_arrival`). Medirlo no cuesta un entreno y puede reordenar las
  hipótesis.
- **Lo que no está aislado**: denominador, texto de las opciones, atención entre
  opciones de la cabeza, encoder congelado, diversidad y capacidad del backbone
  siguen confundidos. Ningún NO-GO barato establece causa (regla R9).

Orden de ejecución resultante: `#T-eval-cardinality` → `#T-option-text` →
`#T-bigk-optsets` → `#T-fullspace-objective` → `#T-encoder-finetune` →
diversidad → `#T-jev-parity`. Los hallazgos de runtime (servidor sin motor,
cargador sin encoder afinado, colisión de caché V1) están en `#T-serve-engine`,
fuera de esta iniciativa porque no tocan la capacidad del modelo.

## Relación con lo que ya existe

- `#generalization-fix` buscaba *por qué* la curva baja, tocando modelo y
  mezcla **dentro** del régimen de fase 1. Sus cuatro brazos medidos se
  conservan como registro de esa fase; ninguno de sus veredictos se hereda aquí.
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
  veredicto propio que releva al de fase 1 (`#T-unfreeze-backbone`).
- La métrica primaria del repo es la cardinalidad completa: ningún gate puede
  publicar una accuracy sin su azar, su K y su IC, y los brazos se eligen sobre
  un corte de desarrollo con el test final reservado y sus consultas
  registradas.
- La diferencia de información con el profesor —texto de las opciones y
  ejemplos en el `STATE`— está medida como número, no descrita como nota.
- Hay un veredicto escrito: o la cifra a 77 vías supera el azar con margen y se
  abre la curva de escalado sobre el objetivo de fase 2, o se declara refutada
  la hipótesis del objetivo y se nombra la siguiente.
