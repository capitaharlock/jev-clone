---
title: "Fase 2 del entreno — del régimen de opciones muestreadas al espacio completo"
updated: 2026-09-24
owner: architect-master
status: vigente
---

# Fase 2 — entrenar y medir sobre el espacio de etiquetas completo

> Documento de transición. Cierra la fase de validación (runtime, corpus y
> arquitectura, probados en un régimen de pocas opciones) y abre la fase de
> entreno a cardinalidad completa, con el ajuste del approach de testing que
> hace falta para conducirla.

## 1. Dónde estábamos — fase 1 y para qué servía

La fase 1 tenía un objetivo acotado: levantar el sistema entero end-to-end —
formato, API, motor de inferencia Candle, `encode_state` separado de `decide`,
caché de estado, cuantización, gates, corpus de 1 M de filas limpias con
licencias y firewall de benchmarks — y comprobar que la arquitectura del head
aprende de verdad. Para eso se entrenó y se midió en un régimen barato y
rápido: **3–8 opciones por fila** (`data/optset.py: k_min=3, k_max=8`), con el
backbone **congelado** para que cada iteración cupiera en un Mac.

Ese régimen hizo su trabajo. El sistema está construido y la arquitectura
aprende: dentro del corpus, 0,94–0,98; sobre etiquetas no vistas con K≤8, hasta
0,358 contra un azar de 0,200. Las cuatro palancas medidas (cabeza ×2,
prior-penalty, unfreeze last-n, apilado) y la ganancia Metal ×2,7 salen de ahí.

## 2. La medición que marca el cambio de fase

El plan siempre incluyó una referencia externa. La primera comparación con
protocolo idéntico al del profesor está en
`artifacts/gates/T-teacher-probe/fullspace.json` (2026-09-23): mejor checkpoint
(`leverstack-d512-prior`, 1 M filas) sobre las **3 080 filas de test de
BANKING77 con sus 77 etiquetas presentes** — el régimen exacto en el que el
profesor publica 0,924.

| | accuracy | azar | K |
|---|---:|---:|---:|
| nuestro head | 0,0123 | 0,0130 | 77 |
| profesor (cifra publicada) | 0,924 | 0,0130 | 77 |

El barrido de cardinalidad del mismo artefacto, sobre las mismas filas: la
ventaja sobre azar aguanta hasta K=40 y se disuelve en K=77. Con K=5 damos
0,215 contra 0,200 de azar.

Lectura: lo que el modelo tiene en régimen de pocas opciones es una
**preferencia local**, no un ranking del espacio. Es exactamente lo que el
objetivo de fase 1 pedía, y no es lo que el producto promete. **La fase 1 no
midió mal: midió otra cosa, a propósito, y ahora toca medir la de verdad.**

## 3. Por qué una preferencia local puede quedar por debajo del azar

No es un error de medida. El azar es uniforme y sin sesgo; el head sí tiene
sesgo — hacia etiquetas frecuentes, cortas y léxicamente cercanas al texto —
aprendido sobre un corpus donde 9 taxonomías cubren el 83,1 % de las filas.
Evaluado en un espacio donde ese sesgo ya no correlaciona con el gold, un
sesgo sistemático puntúa por debajo de tirar un dado. No hay suelo que lo
impida, y por eso la cifra a 77 vías es informativa en vez de anecdótica.

## 4. Por qué más datos no mejoraban la métrica

La curva de escalado (unseen 0,239 @250 k → 0,050 @1 M con backbone congelado)
encaja con lo anterior. Con 8 candidatos, acertar sólo exige descartar 7 cosas,
y el camino más corto es el mapa cerrado `texto → etiqueta` de las taxonomías
vistas. Cada fila nueva afila ese mapa, y el mapa no transfiere a un espacio
nuevo. **Escalar datos amplifica el régimen en el que se entrena**: con el
objetivo de fase 2, esa misma curva hay que volver a trazarla desde cero.

## 5. Qué distingue las dos fases

| | fase 1 (validación) | fase 2 (espacio completo) |
|---|---|---|
| encoder | congelado (`backbone.frozen: true`) | entrenable — se vuelve a medir cuánto |
| espacio de salida | 3–8 opciones muestreadas | el espacio de etiquetas completo |
| pérdida | `cross_entropy` sobre esas K | normalizada sobre todo el pool, con log-Q |
| capacidad entrenable | ~2,9–11 M (head) | 68 M–149 M (encoder + head) |
| métrica primaria | unseen con K≤8 | accuracy a cardinalidad completa + azar |

La receta de fase 2 es la de recuperación densa (DPR / E5 / SetFit), que es la
que el profesor aplica de facto: normalizar la pérdida sobre el pool entero de
candidatos. Es asumible porque los textos de etiqueta son cortos y se cachean —
el encoder de opciones corre una vez por etiqueta única del batch, no una por
fila.

## 6. Qué se conserva y qué se vuelve a medir

**Se conserva íntegro:**

- Todo el runtime Rust/Candle, el servidor, la cuantización, la paridad.
- El corpus: 1 M de filas limpias, licencias fenced, firewall de benchmarks.
  Cambia **cómo se muestrean las opciones de cada fila**, no las filas.
- La ganancia Metal ×2,7 (`#T-metal-throughput`): es lo que hace viable
  reentrenar el encoder en local.
- El pointer head **como arquitectura**: es label-free por construcción, que es
  la propiedad correcta del producto.

**Se vuelve a medir bajo el objetivo nuevo** (no son veredictos anulados: son
veredictos de fase 1, válidos en su régimen y no transferibles al de fase 2 —
regla R4):

- El **NO-GO de `#T-unfreeze-backbone`**. Con K≤8 la representación congelada
  ya bastaba, así que la pérdida no tenía presión que transmitir al encoder. La
  pregunta se rehace en `#T-encoder-finetune`, donde sí la tiene.
- El **NO-GO al escalado de `#T-mix-5m`**. Mide escalado en régimen de fase 1.
- La **curva de escalado** completa: registro histórico de fase 1, no guía de
  decisión para fase 2.
- Las cifras `unseen` **0,29 / 0,288**: son con K≤8 y sólo se publican con su K
  al lado, nunca frente a una cifra externa medida a 77 vías.

## 7. El ajuste del approach de testing

El eval de fase 1 compartía régimen con el entreno (K≤8), que es lo coherente
mientras el objetivo es K≤8 — y también la razón por la que la distancia real
sólo aparece al medir a cardinalidad completa. Fase 2 mueve el eval primero,
para que el objetivo no pueda volver a adelantar a la métrica. Seis reglas, de
obligado cumplimiento para cualquier agente que toque entreno o eval:

- **R1 — Se entrena la tarea que se mide.** Si la métrica primaria es
  cardinalidad completa, la pérdida es sobre cardinalidad completa. Cualquier
  divergencia entre régimen de entreno y régimen de eval se declara en el gate.
- **R2 — Toda cifra se publica con su azar al lado** y con `beats_chance`. Una
  accuracy sin su azar no es un resultado, es un número.
- **R3 — Ninguna cifra se compara con el exterior sin el mismo protocolo**:
  mismas filas, misma cardinalidad, mismo split. Si es una cita y no una
  medición, se estampa `same_rows: false`.
- **R4 — Un GO/NO-GO sólo vale dentro del régimen con el que se midió.** Al
  cambiar de régimen, los veredictos anteriores se marcan como no transferibles
  en vez de heredarse.
- **R5 — Antes de escalar datos, demostrar la pendiente en pequeño.** No se
  lanza un run de 1 M para una hipótesis que no se haya visto moverse a 62 k.
- **R6 — Una referencia externa en la mesa desde el primer día** de cada
  work-stream de modelo, aunque sea un baseline trivial.

## 8. Dónde se ejecuta

`#full-space-training` — nace de este documento y no ajusta hiperparámetros:
cambia el objetivo de entreno, reabre el fine-tune del encoder y trae los
espacios de etiquetas diversos. `#honest-eval` recibe `#T-eval-cardinality`:
cardinalidad completa como métrica primaria, `beats_chance` obligatorio en
`eval/gate_rules.py` y un solo scoreboard en lugar de tres cortes que hay que
cruzar a mano.

Riesgo declarado antes de medir: si con el objetivo nuevo la cifra a 77 vías
sigue pegada al azar, la hipótesis del objetivo queda refutada y el siguiente
sospechoso es la capacidad del backbone (68 M → 149 M+), que se escribirá como
NO-GO igual de explícito.
