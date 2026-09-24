# Decisión escrita sobre la cabeza — #T-fullspace-objective

> Done-when 3: *«publicado el coste de la cabeza a K=8/41/77 y cientos
> —tiempo y memoria— y decidido por escrito si el objetivo conserva la
> cabeza o la cambia; si la cambia, el brazo se etiqueta como cambio de
> arquitectura».*
>
> Fecha: 2026-09-24 · agente `developer-copy` (A040) · artefacto:
> `artifacts/gates/T-fullspace-objective/cost.json`

## Decisión

**Se CONSERVA `model/decision_head.py` con `set_attention=True`.** El brazo
de esta task mueve el DENOMINADOR y no la arquitectura, así que no lleva
etiqueta de cambio de arquitectura.

La alternativa —scoring independiente, sin atención entre candidatos— queda
**implementada, testeada y apagada**: `PointerDecisionHead(...,
set_attention=False)`, con `model/test_decision_head.py::
TestIndependentScoringAblation` fijando las dos mitades de la frase (con el
flag apagado el logit de una opción deja de depender de sus rivales; con el
flag por defecto la cabeza publicada es parámetro por parámetro la misma).
Cuando se corra, se corre **etiquetada CAMBIO DE ARQUITECTURA** y ningún
veredicto transfiere entre las dos (R4, R9).

## El coste medido

K=8/41/77 de la cabeza actual se **reutilizan** de `#T-bigk-optsets`
(misma máquina, mismo arnés); K=154 y K=308 son nuevos, y la ablación se
mide en todo el barrido. Cada medida en su propio subproceso: el pool de
MPS es una marca de agua alta dentro de un proceso, así que medir varios K
seguidos publicaba la memoria del mayor en todos — en la primera corrida de
este arnés salió 4,1 GiB en las cinco filas de la ablación, que era el
artefacto y no la memoria.

| K | cabeza actual (`set_attn`) | | | ablación (independiente) | | |
|---:|---:|---:|---:|---:|---:|---:|
| | filas/s | ms cabeza | pico GiB | filas/s | ms cabeza | pico GiB |
| 8 | 421,2 | 29,9 | 1,10 | 249,9 | 85,5 | 1,09 |
| 41 | 290,1 | 64,3 | 1,11 | 114,1 | 83,7 | 1,07 |
| 77 | 244,8 | 102,7 | 2,11 | 108,8 | 174,3 | 1,08 |
| **154** | **83,4** | **338,1** | **2,09** | **277,1** | **105,1** | **2,09** |
| **308** | **47,8** | **719,2** | **4,12** | **66,5** | **415,2** | **3,11** |

### Qué de esta tabla se puede leer, y qué no

Este Mac **no estaba ocioso**: otro agente trabajaba en paralelo. La nota
de honestidad de `#T-bigk-optsets` ya había medido que un job paralelo
cuesta cerca de un tercio del throughput, y aquí se ve: `noise_check`
detecta que la columna de la ablación **sube** de 108,8 a 277,1 filas/s
entre K=77 y K=154, que es imposible por construcción.

Así que:

- La curva de la **cabeza actual sí es monótona** (`noise_check:
  monotone_in_k: true`) y se lee como curva: 421 → 290 → 245 → 83 → 48
  filas/s. La rebanada de la cabeza sola va 29,9 → 64,3 → 102,7 → 338,1 →
  719,2 ms: de K=77 a K=154 se multiplica por 3,3 al doblar K, que es el
  término ≈K² que la caché de textos de etiqueta no amortiza.
- De la ablación sólo son comparaciones los **pares al mismo K medidos
  seguidos**, y los únicos contemporáneos son K=154 y K=308: ahí la
  ablación da **3,32×** y **1,39×** el throughput, y **0,31×** y **0,58×**
  el tiempo de cabeza. Los ratios de K=8/41/77 comparan contra filas
  medidas **otro día** y no son comparaciones; están en el artefacto
  etiquetados como tales, no en esta decisión.
- La **memoria** sí es limpia (cada medida, su proceso): a K=308 la cabeza
  actual pide 4,12 GiB contra 3,11 GiB de la ablación.

## Por qué se conserva, con este coste sobre la mesa

1. **El K de este brazo no es 308, es ~90.** Medido, no supuesto:
   `batch-composition.json` dice que con batches mixtos la mediana de
   etiquetas ofrecidas únicas por batch es 90,5 (máx 123), y el techo de
   todo el corpus entrenable son 175 textos distintos. A ese K el precio
   está entre las filas de 77 y 154 — del orden de 80-240 filas/s, o sea
   **entre 4 y 13 minutos las 62 500 filas del brazo**. El coste no obliga
   a cambiar nada.
2. **Cambiar el scoring confundiría la única variable declarada.** El brazo
   compara contra `bigk-fullspace` para responder si extender el
   denominador más allá del espacio de la fila mueve la cifra. Con la
   arquitectura movida a la vez, un NO-GO no diría cuál de las dos cosas
   falló, y con `#T-bigk-optsets` recién cerrado en NO-GO eso es
   exactamente el gasto que R9 manda no hacer.
3. **La ablación toca también la abstención, no sólo el coste.** El logit
   `unknown` se construye con `ctx.mean(0)` sobre los contextos de las
   opciones (`decision_head.py:176-182`); sin atención entre candidatos ese
   resumen cambia de significado. Sería un segundo confundido dentro del
   mismo brazo, y la abstención es precisamente uno de los done-when.
4. **Lo que la ablación compra es escala, y la escala no está en esta
   task.** Su ventaja aparece donde K se va a cientos — espacios no
   enumerables, o `#T-encoder-finetune` con el backbone entrenable encima.
   Ahí se corre, con su etiqueta, y contra un control propio.

## El techo que esto deja escrito

Con la cabeza actual, **K=308 son 47,8 filas/s**: por debajo del umbral de
~40 filas/s sólo por un pelo, y 1 M de filas a ese ritmo son ~5,8 horas en
esta máquina. Esa es la frontera a partir de la cual el scoring
independiente deja de ser una ablación opcional y pasa a ser la condición
para que el run exista. `#T-encoder-finetune` hereda esta cifra como
dependencia, junto con la memoria: 4,12 GiB de pico a K=308 **con el
backbone congelado**, antes de sumarle los gradientes del encoder.
