# Veredicto pre-registrado — #T-fullspace-objective

> Escrito y commiteado **antes** de que exista ningún checkpoint de esta
> task y antes de mirar ninguna cifra suya. El commit que trae este fichero
> precede al arranque del job; eso es la prueba de que la regla de decisión
> no se eligió después de ver el número.
>
> Fecha: 2026-09-24 · agente `developer-copy` (A040)

## Lo que esta task ya NO es

`#T-bigk-optsets` cerró con NO-GO medido (`artifacts/gates/T-bigk-optsets/
gate.json`): a K=|espacio|, 0,0090 con IC [0,0047, 0,0170] contra un azar
de 0,012987 sobre n=1 000 de desarrollo — el intervalo **contiene el azar**,
y el brazo queda además por debajo del control K≤8. Lectura publicada allí:
la **cardinalidad del objetivo queda descartada como causa** (a 250 048
filas, una seed, backbone congelado — R9: eso limita gasto, no establece
causa), y el siguiente sospechoso es la **representación**.

Así que esta task deja de ser la continuación de la hipótesis de
cardinalidad. Lo que entrega es (a) la especificación y el coste que
`#T-encoder-finetune` necesita como dependencia, y (b) la vía para los
espacios **no enumerables**. Los done-when de especificación y medida van
primero, y el brazo de entreno va último y condicionado.

## La medida que decide la FORMA del brazo (ya hecha, antes de pre-registrar)

`artifacts/gates/T-fullspace-objective/batch-composition.json`, sobre la
mezcla real del brazo comparado (`decision-mix-clean-1m`, `--fence-clean`,
mix-seed 20260922, B=64, 400 batches, 12 datasets):

| | como sirve hoy | si los batches fueran mixtos |
|---|---:|---:|
| datasets por batch | **1,0** (100 % de los batches) | 7,75 (mediana 8,5) |
| etiquetas oro únicas | mediana **5** (media 5,6; p10 = 1) | mediana 21 |
| etiquetas ofrecidas únicas | mediana **5** (media 12,2; máx 45) | mediana 90,5 (máx 123) |

Dos consecuencias, y las dos cambian el brazo:

1. **«Cientos de etiquetas por batch» es falso** y se retira. El techo de
   todo el corpus entrenable son 175 textos de etiqueta distintos: ni
   siquiera con batches mixtos hay cientos.
2. Con el cargador de hoy, los negativos in-batch son un **subconjunto
   estricto del propio espacio de la fila** (un batch = un dataset). Un
   brazo así está **dominado**: no puede ser más que una versión ruidosa
   del brazo exacto que `#T-bigk-optsets` ya midió en NO-GO. Correrlo no
   responde nada y gasta el carril.

Por eso el único brazo con contenido es el de **batches mixtos + negativos
in-batch entre espacios + corrección por inclusión + filtro de colisiones**,
y el cambio del cargador es parte de la variable declarada, no un extra
silencioso.

## Qué se compara, si se corre

Par emparejado contra el brazo de la vía exacta, una sola variable
declarada — *el denominador se extiende más allá del espacio de la fila con
negativos in-batch entre espacios, corregidos*:

| | control | brazo |
|---|---|---|
| run | `bigk-fullspace-d512-prior-ettin-68m-s20260922` | `fullspace-sampled-d512-prior-ettin-68m-s20260922` |
| denominador | el espacio entero de la fila + `unknown` | eso **más** las etiquetas in-batch de otros espacios, con `log π` |
| cargador | un dataset por batch | batches mixtos (parte de la variable, declarada) |
| corpus | `decision-mix-clean-1m`, `--fence-clean` | idéntico |
| seed / mix-seed | 20260922 / 20260922 | idénticos |
| arquitectura | ettin-68m congelado, d512, 2 capas, 8 cabezas, B=64 | **idéntica** — `set_attention=True` |
| presupuesto | etapa 62 500 (R5) | la misma |

`set_attention` se queda en `True`: el brazo NO es un cambio de
arquitectura. Si en algún momento se corre con `set_attention=False`, ese
brazo se etiqueta **CAMBIO DE ARQUITECTURA** y ningún veredicto del uno
transfiere al otro (R4, R9).

## La métrica que decide

**Primaria:** accuracy a cardinalidad completa sobre BANKING77 (77
etiquetas, azar 0,012987) en el **corte de desarrollo congelado**
(`eval.cuts` DEV, n=1 000), vía `eval.fullspace`. `beats_chance` = límite
inferior Wilson 95 % > azar (`eval/gate_rules.py`, R2). El corte reservado
(n=3 080) se lee **una sola vez**, sólo si la primaria de desarrollo pasa,
y con su entrada en `artifacts/gates/T-eval-cardinality/test-queries.json`
(R7).

**Diagnósticos, publicados y nunca en el titular:** `eval.unseen` y la
stage eval del trainer, ambas en el régimen K≤8 de fase 1 (divergencia
entreno/eval declarada en `cardinality_regime`, R1); y la **tasa de
abstención en los dos regímenes**, K≤8 y K=|espacio|, sobre las mismas
filas.

## La regla, escrita antes

1. **GO** si el límite inferior Wilson 95 % de la primaria de desarrollo
   supera 0,012987 **y** la estimación puntual supera la del control
   (`bigk-fullspace`, 0,0090) en el mismo corte y presupuesto. Sólo
   entonces se repite el brazo en **otra seed** (R9) y, si aguanta, se lee
   el corte reservado una vez.
2. **NO-GO** si el intervalo de la primaria contiene el azar. Lectura
   obligatoria: **la extensión muestreada del denominador queda descartada
   como causa** del fracaso de transferencia — *a 62 500 filas, una seed,
   backbone congelado, sobre un pool in-batch cuyo techo medido son 175
   textos*. El siguiente sospechoso sigue siendo la **representación**:
   `#T-encoder-finetune` primero, capacidad del backbone (68 M → 149 M+)
   después. No se escribirá «el denominador no es la causa» como hecho
   establecido.
3. **R9, sin excepción:** un NO-GO a este presupuesto limita gasto y no
   establece causa. Ninguna afirmación causal sin repetir el brazo ganador
   en otra seed y a mayor presupuesto.
4. **Trampa de abstención:** si la abstención a K=|espacio| se va a ~1,0 la
   primaria no es interpretable (una cabeza que se abstiene siempre publica
   accuracy 0 al lado de un azar de 0,013) y eso va en el titular en vez de
   publicar el 0 como resultado.
5. **Trampa del estimador:** la pérdida muestreada **subestima** la pérdida
   sobre el espacio entero (Jensen, `log` cóncava — medido en
   `test_the_loss_estimator_underestimates`). Una curva de `train_loss` más
   baja que la del control **no es** una mejora: es el estimador. Sólo la
   primaria a 77 vías decide, y la comparación de pérdidas entre los dos
   brazos se declara no comparable.
6. **No se corre el brazo dominado.** Si por lo que sea el brazo acaba
   ejecutándose con el cargador de un dataset por batch, su resultado no se
   publica como veredicto de esta hipótesis: es el brazo exacto de
   `#T-bigk-optsets` con ruido, y así se dice.

## Riesgo declarado antes de medir

El coste está medido (`cost.json`): la rebanada de la cabeza crece ≈K² y la
caché de textos no amortiza nada de eso. Con batches mixtos el K efectivo
por fila sube de la mediana 5 de hoy a ~90, así que el precio del brazo se
lee en el tramo de cientos de ese artefacto, no en el de K=8. A 62 500
filas es asumible; a 1 M no se compromete nada sin que la pendiente se haya
visto primero (R5).

Y el riesgo de fondo, escrito antes: dado el NO-GO de `#T-bigk-optsets`,
**lo más probable es que este brazo tampoco mueva la cifra**. Se corre
porque cierra la especificación y el coste que `#T-encoder-finetune`
necesita, y porque los espacios no enumerables no tienen otra vía — no
porque se espere que gane.
