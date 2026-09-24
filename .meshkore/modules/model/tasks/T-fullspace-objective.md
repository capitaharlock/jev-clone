---
id: T-fullspace-objective
title: La pérdida sobre el espacio entero — extensión muestreada con log-Q
status: active
priority: high
owner: unassigned
category: model
initiative: full-space-training
depends_on:
  - T-bigk-optsets
created: 2026-09-24
updated: 2026-09-24
---

# La pérdida sobre el espacio entero — la vía muestreada

Hoy `training/python/train_decision.py` hace `cross_entropy(logits, gold)` sobre
los K∈[3,8] candidatos que `data/optset.py` muestreó para esa fila. El modelo
nunca ve, en la misma normalización, una etiqueta que no sea una de esas ocho.

**Esta task ya no es la primera del cambio de objetivo.** La revisión externa
(`.meshkore/docs/revision-externa-2026-09-24.md`) corrigió una premisa que la
redacción original daba por buena: `cross_entropy` ya normaliza sobre las
columnas que se le den, así que entregar el espacio enumerable completo amplía
el denominador **sin escribir ninguna fórmula nueva**. Esa es la vía exacta y
la tiene `#T-bigk-optsets`. Lo que queda aquí es lo que la vía exacta no puede
hacer: los espacios que **no** son enumerables o no caben.

## Qué se construye

**Negativos in-batch con corrección de muestreo.** Puntuar una fila contra los
textos de etiqueta de más filas del batch, no sólo contra los suyos, con
corrección `log Q` —restar el log de la probabilidad de muestreo de cada
negativo— para que las etiquetas frecuentes no se penalicen de más.

**`unknown` bajo el régimen nuevo.** El logit de abstención es hoy un escalar
comparable con 8 rivales; con cientos, su calibración cambia por construcción.
Se mide y se reporta la tasa de abstención en ambos regímenes, o el arreglo de
la accuracy se paga en abstención sin que nadie lo vea.

## Tres cosas que hay que resolver ANTES de escribir el run

### 1. La cabeza no puntúa opciones de forma independiente

`model/decision_head.py:105-110` aplica `set_attn` entre todas las opciones, y
el logit `unknown` se construye con `ctx.mean(0)` (`:176-182`): el contexto de
cada opción y la abstención dependen del conjunto entero. De ahí, tres
consecuencias que la redacción anterior no recogía:

- Cachear las claves de etiqueta ahorra forwards del encoder, **no** el coste de
  atención por fila, que crece ≈ K². La promesa «el coste por fila no crece con
  K» no se sigue de esta arquitectura y se retira.
- Los logits calculados sobre conjuntos distintos **no se pueden unir** en un
  solo softmax. Negativos in-batch exige pasar el conjunto ampliado por la
  cabeza para cada fila.
- La alternativa —una rama de scoring independiente, `score(query, key(texto))`
  sin atención entre candidatos— es un **cambio de arquitectura** y se evalúa
  como ablación explícita, no como «la misma cabeza con otro denominador».

Se decide y se escribe cuál de las dos, con el coste medido a K=8/41/77 y
cientos (tiempo y memoria), antes de comprometer ningún run.

### 2. La especificación matemática es la puerta de entrada

La redacción anterior mezclaba tres universos de negativos —espacio de la fila,
etiquetas únicas del batch, etiquetas de otros espacios— sin definir qué pasa
con los falsos negativos (una etiqueta ajena puede ser correcta para la fila),
los duplicados, el gold, `unknown` ni el soporte positivo. Y ojo: el
`prior_penalty` actual (`train_decision.py:741-784`) resta un log de frecuencia
empírica, que **no** es una corrección log-Q.

Entregable previo al run: la pérdida escrita con su ecuación y dos modos
verificables.

- **Modo exacto** (espacio enumerado): test de igualdad de valor y de gradiente
  contra el softmax completo.
- **Modo muestreado**: verificar las propiedades del estimador elegido y su
  distribución de propuesta o de inclusión —no son intercambiables— sin exigir
  igualdad para una sola muestra.
- Test de colisión de etiquetas y falsos negativos entre espacios.
- Declarar explícitamente cuándo se omite log-Q y por qué.

### 3. Cuántas etiquetas únicas hay de verdad en un batch

`MixtureStream.epoch()` (`train_decision.py:399-416`) sirve batches de **un solo
dataset**. En una tarea binaria, la unión de etiquetas de un batch de 128 filas
tiene dos candidatos: «cientos de etiquetas por batch» no se sigue del cargador
actual. Se mide y se publica la distribución de etiquetas únicas y de espacios
por batch **antes** de atribuir a B el tamaño del denominador; si hacen falta
batches mixtos, ese cambio del cargador es parte de esta task y se declara.

## Protocolo de medida

Un brazo a **62 k** primero (regla R5), misma seed y mismo corpus que el brazo
de `#T-bigk-optsets` con el que se compara, y una sola variable declarada. Los
brazos se eligen sobre el **corte de desarrollo** de `#T-eval-cardinality`; el
test final se consulta una vez y queda registrado (R7). Si la cifra no se mueve
del azar a 62 k se escribe el NO-GO y no se lanza el 1 M — sabiendo que ese
NO-GO limita gasto y no establece causa (R9): el brazo que sí prometa se repite
en otra seed antes de cualquier veredicto causal.

## Done when

- La pérdida está especificada con ecuación y tests: modo exacto verificado
  contra softmax completo (valor y gradiente), modo muestreado con las
  propiedades de su estimador, colisiones y falsos negativos cubiertos.
- Publicada la distribución de etiquetas únicas y espacios por batch del
  cargador real, y declarado si se cambia a batches mixtos.
- Publicado el coste de la cabeza a K=8/41/77 y cientos —tiempo y memoria— y
  decidido por escrito si el objetivo conserva la cabeza o la cambia; si la
  cambia, el brazo se etiqueta como cambio de arquitectura.
- Un brazo a 62 k comparable, con `eval.fullspace` a 77 vías y `eval.unseen`
  publicados al lado del brazo de la vía exacta, ambos con azar, K e IC.
- El gate declara el régimen de cardinalidad del entreno y la tasa de
  abstención en ambos regímenes.
- Veredicto escrito: GO (la cifra a 77 vías bate el azar con margen, confirmada
  en otra seed, y se escala) o NO-GO (se nombra el siguiente sospechoso).
