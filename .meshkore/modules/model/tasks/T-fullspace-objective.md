---
id: T-fullspace-objective
title: La pérdida sobre el espacio de etiquetas entero, no sobre 8 opciones
status: active
priority: high
owner: unassigned
category: model
initiative: full-space-training
created: 2026-09-24
updated: 2026-09-24
---

# La pérdida sobre el espacio entero

Task central de `#full-space-training` y prerrequisito de las demás. Hoy
`training/python/train_decision.py` hace `cross_entropy(logits, gold)` sobre
los K∈[3,8] candidatos que `data/optset.py` muestreó para esa fila. El modelo
nunca ve, en la misma normalización, una etiqueta que no sea una de esas ocho.

## Qué se construye

**Negativos in-batch con corrección de muestreo.** El pointer head ya puntúa
por contenido (`score(query, key(texto_opción))`), así que nada impide puntuar
una fila contra los textos de etiqueta de **todo el batch**, no sólo contra los
suyos. El denominador del softmax pasa de 8 a `|etiquetas únicas del batch|`
(cientos), que es una aproximación muestreada del espacio real; la corrección
`log Q` —restar el log de la probabilidad de muestreo de cada negativo— evita
que las etiquetas frecuentes se penalicen de más, que es el sesgo que hoy
medimos como "por debajo del azar".

**Caché de claves de etiqueta.** El coste no debe crecer con K: cada texto de
etiqueta único del batch se codifica **una vez** y su clave se reutiliza para
todas las filas que la tengan como candidata. Sin esto el batch grande no cabe
en MPS; con esto el coste extra es el del producto escalar, no el del encoder.
La ruta batched de `#T-metal-throughput` es el punto de enganche.

**Espacio completo cuando existe.** Si la fila viene de un dataset cuyo espacio
de etiquetas es enumerable y cabe (banking77 = 77, huffpost = 41, goemotions =
28), se usa **entero**, no muestreado. `--fullspace-mode {batch,space,both}`.

**`unknown` bajo el régimen nuevo.** El logit de abstención es hoy un escalar
comparable con 8 rivales; con cientos, su calibración cambia por construcción.
Se mide y se reporta la tasa de abstención en ambos regímenes, o el arreglo de
la accuracy se paga en abstención sin que nadie lo vea.

## Protocolo de medida

Un brazo a **62 k** primero (regla R5: nada de 1 M para una hipótesis no vista
moverse en pequeño), misma seed y mismo corpus que `leverstack-d512-prior`,
única variable el denominador de la pérdida. Si la cifra `eval.fullspace` a 77
vías no se mueve del azar a 62 k, se escribe el NO-GO y no se lanza el 1 M.

## Done when

- El trainer acepta un régimen de pérdida sobre el espacio completo
  (in-batch + log-Q + espacio enumerado cuando exista), con tests que
  verifican la corrección de muestreo y la invariancia al orden de opciones.
- El coste por fila NO crece linealmente con el número de candidatos: la caché
  de claves de etiqueta está medida y publicada en el gate (filas/s vs K).
- Un brazo a 62 k comparable con `leverstack-d512-prior`, con `eval.fullspace`
  a 77 vías y `eval.unseen` publicados al lado de los del brazo viejo.
- El gate declara el régimen de cardinalidad del entreno y la tasa de
  abstención en ambos regímenes.
- Veredicto escrito: GO (la cifra a 77 vías bate el azar con margen y se
  escala) o NO-GO (se nombra el siguiente sospechoso).
