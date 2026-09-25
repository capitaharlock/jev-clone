---
id: T-ce-scorer
title: El scorer compartido — un solo modelo lee estado, pregunta y opción
status: active
priority: high
owner: unassigned
category: model
initiative: cross-encoder-pilot
created: 2026-09-25
updated: 2026-09-25
---

# El scorer compartido — un solo modelo lee estado, pregunta y opción

Primera pieza de la hipótesis de recuperación. Un único scorer preentrenado para
relacionar textos puntúa cada candidato leyendo los tres textos **juntos**:

```text
z_i = scorer_compartido(ESTADO, PREGUNTA, RESPUESTA_i)
p   = softmax(z_1 … z_K)
L   = -log p_correcta
```

Todos los candidatos pasan por el mismo modelo y la misma salida escalar. No hay
una neurona por etiqueta del catálogo, y la interacción entre los tokens del
estado, la pregunta y la opción ocurre **dentro** del encoder, no en una cabeza
añadida encima de representaciones congeladas. Sigue siendo un modelo pequeño y
no generativo que devuelve pesos sobre opciones nuevas: el producto no cambia.

Alcance de esta task: la pieza y su medición sin entrenar. K=2–8 como régimen
inicial; K mayores se incluyen después y **siempre medidos**. La CE sobre las
opciones ofrecidas ya es un objetivo válido para el producto descrito: no hace
falta normalizar contra un universo global de etiquetas ajenas a la pregunta para
aprender a elegir entre tres colores.

Puntos de partida a cablear, no a elegir por intuición:

- `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` para ES+EN, evaluado por
  entailment con hipótesis explícitas («La respuesta a esta pregunta es …»).
- `ModernBERT-base-zeroshot-v2.0` como alternativa en inglés — **su mezcla
  publicada incluye BANKING77**, así que no puede presentarse como transferencia
  limpia a ese benchmark; sí puede medirse en la batería privada.
- `GLiClass` queda anotada como siguiente candidato **sólo** si el coste de K
  pasadas es el obstáculo. No se abren diez brazos a la vez.

Implementar el **contrato de contexto comparativo** de `#T-episode-contract`: en
las familias donde la información decisiva vive en las opciones, el scorer
recibe el contexto de candidatos, porque puntuar una opción sin ver las demás
pierde el significado de «mejor».

Corrección pendiente que cae en esta task: el docstring de
`training/python/fullspace_loss.py` afirma insesgadez Horvitz–Thompson para
`sum(exp(z_c)/pi_c)`, y esa identidad exige valores poblacionales fijos respecto
al muestreo. Con `CrossBlock` los logits cambian al cambiar el conjunto, así que
los tests con logits fijos no demuestran nada sobre los logits reales. O se
describe como otro objetivo sin esa garantía, o la pérdida sale del árbol activo.
El piloto no la necesita.

## Verification gate

- Test: permutar los candidatos no cambia las probabilidades realineadas más allá
  de la tolerancia numérica escrita.
- Test: intercambiar los textos de dos candidatos conservando sus IDs intercambia
  sus puntuaciones (el scorer lee texto, no índice).
- Test: con el contexto comparativo activo, cambiar el atributo de un candidato
  **ajeno** cambia la puntuación del candidato evaluado; con el contexto
  desactivado, no.
- El docstring de `fullspace_loss.py` no reclama insesgadez sin justificación.

## Done when

- El scorer puntúa listas de candidatos arbitrarios con CE por pregunta y corre en
  el hardware local por batches.
- Los dos checkpoints de partida cargan y producen puntuaciones **sin entrenar**,
  con el formato de hipótesis documentado y fijado.
- El contrato de contexto comparativo está implementado y probado en las dos
  direcciones.
- La pérdida muestreada de fase 2 deja de afirmar una garantía que no tiene.
