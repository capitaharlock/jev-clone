---
id: T-ce-scorer
title: El scorer compartido — un solo modelo lee estado, pregunta y opción
status: done
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

## Resultado (2026-09-26)

La pieza es `model/ce_scorer.py`; la medición sin entrenar, `eval/ce_nograd.py`.
Artefactos: `artifacts/gates/T-ce-scorer/contract.json` (las tres comprobaciones
del gate, sin pesos) y `.../nograd.json` (los dos checkpoints, con sus tres
comprobaciones repetidas sobre los pesos reales).

Cifra sin entrenar, 28 sondas escritas a mano, cinco familias, ES+EN, K=3,
azar 0,333, CPU:

| Checkpoint | Forzado | IC95 % | Pares contrafactuales | ms/par |
|---|---|---|---|---|
| `minilmv2-l6-mnli-xnli` | 17/28 = 0,607 | 0,424–0,764 | 6/13 | 16,4 |
| `modernbert-zeroshot-v2` | 18/28 = 0,643 | 0,458–0,793 | 7/13 | 55,3 |

**No es la batería** de `#T-battery-dev` (400 casos revisados, aún no existe) y
no sostiene ninguna comparación con el 70 %: con n=28 el intervalo es el dato.
Lo accionable para `#T-ce-finetune` es el desglose, no el punto: extracción
8/8 y descripciones 3/4 con ModernBERT frente a comparación de atributos 1/6 y
prioridades 2/4. Un checkpoint NLI sin ajustar reconoce el hecho explícito del
estado y no resuelve aritmética ni reglas de desempate — exactamente lo que el
plan §5 pedía no dar por hecho. Ninguno de los dos se evaluó sobre BANKING77;
la contaminación de `modernbert-zeroshot-v2` viaja en el registro de pesos.

La corrección de `training/python/fullspace_loss.py` es la opción «otro objetivo
sin esa garantía»: la precondición de valores fijos respecto al muestreo está
escrita, la cabeza publicada la viola (ya lo medía
`test_logits_depend_on_the_candidate_set`), y dos tests nuevos impiden que la
afirmación desnuda vuelva a la spec o al docstring.
