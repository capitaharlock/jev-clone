---
id: T-ce-mechanics
title: Validar la mecánica antes de gastar presupuesto — sobreajustar 32–64 casos
status: blocked
priority: high
owner: unassigned
category: model
initiative: cross-encoder-pilot
depends_on:
  - T-ce-scorer
created: 2026-09-25
updated: 2026-09-25
failed_at: 2026-09-26T11:51:22.469Z
resolved_by: A043
resolved_by_conv: work-cross-encoder-pilot-T-ce-mechanics-1790310000
---
# Validar la mecánica antes de gastar presupuesto — sobreajustar 32–64 casos

Paso 3 del piloto, y el más barato de todos. Antes de generar 5.000 decisiones y
antes de ocupar la GPU durante horas, comprobar que la tubería **puede** aprender:
ajustar entre 32 y 64 ejemplos inequívocos y exigir >95 % sobre ellos mismos.

Esto **no valida generalización** y no se puede citar como si lo hiciera. Valida
que los gradientes llegan, que el gold está donde el trainer cree, que el
tokenizado no corta el estado, que la máscara de candidatos es correcta y que el
formato de hipótesis es el mismo en entreno y en evaluación. Es exactamente la
clase de fallo que la fase 1 tardó semanas en descartar.

Si falla: se arregla la tubería, no se cambia de arquitectura ni se añaden datos.

## Verification gate

- Test: >95 % de acierto sobre los mismos 32–64 ejemplos ajustados, con la semilla
  y el comando registrados.
- Test: el mismo checkpoint sin ajustar sobre esos ejemplos queda claramente por
  debajo — si ya los acierta, el conjunto no es válido como prueba de mecánica.
- El gate deja constancia explícita de que la cifra **no** es una medida de
  generalización.

## Done when

- El sobreajuste supera el 95 % y el artefacto lo registra con su comando
  reproducible.
- Cualquier fallo encontrado en el camino (gold mal alineado, truncado del
  estado, máscara, discrepancia de formato entreno/eval) queda corregido y
  cubierto por un test.
- El formato de hipótesis queda congelado para el resto del piloto.

## Formato de hipótesis CONGELADO

Punto 3 del `Done when`, y vale para el resto del piloto. Vive en UN solo
sitio importable, `model/ce_scorer.py`, y nadie más en el árbol tiene una
plantilla (`test_no_second_hypothesis_template_lives_in_the_tree` recorre
todos los `.py` y exige que el único que la contenga sea ese).

    premisa   = "State: {state}\nQuestion: {question}"        # en
                "Estado: {state}\nPregunta: {question}"        # es
    hipótesis = "The answer to this question is: {option}."     # en
                "La respuesta a esta pregunta es: {option}."    # es

Y con la premisa va, sólo en las familias cuyo contrato lo pide
(`attribute_comparison`, `priority_decision`), el bloque comparativo:
`"Options under comparison:" / "Opciones en comparación:"` seguido de los
K textos ordenados alfabéticamente y sin ids.

Lo demás que el formato fija, porque también decide qué texto ve el modelo:

- reducción a escalar: `entail_logit` (el logit crudo de `entailment`, leído
  del `config.json` del checkpoint, nunca asumido en el índice 0);
- recorte: `truncation="only_first"` con ventana 512 — se recorta la
  PREMISA, nunca la opción que se juzga, y un par que no cabe PARA el run
  en vez de recortar el estado en silencio;
- un solo camino de render y de tokenizado: `ce_scorer.flatten_pairs` y
  `ce_scorer.encode_pairs`. El entreno (`training/python/ce_overfit.py`) y
  la evaluación (`eval/ce_nograd.py`) llaman a esas dos y a ninguna otra.

**Huella: `b215e3003cc60c0f`** (`ce_scorer.format_fingerprint()`, que tiene
que ser igual a `HYPOTHESIS_FORMAT_ID` o el test falla). Cambiar cualquiera
de las piezas de arriba mueve la huella: entonces hay que actualizar la
constante y republicar toda cifra medida con la anterior. Las cifras de
`artifacts/gates/T-ce-scorer/` y `artifacts/gates/T-ce-mechanics/` llevan
la huella dentro para que se sepa cuál es cuál.

## Resolution

**Mecánica entregada y verde; la cifra de sobreajuste NO está medida.**

El operador está trabajando en la máquina y prohibió arrancar entreno (ni
CPU ni GPU) en esta tanda, así que el gate queda SIN FIRMAR a propósito:
`artifacts/gates/T-ce-mechanics/overfit.json` lleva `pass: null`,
`status: "awaiting-operator-compute"` y todas las casillas de cifra en
`null`. No hay ningún número fabricado. Para firmarlo basta arrancar el
job parado del daemon `ce-overfit-mechanics`, que lleva el comando exacto
en `--device cpu`; el run sobreescribe el artefacto con `status: measured`.

Lo que sí está hecho y cubierto por test de unidad (modelo de juguete de
dos capas, CPU, sin pesos reales — `training/python/test_ce_overfit.py`,
31 tests):

- **conjunto**: 48 casos (rango 32–64) en las dos familias que el scorer
  sin entrenar resuelve peor, ES+EN, K=3 y K=4, huella
  `aadbb6a235c2361e`. 36 de los 48 golds se RECALCULAN aplicando la regla
  de la familia a los números que se leen del texto renderizado del
  candidato; un id tecleado que no cumple la regla no entra
  (`validate_set` levanta). Los 12 de inferencia/negación son a mano y el
  artefacto dice que lo son;
- **gold alineado**: se resuelve por id contra la misma tupla que se acaba
  de renderizar, y los candidatos se permutan en cada época; el test mide
  que permutar no mueve `p_gold` y, con teeth, que leerlo por POSICIÓN sí
  lo movería;
- **truncado**: `only_first`, y un par que no cabe levanta
  `ScorerContractError` antes de empezar; cuando se recorta a propósito el
  test comprueba que lo que sobrevive entero es la hipótesis;
- **máscara**: el relleno se queda a probabilidad exactamente 0, una fila
  de K=3 puntúa igual sola que mezclada con filas de K=4, y ningún
  gradiente llega a una columna que no existe;
- **formato entreno/eval**: se comparan los strings que tokeniza el
  entreno con los que construye `eval/ce_nograd.py` (su `measure_one`
  entero, con juguete inyectado) para las MISMAS decisiones — bytes
  iguales, mismo recorte y misma ventana;
- **el gradiente llega**: unos pasos de AdamW sobre el juguete bajan la
  pérdida listwise, con gradiente no nulo comprobado en cada paso.

Umbrales escritos antes de medir: `>0,95` de sobreajuste, `<=0,60` el
checkpoint pelado, y `>=0,35` de distancia entre las dos. Y el artefacto
repite, en el campo `not_generalization`, que la cifra de sobreajuste no
mide generalización y no se compara con nada externo.
