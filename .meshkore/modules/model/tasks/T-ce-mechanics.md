---
id: T-ce-mechanics
title: Validar la mecánica antes de gastar presupuesto — sobreajustar 32–64 casos
status: done
priority: high
owner: unassigned
category: model
initiative: cross-encoder-pilot
depends_on:
  - T-ce-scorer
created: 2026-09-25
updated: 2026-09-26
resolved_by: A003
resolved_by_conv: roadmap-architect-uwgjq
completed_at: 2026-09-26T17:51:10.512Z
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

**Veredicto: la tubería aprende. El código funciona.**

Sobreajuste **46/48 = 95,8 %** (umbral >95 %) con pesos reales en MPS, 60 épocas en **117 s**. El mismo checkpoint sin ajustar: **25 %**, por debajo del azar (31,9 %) — margen 0,708. Los cuatro chequeos que podían invalidar meses de entreno dan PASS: gold alineado, máscara de candidatos, formato entreno/eval idéntico, sin truncar el estado. 714 tests verdes; 2 rojos preexistentes y ajenos.

No es generalización — es la prueba de que gastar cómputo tiene sentido. Siguiente paso real: `#T-ce-finetune`.

↪ A051 (`developer-copy`) registrando el gate medido y cerrando la task.

— T-ce-mechanics · la mecánica del entrenador queda validada con pesos reales (46/48, gate PASS) — pendiente solo el commit de A051

22.1M tokens
