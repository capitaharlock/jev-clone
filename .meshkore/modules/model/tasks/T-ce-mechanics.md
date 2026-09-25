---
id: T-ce-mechanics
title: Validar la mecánica antes de gastar presupuesto — sobreajustar 32–64 casos
status: next
priority: high
owner: unassigned
category: model
initiative: cross-encoder-pilot
depends_on:
  - T-ce-scorer
created: 2026-09-25
updated: 2026-09-25
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
