---
id: T-jev-parity
title: Paridad de protocolo con la receta publicada de Jev
status: next
priority: high
owner: unassigned
category: eval
initiative: full-space-training
created: 2026-09-24
updated: 2026-09-24
---

# Paridad de protocolo con la receta publicada de Jev

La cifra ajena que ya tenemos en la mesa —0,924 de Jev sobre las 3 080 filas de
test de BANKING77 con sus 77 intents— sólo es comparable si nuestro número sale
del **mismo protocolo**. Hoy `eval/fullspace.py` estampa `same_rows: false`
porque el profesor vio 24 ejemplos etiquetados por predicción y nosotros no.
Esta task convierte esa cita en una medición con las diferencias declaradas una
a una, y la deja como **métrica de cabecera de cada run** del objetivo nuevo.

Es también la task que faltaba desde el día 1 (regla R6): ninguna elección de
arquitectura de este repo se tomó nunca con una cifra ajena delante.

## Qué se fija

- **Las filas**: las 3 080 oficiales de test, no una muestra.
- **La cardinalidad**: 77, siempre, con el azar (0,0130) impreso al lado.
- **El régimen de ejemplos**: el profesor ve ejemplos etiquetados por
  predicción; nosotros no vemos ninguno. La diferencia se declara en el
  artefacto en vez de esconderse — y se añade un brazo nuestro *con* ejemplos
  en el `STATE` para que la comparación tenga las dos lecturas.
- **La contaminación**: BANKING77 es público. Se declara que nadie puede
  auditar los datos de entreno del profesor y que nuestro firewall sí es
  auditable.

## Done when

- `eval.fullspace` produce un artefacto de paridad con las 3 080 filas, K=77,
  azar, IC 95 % y las diferencias de protocolo enumeradas.
- Un brazo nuestro con ejemplos etiquetados en el `STATE` medido al lado del
  brazo sin ejemplos, para que la diferencia de régimen sea un número.
- El gate corre sobre cualquier checkpoint por ruta y entra en el reporte de
  todo run de `#full-space-training` sin intervención manual.
- La cifra vieja (0,0123) se conserva en el artefacto como línea base, de modo
  que cada run publique su distancia a ella y al profesor.
