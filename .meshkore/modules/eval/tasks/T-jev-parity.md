---
id: T-jev-parity
title: Paridad de protocolo con la receta publicada de Jev
status: done
priority: high
owner: unassigned
category: eval
initiative: full-space-training
depends_on:
  - T-option-text
created: 2026-09-24
updated: 2026-09-24
---

# Paridad de protocolo con la receta publicada de Jev

La cifra ajena que ya tenemos en la mesa —0,924 (2 846/3 080) sobre las filas de
test de BANKING77 con sus 77 intents— sólo es comparable si nuestro número sale
del **mismo protocolo**. Hoy `eval/fullspace.py:253-264` estampa
`same_rows: false` y «different rows», que no describe bien la diferencia: la
referencia se describe como las 3 080 filas oficiales. Lo verificable es otra
cosa —**al profesor no se le ha puntuado con nuestro pipeline y no hay evidencia
fila a fila**, además de diferir el régimen de ejemplos— y así se redacta el
artefacto.
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
  auditable. Del lado nuestro se declara con precisión: `run.json` del
  checkpoint de referencia excluye el **dataset** banking77 (`fence.clean_1m`,
  `fenced_datasets`), lo que no es lo mismo que la ausencia literal de cada
  cadena de etiqueta en las otras 12 fuentes —eso se comprueba— ni dice nada de
  la exposición previa del backbone preentrenado, que se declara aparte como no
  auditable.
- **La identificación de filas**: identificador estable por fila, para que
  «mismas filas» sea comprobable y no una afirmación. Si en algún run futuro
  BANKING77 entra en el entreno, el scoreboard separa gold visto/no visto dentro
  del mismo K=77.

## Done when

- `eval.fullspace` produce un artefacto de paridad con las 3 080 filas, K=77,
  azar, IC 95 % y las diferencias de protocolo enumeradas.
- Los brazos de `#T-option-text` (identificadores / descripciones / con
  ejemplos) enganchados a este artefacto, para que la diferencia de régimen de
  información sea un número y no una nota al pie.
- `same_rows` y su texto dicen exactamente qué no está verificado (puntuación
  del profesor con nuestro pipeline, evidencia fila a fila) en vez de «different
  rows».
- El gate corre sobre cualquier checkpoint por ruta y entra en el reporte de
  todo run de `#full-space-training` sin intervención manual.
- La cifra vieja (0,0123) se conserva en el artefacto como línea base, de modo
  que cada run publique su distancia a ella y al profesor.
