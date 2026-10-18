---
id: T-episode-splits
title: Splits que no filtran — por familia, entidad, espacio y grupo de variantes
status: next
priority: high
owner: unassigned
category: data
initiative: episodic-data
depends_on:
  - T-counterfactuals
  - T-episode-verify
created: 2026-09-25
updated: 2026-09-27
---

> **2026-09-27:** P1, no bloquea el bucle. El bucle evalúa en la batería dev (escrita aparte) y el productor deduplica contra dev y sellado. Esta task formaliza el repartidor de cuatro dimensiones para las mezclas publicadas.

# Splits que no filtran — por familia, entidad, espacio y grupo de variantes

`#T-split-domain` ya prohibió el split por índice de fila. Los episodios añaden
un modo de fuga nuevo y peor: si dos variantes del mismo caso caen en train y en
test, el test mide memorización del caso, no comprensión del cambio.

Repartir simultáneamente por **familia de plantilla, entidad, espacio de
etiquetas y `variant_group`**. Ninguna de las cuatro dimensiones puede cruzar el
corte. Y mantener un **corte privado que el profesor no haya producido ni haya
visto durante la generación** — con la honestidad de no declarar ausencia de
contaminación del preentrenamiento, que no es verificable.

`data/firewall.py` y `data/leakage.py` están bien hechos y se reutilizan tal
cual: lo que falta es registrar y usar, no detectar.

## Verification gate

- Test: inyectar deliberadamente una variante del mismo `variant_group` en train
  y en test hace fallar el repartidor.
- Test: el detector de fuga (hash exacto, normalizado y semántico) corre sobre
  cada mezcla publicada y su resultado va al manifest.
- El manifest de cada mezcla declara qué familias, entidades y espacios quedan
  fuera del train y con qué semilla.

## Done when

- Las cuatro dimensiones de split se aplican juntas y están verificadas por test.
- Existe un corte privado no producido por el profesor, con su sha registrado.
- Cada mezcla publicada lleva manifest reproducible por semilla, y el consumo
  efectivo de un entreno se puede auditar contra él.
