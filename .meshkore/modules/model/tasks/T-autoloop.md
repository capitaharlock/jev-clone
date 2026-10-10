---
id: T-autoloop
title: Bucle cerrado de mejora continua
status: active
priority: critical
owner: unassigned
category: model
initiative: daily-learning-loop
depends_on:
  - T-open-decision-backbone
  - T-priority-recovery
created: 2026-10-10
updated: 2026-10-10
---

# Bucle cerrado

`tools/autoloop.py`, bajo `com.jevclone.autoloop`, sustituye al productor de
reglas y al supervisor del entrenador. Cada ciclo genera con Qwen las familias
débiles, las verifica con un auditor separado, construye una mezcla y entrena
200 pasos desde el campeón. Después mide `battery_dev` en orden escrito y ocho
permutaciones, y promociona o descarta.

## Por qué

- El entrenador con `--limit 660000` leía solo los primeros lotes de reglas;
  más de 5 000 lotes posteriores nunca entraron.
- Solo veía dos familias con prosa de plantilla. La batería usa lenguaje
  natural, y la prioridad cayó a 0,7750 mientras el holdout de reglas marcaba
  1,0000.
- Los episodios públicos y los verificados por Qwen no se usaban.

## Mezcla por ciclo

- 45 % Qwen verificado, repetido como máximo cuatro veces.
- 30 % público de entrenamiento; las etiquetas anchas se recortan a `K` de 2, 3
  u 8 conservando el gold.
- 25 % reglas, muestreadas de ocho lotes al azar entre todos.
- Filtro de trigramas contra `battery_dev`, la sellada y el test tipado.

## Promoción

- Media de los nueve órdenes estrictamente mayor que la del campeón.
- Peor orden no inferior al del campeón.
- Ninguna familia cae más de 0,02.

## Riesgo abierto

Seleccionar siempre sobre `battery_dev` la desgasta. `T-dev-rotation` debe
dar un segundo corte antes de declarar el 95 % sostenido.
