---
id: T-priority-recovery
title: Recuperar prioridad sin degradar las otras familias
status: active
priority: critical
owner: unassigned
category: model
initiative: daily-learning-loop
depends_on:
  - T-open-decision-backbone
  - T-dev-rotation
created: 2026-10-09
updated: 2026-10-09
---

# Recuperar prioridad

La auditoría del paso 3800 dejó prioridad en 0,7750 frente a 0,8750 en el paso
996. El holdout de reglas alcanzó 1,0000 al mismo tiempo. Continuar el mismo
entrenamiento no está autorizado.

## Trabajo

- Clasificar los 18 errores de prioridad por idioma, `K`, operación y patrón.
- Comparar su cobertura contra el generador y sus pesos reales.
- Añadir casos verificables para prioridades condicionales, empates,
  exclusiones, negación y cambio de criterio.
- Preparar un desarrollo rotado antes de entrenar.
- Ejecutar un brazo corto desde el mejor punto recuperable o desde TinyJev 4B
  v1, guardando checkpoints frecuentes.
- Medir orden escrito, ocho permutaciones, familia, idioma y `K`.

## Gate

- Prioridad supera 0,8750.
- Media y mínimo de permutaciones superan el paso 996.
- Ninguna otra familia cae más de 0,02.
- El resultado se repite en el desarrollo rotado.
- La batería sellada permanece cerrada.
