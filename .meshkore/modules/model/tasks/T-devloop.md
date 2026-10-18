---
id: T-devloop
title: Bucle de desarrollo sobre el bucle cerrado
status: active
priority: critical
owner: unassigned
category: model
initiative: daily-learning-loop
depends_on:
  - T-autoloop
created: 2026-10-13
updated: 2026-10-18
---

# Bucle de desarrollo

`tools/devloop.py`, bajo `com.jevclone.devloop`, vigila el bucle cerrado cada
diez minutos y cambia la receta cuando deja de rendir.

## Por qué

Del ciclo 1 al 8 la receta fue siempre la misma: 200 pasos, lr 2e-5, cien de
calentamiento y mezcla 45/30/25. Tras la promoción del ciclo 5 los candidatos
quedaron de media un punto por debajo del campeón. Repetir esa receta no
mejora el modelo.

## Brazos

- `base`: la receta de los ciclos 1 a 9.
- `short-warmup`: 20 pasos de calentamiento en vez de 100.
- `gentle`: lr 1e-5; los rechazos pierden sobre todo familias ya resueltas.
- `priority-focus`: Qwen genera prioridad y negación; reglas de prioridad x4.
- `rules-heavy`: mezcla 35/20/45; el gold de reglas es exacto.
- `longer`: 400 pasos.

Cada brazo corre una vez; después gana la mejor cota superior de la ganancia
medida (media del candidato menos media del campeón de su momento). El bucle
cerrado pide el brazo al empezar cada ciclo y lo fija hasta el gate.

## Retiro por mejor ganancia (2026-10-18)

Del 15 al 18 de octubre el selector quedó fijo en `nudge` (50 pasos, lr
2e-6): la regla antigua retiraba para siempre todo brazo de paso grande en
cuanto todos perdían de media. `nudge` apenas mueve los pesos, pierde poco y
nunca promociona: 45 ciclos, media -0.0046, mejor -0.0011. En v3 los
candidatos 65 a 70 quedaron entre 851 y 856 aciertos sobre 1350, frente a 860
del campeón.

La regla nueva retira un brazo tras ocho intentos sin ganar nunca al campeón
y elige por cota superior entre los que siguen vivos. Dos brazos nuevos:

- `long-continue`: 1200 pasos, lr 1e-5, mezcla de 60000 filas. A 48
  decisiones por paso es una pasada; con la mezcla de 12000 serían ocho.
- `long-low`: lo mismo a lr 5e-6.

Un ciclo largo dura unas 7 h (5.5 h de entrenamiento a unos 16 s por paso).
El tope de pasos del bucle cerrado sube a 1500 y la mezcla acepta de 4000 a
80000 filas.

## Salud

Comprueba el bucle cerrado, Ollama, el monitor, el disco y la corriente.
Relanza con `launchctl kickstart` lo que launchd supervisa y deja alertas del
resto. Escribe un informe diario en `artifacts/devloop/reports/`.

## Límites

- No toca el gate, la batería sellada ni los datos.
- La ganancia se mide en `battery_dev`; elegir brazos sobre ella la desgasta
  igual que el gate. `T-dev-rotation` sigue pendiente.
- Con ocho ciclos sin promoción marca meseta: hace falta dato o idea nueva de
  la revisión semanal, y una medida sellada del campeón.
