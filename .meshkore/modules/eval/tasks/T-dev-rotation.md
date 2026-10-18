---
id: T-dev-rotation
title: Rotar el corte de desarrollo — que elegir cada ciclo no agote la batería
status: next
priority: high
owner: unassigned
category: eval
initiative: daily-learning-loop
depends_on:
  - T-teacher-probe
created: 2026-09-27
updated: 2026-09-28
---

> **2026-09-28, sube a P0 y cambia de objetivo.** Jev saca **1,000** en el dev actual
> (`artifacts/gates/T-teacher-probe/battery.json`), así que ese dev no mide la parte alta.
> El dev v2 tiene que ser **más difícil**, no solo nuevo: K hasta 20, distractores
> cercanos, cadenas de 3 criterios, negación doble, estados largos con ruido. **Criterio de
> aceptación:** Jev medido encima con `eval/jev_battery.py` (apuntándolo al fichero nuevo)
> **no** puede sacar 1,0, y Qwen 3.8 tampoco. Ya no espera a `#T-loop-nightly`. Ver
> `docs/bucle-infinito.md` §0.1.

# Rotar el corte de desarrollo — que elegir cada ciclo no agote la batería

**Por qué.** El bucle elige en cada ciclo con las mismas 400 filas de
`data/battery_dev.jsonl`, y eso termina sobreajustando a dev
(`docs/bucle-infinito.md` §6). Esta task produce un dev nuevo cada 10 ciclos
promovidos o en cada hito.

**Qué hacer.**
1. Mismo contrato que `#T-battery-dev`: cinco familias, ES/EN, K=2/3/8,
   grupos contrafactuales, gold por regla o verificado, y mezcla declarada
   antes. Usa `data/battery_dev.py` como auditor. El texto lo redacta Qwen 3.8
   con el vocabulario de entidades separado del de train (`data/leakage.py`),
   más una muestra humana de 5 por celda para el operador.
2. Versiona: `data/battery_dev_v2.jsonl` (v3, …) con manifest y sha.
   `training.python.loop` lee cuál es el dev vigente de `artifacts/loop/state.json`.
3. El dev anterior pasa a **regresión**: se sigue midiendo y se publica en el
   marcador, pero ya no decide.
4. El sellado **no** se rota aquí; lo rota `#T-ce-confirm` después de abrirlo.

**Done when.** Un dev v2 publicado y auditado, el bucle cambiando de dev sin
tocar código, y la columna de regresión en el marcador.
