---
id: T-dev-rotation
title: Rotar el corte de desarrollo — que elegir cada ciclo no agote la batería
status: next
priority: medium
owner: unassigned
category: eval
initiative: daily-learning-loop
depends_on:
  - T-loop-nightly
created: 2026-09-27
updated: 2026-09-27
---

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
