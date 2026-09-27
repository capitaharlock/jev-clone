---
id: T-qwen38-ref
title: Qwen 3.8 sobre la batería de desarrollo — el profesor nuevo, medido antes de que genere
status: next
priority: high
owner: unassigned
category: eval
initiative: daily-learning-loop
created: 2026-09-27
updated: 2026-09-27
---

# Qwen 3.8 sobre la batería de desarrollo — el profesor nuevo, medido antes de que genere

**Por qué.** El operador cambió el modelo local a `qwen3.8:27b-mlx` (default de
`QWEN_MODEL` en `data/episode_gen.py`). Todo el volumen del bucle lo genera y
verifica ese modelo. Su referencia medida es la del 3.6: 0,965 forzada en dev
(`artifacts/gates/T-preflight-refs/`). Antes de que produzca miles de episodios
hay que saber si es igual, mejor o peor.

**Qué hacer.**
1. Relee `eval/preflight_refs.py`: la columna `qwen-local` y su control de
   permutación (el 3.6 lo falló, con un flip_rate de 0,125 a temperatura 0,7).
2. Corre la columna con `QWEN_MODEL=qwen3.8:27b-mlx` sobre las mismas 400 filas
   (`rows_sha256` `8e8ccea5…`). Tarda unos 15 min con ollama `OLLAMA_NUM_PARALLEL=4`.
   Guarda el resultado en `artifacts/gates/T-qwen38-ref/refs-qwen-local-3.8.json`,
   **sin sobrescribir** la columna del 3.6.
3. Repite el control de permutación **a temperatura 0**. Es la temperatura que
   usa el verificador.
4. Gate: forzada, macro por familia, contrafactual conjunto e IC95 %, al lado
   del 3.6. Da GO si la forzada del 3.8 tiene un IC que no queda por debajo del
   3.6. Si queda por debajo, NO-GO, y el productor (`data.stream`) vuelve a
   `QWEN_MODEL=qwen3.6:27b-mlx` hasta que el operador decida.

**Done when.** Gate firmado con las dos columnas sobre las mismas filas y la
decisión escrita de qué modelo usa el productor.
