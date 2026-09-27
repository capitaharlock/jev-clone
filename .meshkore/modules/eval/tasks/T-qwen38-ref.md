---
id: T-qwen38-ref
title: Qwen 3.8 sobre la batería de desarrollo — el profesor nuevo, medido antes de que genere
status: active
priority: high
owner: architect-master
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

## Estado 2026-09-27 (arquitecto)

**Construido y verde, midiendo.** `eval/qwen38_ref.py` + `eval/test_qwen38_ref.py`
(14 tests, 0,48 s, sin cargar un modelo). La columna del 3.6 no se toca: se lee
de `#T-preflight-refs` y se reutiliza.

- **La regla está escrita antes de la cifra** (`GATE_RULE` en el módulo, y
  publicada en el gate): GO si el IC95 % de la diferencia **pareada** de la
  forzada (3.8 − 3.6, bootstrap de 2 000 réplicas, semilla 20260927, sobre las
  mismas 400 filas) tiene el extremo superior ≥ 0. NO-GO si el IC queda entero
  por debajo de 0, y entonces el productor vuelve a `QWEN_MODEL=qwen3.6:27b-mlx`.
  «Mejor» se publica aparte (`forced.ci95[0] > 0`) y no decide nada aquí.
- **Mismas filas comprobadas dos veces:** `rows_sha256` `8e8ccea5…` y
  `same_rows()` fila a fila. Sellado no leído.
- **Control de permutación a las dos temperaturas.** La columna se mide a la
  del generador (0,7, la misma a la que se midió el 3.6, que falló el control
  con `flip_rate` 0,125) y el control se repite en greedy (0, la del
  verificador). Para eso la temperatura del profesor pasó a ser un parámetro
  (`data.episode_gen.TEMPERATURE`, por defecto 0,7: nada cambia de
  comportamiento).
- **Protocolo por encima de la velocidad:** se mide fila a fila con
  `teacher_structured`, NO con el lote, porque el 3.6 se midió así. Con
  `OLLAMA_NUM_PARALLEL=4` (ya en el servidor) salen ~3,3 filas/min → **≈ 2 h**,
  no los 15 min que estimaba esta task. El lote (13 s/episodio) sería otro
  protocolo y no compararía.

Comandos:

```bash
PYTHONPATH=. python3 -m eval.qwen38_ref plan    # la regla, sin medir
PYTHONPATH=. python3 -m eval.qwen38_ref gate    # gate sobre lo medido
# job `qwen38-ref` (~2 h), reanudable por el log de picks:
env PYTHONPATH=. QWEN_MODEL=qwen3.8:27b-mlx python3 -u -m eval.qwen38_ref measure
```

Falta: que el job termine las 400 filas, `gate` firmado y la decisión de
productor escrita en `artifacts/gates/T-qwen38-ref/gate.json`.
