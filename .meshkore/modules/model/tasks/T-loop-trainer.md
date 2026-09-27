---
id: T-loop-trainer
title: El trainer que continúa — mezcla multi-fuente declarada y checkpoint que no empieza de cero
status: next
priority: high
owner: unassigned
category: model
initiative: daily-learning-loop
depends_on:
  - T-ce-finetune
created: 2026-09-27
updated: 2026-09-27
---

# El trainer que continúa — mezcla multi-fuente declarada y checkpoint que no empieza de cero

## Contexto

`#T-ce-finetune` entrega `training/python/ce_finetune.py` para un entreno
acotado. El bucle diario necesita tres cosas más, y ninguna es «más épocas»:

1. **Continuar** desde `artifacts/checkpoints/ce/current` (enlace o fichero
   `current.json` con el sha del checkpoint promovido) con el optimizador
   reiniciado o restaurado — **decídelo midiendo** en dev y escribe cuál.
2. **Mezcla declarada** (`training/python/mix_manifest.py`): lista de fuentes
   (`artifacts/episodes-qwen/day-*`, `artifacts/episodes-external/*/train`,
   futuros objetivos suaves de Jev) con peso, cap por dataset (≤ 25 % de lo
   externo), semilla, y el resultado de `data/leakage.py` contra dev, sellado
   y los cortes `eval-only`. El trainer **rechaza** una mezcla sin manifest o
   con fuga.
3. **Consumo auditable:** `train_manifest.json` dice, por fuente, familia e
   idioma, cuántas filas entraron de verdad — es la última casilla del embudo
   de `#T-episode-scale` y `#T-episode-verify`.

## Qué hacer, paso a paso

1. `mix_manifest.py build --sources … --weights … --cap 0.25 --seed S --out
   artifacts/mixes/<fecha>.json`: resuelve fuentes, cuenta, corre fuga,
   escribe. Test: un `eval-only` en las fuentes falla; sin manifest, `train`
   se niega.
2. `ce_finetune.py train --mix artifacts/mixes/<fecha>.json --init current
   --budget N`: muestreo por pesos con semilla, checkpoint de salida en
   `artifacts/checkpoints/ce/<fecha>/`, `curve.jsonl` con evaluación
   intermedia (dev) a presupuestos predeclarados.
3. **Curriculum mínimo medible:** episodios contrafactuales con peso ≥ 1
   respecto al externo (parámetro), y el gate de dev mide el contrafactual
   conjunto aparte: si baja, la mezcla externa está tapando la señal.
4. Gate `artifacts/gates/T-loop-trainer/gate.json`: dos continuaciones reales
   (día 1 → día 2) con sus manifests, tiempos, y la comparación dev día 2 vs
   día 1 vs sin ajustar. Sin cifras a mano.

## Done when

- Un entreno puede partir de `current`, consumir una mezcla con manifest y
  dejar el consumo auditable por fuente/familia/idioma (test).
- Una mezcla con fuga o con un corte `eval-only` no entrena (test).
- Dos días consecutivos de continuación están medidos en dev contra el día
  anterior y contra el checkpoint sin ajustar.
