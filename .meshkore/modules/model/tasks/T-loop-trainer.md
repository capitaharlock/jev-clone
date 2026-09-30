---
id: T-loop-trainer
title: El trainer que continúa — mezcla multi-fuente declarada y checkpoint que no empieza de cero
status: done
priority: high
owner: unassigned
category: model
initiative: daily-learning-loop
depends_on:
  - T-ce-finetune
created: 2026-09-27
updated: 2026-09-30
---

> **2026-09-27:** P0 del bucle (`docs/bucle-infinito.md` §1, §2, §8.3). Añade a lo de abajo: **tope de repetición** (ningún episodio se ve más de 4 veces en total), fuentes = `artifacts/episodes-qwen/{pilot-2k/verified.jsonl,stream/*}` + `artifacts/episodes-rule/*` + `artifacts/episodes-external/typed-decisions/train`, y `--weights` de cualquier peldaño de `#T-backbone-ladder`. Escribe `consumed_ids` en `train_manifest.json`: cierra la casilla «consumido» del embudo de `#T-episode-verify`.

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

## Resolution

El supervisor rota los lotes por regla no consumidos, añade repaso fresco de
las cinco familias y escribe los IDs consumidos, su hash, el número de IDs
únicos y la exposición máxima. El mejor checkpoint dentro de cada ciclo se
conserva; el resto se poda sin tocar la base activa.

La promoción compara candidato y base activa con bootstrap pareado para el
acierto forzado y la consistencia conjunta. El ciclo 34 terminó en 0,7475 y
0,6071, pero no promovió: los IC95 de mejora fueron
`[-0,0100, 0,0425]` y `[-0,0286, 0,0714]`. El supervisor quedó pausado tras
ese ciclo. El siguiente arranque depende de `#T-aligned-ranking-v2`, no de
repetir la mezcla actual.
