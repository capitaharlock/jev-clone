---
id: T-prefetch
title: Prefetch + conversion of all training sources
status: done
initiative: decision-model
module: model
created: 2026-09-20
completed_at: 2026-09-20T11:22:49.056Z
resolved_by: A004
resolved_by_conv: general-09192230
---
# T-prefetch — Prefetch + conversion of all training sources

Download every training/eval source in `.meshkore/docs/source-register.md`
(P0 ×5, P1 ×3, CLINC150 eval-only, 8 backbone snapshots) into the local
HF hub cache ASAP, convert them to the universal schema (`data/schema.py`)
via `data/prefetch.py`, and record paths + revisions + IDs in the
source register so nothing is downloaded twice.

## Done when

- All 9 datasets + 8 model snapshots resolve in the local HF cache and
  `artifacts/data-prefetch/manifest.json` lists repo + revision + SHA per item.
- `artifacts/data-prefetch/<id>.jsonl` (universal schema, splits
  train/calibration/test/ood) exists for every P0 dataset; P1 sets have at
  least a normalized dump or a logged blocker.
- Source register `## Local prefetch cache` documents HF IDs, local paths
  and snapshot revisions; no second download of the same revision happens.
- The train license fence still holds: eval-only sources never enter a
  training manifest (`data/registry.py:check_mix` stays green).

## Resolution

Tienes razón a medias: 1,2 GB no preentrena un LLM, pero no es lo que estamos haciendo. Nosotros afinamos encoders ya preentrenados (17M–400M), no entrenamos un 7B/30B desde cero, y para eso 2,2M de ejemplos sí es un punto de partida serio, aunque corto para razonamiento.

<details><summary>Mi opinión — por qué el volumen encaja pero el balance no</summary>

- Un modelo 7B/30B/500B desde cero necesita TBs de texto porque aprende el idioma desde cero. Nosotros partimos de backbones que ya saben idioma (Ettin, ModernBERT, NeoBERT, LFM) y solo les enseñamos a decidir (clasificar, elegir, calibrar). Eso se hace con millones de ejemplos, no con TBs.
- Dicho eso, tu intuición es correcta en una cosa: lo que tenemos hoy es descompensado. 2M de filas son casi todo Civil Comments (toxicidad), y razonamiento (LogiQA, ReClor) son solo ~91k filas en crudo. Para ser "inteligente" decidiendo nos faltan justo lo previsto en roadmap: gold humano, hard negatives y datos programáticos/de dominio.
- Lo convertido en segundos es la conversión de formato, no la inteligencia. La inteligencia sale del entrenamiento que arrancó ahora (baseline boolq 0,68) más el aumento Qwen en fondo.
</details>

<details><summary>Datasets — nombre, fuente y volumen real en disco</summary>

- Banking77 — `PolyAI/banking77` — 13.083 ejemplos — 6,1 MB — intents, train.
- BoolQ — `google/boolq` — 12.697 ejemplos — 9,8 MB — booleanos, train.
- Civil Comments — `google/civil_comments` — 1.999.514 ejemplos — 1,0 GB — toxicidad/sesgo, train (domina el total).
- HelpSteer2 — `nvidia/HelpSteer2` — 21.362 ejemplos — 73 MB — preferencia/calibración, train.
- HuffPost — `khalidalt/HuffPost` — 66.510 ejemplos — 30 MB — categorías, solo smoke.
- LogiQA 2.0 — `datatune/LogiQA2.0` — 84.976 ejemplos — 66 MB — en crudo, pendiente adapter final.
- MASSIVE (solo en-US+es-ES) — `AmazonScience/massive` — 33.042 ejemplos — 13 MB — multilingüe.
- ReClor — `sxiong/ReClor` — 6.138 ejemplos — 6,5 MB — en crudo, pendiente adapter final.
- Total: 2.237.322 ejemplos — 1,2 GB en `artifacts/data-prefetch/`. Más ~100 variantes Qwen en `artifacts/data-qwen/` (200 KB, creciendo en fondo). CLINC150 está descargado pero es eval-only, no entrena.
</details>

<details><summary>Parámetros a entrenar — qué modelo y cuántos pesos</summary>

- No entrenamos ningún 7B/30B. Los candidatos son: Ettin 17M/32M/68M/150M/400M, ModernBERT 149M, NeoBERT 250M, LFM2.5 230M (todos en `~/.cache/huggingface/hub`).
- Lo que se entrena es el encoder completo o su cabezal de decisión: entre ~17M y ~400M de parámetros según el backbone elegido en el bake-off, más un cabezal lineal pequeño. El baseline actual (TF-IDF+regresión) tiene solo miles de parámetros y es solo referencia (boolq 0,68).
- El Qwen local de 27B (`qwen3.6:27b-mlx` por ollama) no se entrena: solo actúa como profesor/conversor que genera variantes. Fundirlo sirve para aumentar datos, no gasta entrenamiento propio.
</details>

28M tokens
