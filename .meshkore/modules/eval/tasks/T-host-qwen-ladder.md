---
id: T-host-qwen-ladder
title: Esta maquina: MPS para el encoder, Qwen en dos tallas para datos y juez
status: active
priority: high
owner: unassigned
category: eval
initiative: daily-learning-loop
depends_on:
  - T-qwen38-ref
created: 2026-09-27
updated: 2026-09-27
---

# Esta maquina: MPS para el encoder, Qwen en dos tallas para datos y juez

**Por que.** El clon aterrizo en `/Users/rjuncadell001/Prj/asimovia/jev.clone`
en un MacBook Pro **M5 Pro, 48 GB, 20 nucleos GPU**. El operador pidio
arrancar el testing y ajustar el profesor local: el 27B no hace falta para
cada parrafo de dataset, y el tag que habia en Ollama no es el del roadmap.

**Hecho en esta maquina (2026-09-27, antes de pull):**

| Cosa | Estado |
|---|---|
| Chip | Apple M5 Pro, 18 CPU (6+12), 20 GPU, 48 GB |
| Ollama | 0.32.5 en `:11434` |
| Tags | solo `qwen3.6:27b-q8_0` (28 GB, Q8, Modelfile Atlas/QuantLab, 7 semanas) |
| Llama | no hay ningun tag llama |
| `qwen3.8:27b-mlx` | el profesor GO de `#T-qwen38-ref`, **no instalado** |
| `.venv-train` | aun no |

El 27B Atlas **no** es el profesor medido. Tiene system prompt de otro
proyecto. Las llamadas de este repo le ponen `TEACHER_SYSTEM` encima, pero
el peso a instalar para el juez y el verificador es `qwen3.8:27b-mlx`
(18 GB). Para volumen de prosa: `qwen3:8b` (~5 GB). No hay Llama en este
host; no se descarga uno "por si acaso".

El encoder de decision se entrena en **MPS**, no en CPU. La CPU de este
equipo aguanta MiniLM, pero el stack ya midio ~40 s / 1 000 decisiones en
Metal. `--device cpu` queda para los tests unitarios.

## Que hacer

1. `ollama pull qwen3:8b` (volumen) y `ollama pull qwen3.8:27b-mlx`
   (profesor). No tirar `qwen3.8:27b-mlx-bf16` (56 GB) ni un segundo Q8 de
   30 GB: 48 GB no dan para eso y el entreno a la vez.
2. `PYTHONPATH=. python3 -m data.local_models probe` escribe
   `artifacts/gates/T-host-qwen-ladder/speed.json` con frio/caliente de cada
   rol. El 8B es el defecto de `QWEN_VOLUME_MODEL` si esta instalado; el
   27B mlx es el de `QWEN_MODEL` / juez / verificador.
3. `python3 -m unittest data.test_local_models data.test_episode_gen`:
   verde sin levantar pesos. `eval.test_qwen38_ref` pide Python 3.11+
   (`enterContext`); corre en `.venv-train`.
4. Cuando exista `.venv-train`: `make torch-env` y un `ce_finetune eval
   --checkpoint none --device mps` sobre `data/battery_dev.jsonl`. Eso es el
   testing del modelo de decision, no el tester TF-IDF de
   `tools/tester_engine.py` (espera `artifacts/runs/*.pkl`, que no viajan
   en git).

**Red de este host.** `ollama pull` contra `registry.ollama.ai` devuelve 503
HTML de PwC ("Access Denied"). Hasta que el pull se haga fuera de esa red,
el unico peso local es `qwen3.6:27b-q8_0`. El probe mide ese; el 8B y el
3.8 mlx quedan como tags a instalar, no como supuesto.

## Done when

1. `speed.json` publicado con al menos dos filas (profesor y volumen) o una
   sola si el 8B aun no esta, y `foreign_system_prompt` anotado en el 3.6
   Atlas si se midio.
2. Jobs de `.meshkore/public/jobs.yaml` apuntan a este cwd, no al path del
   Mac anterior.
3. `docs/traspaso-maquina.md` y `docs/bucle-infinito.md` dicen MPS + dos
   tallas de Qwen.
4. El encoder no se lanza con `--device cpu` en los jobs de esta maquina.
