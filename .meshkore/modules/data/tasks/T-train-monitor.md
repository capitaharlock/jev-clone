---
id: T-train-monitor
title: Live training monitor (terminal + dashboard, 24/7)
category: data
initiative: data-training
status: done
created: 2026-09-20
closed: 2026-09-20T22:40:00.000Z
resolved_by: A004
resolved_by_conv: general-09192230
commit_shas: ['e6cab6055d1d7204e034c82560854fee1eb6a455']
---
# T-train-monitor — Live training monitor (terminal + dashboard, 24/7)

Petición del operador (2026-09-20): un proceso autónomo sin fin que entrene
sobre todos los conjuntos disponibles, visible en terminal con totalizadores
(parámetros, volumen entrenado / pendiente, evolución del modelo) y, si es
posible, un mini dashboard web en tiempo real. Todo documentado para auditoría.

## Alcance

1. `tools/training_monitor.py` (solo stdlib):
   - Bucle sin fin (`--interval`, `--once` para una pasada).
   - Escanea `artifacts/data-prefetch/manifest.json` + tamaños en disco
     (volumen por dataset, total).
   - Escanea `artifacts/runs/*/metrics.json` (último run = entrenado hasta
     ahora: ejemplos, accuracy por tarea).
   - Totalizadores: nº datasets, ejemplos totales, ejemplos entrenados en el
     último run, pendiente, parámetros estimados del baseline
     (TF-IDF 50k feats × nº clases por tarea), estado de Qwen/Ollama y del
     conversor `data/qwen_convert.py`.
   - Con `--train`: relanza `data/train_baseline.py` cuando los datos son más
     nuevos que el último run; log en `artifacts/logs/monitor/`.
   - Escribe `artifacts/runs/training-monitor/state.json` cada ciclo
     (fuente del dashboard y de auditoría).
   - Con `--port N`: sirve dashboard auto-refrescante con los mismos totales.
2. Lanzamiento desatendido con `nohup` + log en `artifacts/logs/monitor/`.

## Fix v2 (2026-09-20) — el monitor parecía parado

Causa raíz doble, verificada en vivo:
1. El operador lo lanzó con `python3` del sistema (sin sklearn): cada
   ciclo `--train` moría al instante con `ModuleNotFoundError`. El venv
   correcto es `.venv-train/bin/python`. Ahora el monitor elige el venv
   solo y avisa si falta.
2. El entrenamiento bloqueaba el bucle (subprocess síncrono) y entre
   ciclos el monitor dormía 300 s en silencio: terminal y dashboard
   congelados. Ahora el train corre en fondo (`Trainer`, Popen), el
   terminal late cada `--tick` (20 s, con stream `[train]` del log) y el
   dashboard se regenera en cada latido + sección "Entrenamiento en vivo".
   Un test en vivo cazó además un `KeyError: train_note` ya corregido.

Comando correcto:
`cd /Users/ricartjuncadella/Documents/Prj/asimovia/jev-clone && .venv-train/bin/python tools/training_monitor.py --interval 300 --tick 20 --train --port 8791`

## Done when

- [x] `python3 tools/training_monitor.py --once` imprime totales y escribe
  `artifacts/runs/training-monitor/state.json`.
- [x] El dashboard (`--port`) muestra datasets, volúmenes, % entrenado y
  último run sin errores.
- [x] Documentado en iniciativa `data-training` y `coverage.md`.
- [x] Latido visible cada `--tick` en terminal + dashboard aunque el
  training esté en marcha; train en fondo con el venv.

## Resolution

**Done — verified 2026-09-20 22:36Z:** `tools/training_monitor.py --once`
con venv muestra 9 datasets (incl. email-triage, 2.422.995 ejemplos) sin
cambios de código; suite `data` 117/117 en verde. Pendiente del operador:
matar el proceso viejo lanzado con `python3` (sigue parado en su terminal)
y relanzar con `.venv-train/bin/python tools/training_monitor.py --interval
300 --tick 20 --train --port 8791`.

Tenías razón en todo: el monitor estaba roto de verdad, ya está corregido y verificado. Eran dos causas, ninguna era falta de datos.

Tienes que reiniciar tu terminal (Ctrl+C) y relanzar con este comando — el importante es `.venv-train`, no `python3`:
`cd /Users/ricartjuncadella/Documents/Prj/asimovia/jev-clone && .venv-train/bin/python tools/training_monitor.py --interval 300 --tick 20 --train --port 8791`
Ahora mismo hay un reentrenamiento completo corriendo que descongelará tus totalizadores al terminar. Y creé #T-teacher-intent en #data-training para los datasets intencionales + teacher barato.

<details><summary>Causas del "parado" — verificadas, no supuestas</summary>

- Lanzaste con `python3` del sistema, que no tiene sklearn: cada ciclo `--train` moría al instante (`ModuleNotFoundError`, en `artifacts/logs/monitor/train-2026-09-20T214017Z.log`). Los totales llevaban congelados desde el run bueno de las 15:45.
- Además el entrenamiento bloqueaba el bucle y entre ciclos el monitor dormía 300 s en silencio: ni terminal ni dashboard se movían aunque todo fuera bien.
</details>

<details><summary>Qué cambia el monitor v2 (commit hecho, sin push)</summary>

- Elige `.venv-train/bin/python` solo y avisa si falta.
- Entrena en fondo sin bloquear; el terminal late cada `--tick` (20 s) con líneas `[train]` en streaming y el dashboard se regenera en cada latido, con sección nueva "Entrenamiento en vivo".
- Verificación: arranque + streaming reales en vivo, `--once` OK, y el propio test cazó un `KeyError: train_note` ya corregido y re-verificado.
- `coverage.md` actualizado en disco (sin commitear: el fichero trae cambios concurrentes de otros agentes, no los barro en mi commit). Diario escrito.
</details>

<details><summary>GPU, más datos y teacher — mi reconsideración</summary>

- GPU: para el baseline actual (TF-IDF + regresión logística, ~3 min en CPU) no la necesitas. La GPU entra cuando pasemos a afinar encoders de verdad; hasta entonces, gastar GPU es tirar dinero.
- Más datos: sí, pero intencionales, no más volumen genérico. El 89% Civil Comments no enseña a decidir. #T-teacher-intent propone email-triage (archivar/responder/urgente/spam con pesos) como primer dominio y 2-3 más contigo.
- Teacher barato (JEv original): usarlo como labeler/juez solo donde Qwen local falle medido (acuerdo < umbral en piloto), con caché para no pagar dos veces y prompts cortos. Todo lo demás, Qwen local primero.
</details>

**Commit** `e6cab6055` · 3 files

**Files changed (3):**
- `.meshkore/modules/data/tasks/T-teacher-intent.md`
- `.meshkore/modules/data/tasks/T-train-monitor.md`
- `tools/training_monitor.py`
