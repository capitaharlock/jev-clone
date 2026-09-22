---
id: T-train-monitor
title: Live training monitor (terminal + dashboard, 24/7)
category: data
initiative: data-training
status: done
created: 2026-09-20
closed: 2026-09-20T22:40:00.000Z
resolved_by: A003
resolved_by_conv: roadmap-architect-uwgjq
commit_shas: ['0ecbd528efbe71e12376b8de0ecd7814df0ea53b']
completed_at: 2026-09-22T10:31:47.403Z
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

✗ #T-labelspace-div wake es falso · commit `f66a6b4` está en HEAD. La task queda `next` en disco, que es lo correcto: depende de #T-antiscale-diag, que no está hecha. Sin cambio de estado.

Lo que sí encontré: #T-antiscale-diag estaba marcada `active` **sin nadie ejecutándola** (`/runs` = 0 agentes). Era un `active` fantasma bloqueando las tres tareas que cuelgan de ella. Ya no.

🚀 A031 (`developer-copy`, opus, pid 6256 vivo) → #generalization-fix #T-antiscale-diag
· mide los 5 ejes de por qué la accuracy unseen **cae** al escalar, sobre los checkpoints que ya existen — no propone arreglos, nombra la causa dominante y **qué % de la caída explica**. Ese número es el que prioriza #T-unfreeze-backbone vs #T-gen-objective.

<details><summary>Por qué esta y no otra — el resto está encadenado</summary>

- #T-antiscale-diag · `depends_on: []` → única READY del work-stream.
- #T-unfreeze-backbone · `depends_on: T-antiscale-diag` → espera.
- #T-gen-objective · `depends_on: T-antiscale-diag` → espera.
- #T-labelspace-div · `depends_on: T-antiscale-diag` → espera (es la hipótesis gemela por el lado de datos).
- #honest-eval #T-release-gate · espera tu firma, no un agente.
- Paralelo cruzado de iniciativa: el daemon lo rechaza (409 `initiative-already-in-flight`). El segundo agente (`muse`) queda libre para la siguiente ola en cuanto A031 drene.

Restricciones que le puse: usar los checkpoints existentes (solo el eje 3 permite dos runs cortos), lanzar esos runs como **job del daemon** y no dentro de su turno, y marcar `measured: false` con la razón antes que inventar un número.
</details>

Te aviso cuando A031 cierre con el eje dominante.

— #T-labelspace-div · verificada contra HEAD, sin flip (wake falso)
— #T-antiscale-diag · desbloqueada del `active` fantasma y despachada a A031

3.5M tokens
