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
completed_at: 2026-09-21T09:23:16.823Z
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

No, no estamos a máxima capacidad. El tren anda (1 núcleo al 99%) pero el generador está muerto y Q-W-E-N aporta cero.

Lo que veo ahora: `train_baseline.py` corriendo al 98% CPU, monitor vivo en 8794, pero `data_gen_loop.py` sin proceso vivo y `synth-loop.jsonl` congelado en 26.700 filas desde las 08:01. Carga media 7,2 sobre 16 núcleos: la máquina va a medio gas. Y confirmo lo importante: entrenamos encoders de decisión (TF-IDF + regresión por tarea), no un LLM; Q-W-E-N solo inventa plantillas, nunca toca pesos de lenguaje.

<details><summary>Estado medido — generador, train, Q-W-E-N</summary>

- Generador: `make gen-status` dice "(ningún data_gen_loop vivo)". Últimas líneas del log: batch 5 → 26.700 filas (15,1 MB), y ahí se para. Fichero con mtime 08:01, ~3h congelado.
- Train: pid 77264 con 3 min de CPU al 98,7% en la primera medida; en la segunda ya no listaba (el reentreno completo terminó entre ambas). Es el comportamiento normal: el monitor solo reentrena al detectar datos nuevos, y sin datos nuevos no hay nada que entrenar.
- Q-W-E-N: `plantillas frescas 0`, `grep qwen synth-loop = 0`, y el arranque registró `qwen-templates-disk=0` sin ninguna línea `[Q-W-E-N]` después. El hook existe (`--qwen-every 24` pide 5 plantillas a `qwen3.6:27b-mlx` en ollama :11434, que sí responde y tiene el modelo), pero en esta vida del generador nunca llegó al batch 24 o el fetch falló en silencio. Resultado: todo lo generado es plantilla programática, cero ideas frescas del modelo.
- Desfase: 26.700 en disco vs 9.760 entrenadas → hay ~17k filas en cola que el próximo reentreno sí consumirá.
</details>

<details><summary>Capacidad — cuánto margen hay</summary>

- CPU: 1 proceso Python a 1 núcleo de 16. El baseline sklearn es monohilo por tarea; paralelizar por tarea o subir el muestreo lo pondría a 8-16 núcleos sin tocar hardware.
- Datos: el cuello no es disco (15 MB) sino que el generador lleva 3h parado: cero casos nuevos = cero entrenamiento útil aunque el train diga "EN MARCHA".
- Q-W-E-N local (27b) da para plantillas cada ~4 min sin frenar nada; hoy aporta 0 por el punto anterior, no por falta de máquina.
</details>

<details><summary>Qué haría para ir al máximo — 3 pasos</summary>

1. Rearrancar el generador (es lo que te muestra "off"): `cd /Users/ricartjuncadella/Documents/Prj/asimovia/jev-clone && nohup .venv-train/bin/python tools/data_gen_loop.py --batch 500 --interval 10 >> /tmp/gen-loop.log 2>&1 &`. Con eso el monitor vuelve a ver `mtime` nuevo y reentrena solo.
2. Bajar `--qwen-every` de 24 a 6-8 para que Q-W-E-N aporte plantillas desde el primer minuto y comprobar con `tail -f /tmp/gen-loop.log` que aparece `[Q-W-E-N] +N templates live`. Si sigue en +0, el fetch a ollama falla y hay que mirarlo.
3. Entrenamiento: pasar de reentreno completo ocasional a paralelo por tarea (un worker por núcleo) y subir el muestreo de civil-comments; eso sí pondría la carga cerca de 16. E indexar internet (Wikidata/programático, el #T-prog-gold) es el siguiente volumen real, no más velocidad del loop actual.
</details>

Si quieres lo rearranco yo ahora y te confirmo con `make gen-status` que vuelve a sumar filas.
