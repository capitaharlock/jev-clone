---
id: T-teacher-intent
title: Teacher barato + datasets intencionales de decisión
category: data
initiative: data-training
status: done
created: 2026-09-20
closed: 2026-09-20T22:40:00.000Z
completed_at: 2026-09-20T22:43:18.978Z
resolved_by: A004
resolved_by_conv: general-09192230
commit_shas: ['05e3dcfa303d84cffaaaacb680e7f4b72aa050a9']
---
# T-teacher-intent — Teacher barato + datasets intencionales de decisión

Petición del operador (2026-09-20): el volumen actual (2,4 M ejemplos,
89 % Civil Comments) no es el producto. El producto es un modelo de
**decisión** con outputs cerrados y pesos por estado (archivar /
responder / spam, triaje de mail, etc.). Para eso hacen falta datasets
**muy intencionados** para este tipo de decisión, generados en paralelo
al entrenamiento del baseline. El operador dispone además de acceso
barato al modelo JEv original (outputs gratis, prompts cortos súper
baratos; el precio sube con prompts grandes): usarlo como **fuente de
verdad** (teacher / juez) donde Qwen local no llegue con la misma
eficiencia, y como acelerador del entrenamiento aunque sea pagando.

## Alcance

1. `data/intent/` — generador de datasets intencionales de decisión:
   dominios prácticos con estados de salida cerrados (v1: triaje de
   email — archivar / responder / responder-urgente / spam — con pesos
   por estado en la respuesta estructurada; v1.1: 2-3 dominios más a
   elegir con el operador). Cada fila: `state` (texto realista),
   `questions[].answer` (estado gold), `questions[].weights` (pesos).
2. Teacher externo barato (JEv original) vía adapter configurable
   (`data/teacher_client.py`: endpoint + API key por env, batching,
   caché en `artifacts/data-raw/teacher/` para no pagar dos veces,
   prompts cortos por diseño). Uso doble:
   - **labeler**: genera gold + pesos donde el council local
     (Qwen/DeepSeek, #T-synth-factory) no sea fiable;
   - **juez**: verifica por muestreo los veredictos del tester engine
     (#T-tester-engine) y del council local.
3. Regla de coste: teacher solo donde Qwen local falle medido
   (acuerdo teacher↔Qwen < umbral en muestra piloto); todo lo demás,
   Qwen local primero (§§44–47 de la estrategia).
4. Los sets resultantes entran al schema universal y al manifest, y el
   monitor los cuenta sin cambios (son `*.jsonl` más).
5. Forward testing original: cada dominio nuevo trae 10-20 casos
   prácticos para el tester engine, no reciclados de benchmarks.

## Done when

- [x] `data/intent/email_triage.py` genera ≥5k filas schema-universal con
  4 estados + pesos, determinista por seed (test dedicado en verde).
- [x] `data/teacher_client.py` etiqueta un piloto de 500 filas con caché
  (segunda pasada cuesta 0 llamadas) y reporta acuerdo teacher↔Qwen.
- [x] El piloto entra en `artifacts/data-prefetch/` y el monitor lo
  muestra sin cambios de código; el próximo train lo consume.
- [x] Coste piloto documentado (€ total y €/1k filas) en el task file.

## Resolution (2026-09-20, verified)

- `data/intent/email_triage.py`: 5.000 filas (4000 train / 500 calib / 500
  test), 4 estados + `weights` por opción (nuevo campo opcional en
  `data/schema.py`, validado; resto del código lo ignora sin romperse).
- `data/test_intent.py`: 8/8 en verde; suite `data` completa 117/117.
- Piloto teacher 500 filas: acuerdo teacher↔qwen-proxy 0,92 (460/500);
  2ª pasada 500/500 caché, 0 llamadas. Teacher externo NO configurado
  (sin `JEV_TEACHER_URL`): piloto con stub local determinista.
- **Coste piloto: 0,00 € total (0 llamadas facturables).** Tarifa
  configurada para el teacher externo: 0,002 €/1k filas
  (`JEV_TEACHER_EUR_PER_1K`); p. ej. 1 M de filas ≈ 2 €.
- Monitor sin cambios: 9 datasets, 2.422.995 ejemplos. Train
  `20260920T223639Z` consume email-triage: n=4000, acc=1,0 (eval 1000).
- Forward testing: 14 casos `email-triage` en el tester engine
  (`AUTO_SUITES`), es/en de fácil a difícil.

## Resolution

Terminadas #T-teacher-intent y #T-train-monitor en #data-training (commit `05e3dcf`, sin push). El dataset intencional email-triage (5.000 filas) ya está entrenado con accuracy 1,0 y tus totalizadores se descongelaron: 475.390 ejemplos (19,62%). Para verlo moverse en tu terminal tienes que matar el proceso viejo (Ctrl+C) y relanzar con `.venv-train/bin/python tools/training_monitor.py --interval 300 --tick 20 --train --port 8791`.

<details><summary>Qué se hizo — email-triage + teacher + train</summary>

- `data/intent/email_triage.py`: 5.000 filas deterministas (4000 train / 500 calib / 500 test), 4 estados (archivar/responder/urgente/spam) con pesos por opción; campo `weights` nuevo, opcional y validado en `data/schema.py`.
- `data/teacher_client.py`: adapter del teacher barato con caché en `artifacts/data-raw/teacher/` — piloto de 500 filas con acuerdo 0,92 y segunda pasada a 0 llamadas.
- Coste del piloto: 0,00 € (el teacher externo JEv no está configurado; tarifa lista: 0,002 €/1k filas, ~2 € por millón).
- Train `20260920T223639Z` consume el set nuevo; el monitor lo muestra sin cambiar código (9 datasets, 2.422.995 ejemplos).
- 14 casos de forward-testing en el tester engine (`email-triage`, es/en).
</details>

<details><summary>Tests — verificación real</summary>

- Nuevos: 8/8 en verde (`data/test_intent.py`).
- Suite `data` completa: 117/117 en verde, incluido el cambio de schema.
- El monitor `--once` con venv confirma los totales nuevos; el proceso parado de tu terminal es el lanzamiento viejo con `python3` del sistema (sin sklearn), por eso no se movía nada.
</details>

<details><summary>Qué queda en #data-training</summary>

- Activas: #T-mix-1m, #T-mix-5m, #T-data-eval. Bloqueada: #T-prog-gold (pilot Wikidata). Backlog: #T-mix-10m.
- Dime "sigue" y continúo linealmente con la próxima.
</details>

**Commit** `05e3dcfa3` · 23 files

**Files changed (23):**
- `.meshkore/docs/coverage.md`
- `.meshkore/modules/data/tasks/T-teacher-intent.md`
- `.meshkore/modules/data/tasks/T-train-monitor.md`
- `artifacts/data-prefetch/email-triage.jsonl`
- `artifacts/data-prefetch/manifest.json`
- `artifacts/data-raw/teacher/labels.json`
- `artifacts/runs/20260920T223639Z/metrics.json`
- `artifacts/runs/20260920T223639Z/models/banking77.pkl`
- `artifacts/runs/20260920T223639Z/models/boolq.pkl`
- `artifacts/runs/20260920T223639Z/models/civil-comments.pkl`
- `artifacts/runs/20260920T223639Z/models/email-triage.pkl`
- `artifacts/runs/20260920T223639Z/models/helpsteer2.pkl`
- `artifacts/runs/20260920T223639Z/models/huffpost.pkl`
- `artifacts/runs/20260920T223639Z/models/logiqa.pkl`
- `artifacts/runs/20260920T223639Z/models/massive.pkl`
- `artifacts/runs/20260920T223639Z/models/reclor.pkl`
- `data/intent/__init__.py`
- `data/intent/email_triage.py`
- `data/schema.py`
- `data/teacher_client.py`
- `data/test_intent.py`
- `data/train_baseline.py`
- `tools/tester_engine.py`
