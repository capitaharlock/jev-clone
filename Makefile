# JEv training monitor — arranque/parada limpios (macOS + Linux).
# Uso:
#   make start          # mata el ocupante previo del puerto y arranca el monitor
#   make stop           # detiene el monitor (pidfile + puerto + restos)
#   make restart        # stop + start
#   make status         # puerto, proceso, dashboard y últimos totales
#   make logs           # cola del log del monitor
#   make start PORT=8791  # puerto alternativo
#   make retrain        # reentrena el baseline HISTORICO TF-IDF (no el producto)
#
# Siempre con .venv-train (el python del sistema no tiene sklearn y el
# modo --train moriría en cada ciclo). Ver memoria del rol `custom`.
#
# #T-train-real: el monitor ya NO arranca con --train. Ese flag reentrena
# data/train_baseline.py, que es el BASELINE HISTORICO (TF-IDF+LogReg, un
# pickle por dataset, sin texto de opción) y está fuera de la ruta de
# producto. El entrenamiento real es el job `train-decision` del daemon
# (training/python/train_decision.py); este monitor solo LEE sus
# artifacts/runs/<run_id>/metrics.jsonl y titula la métrica de etiquetas no
# vistas. `make retrain` sigue existiendo para reproducir el histórico.

PORT ?= 8794
VENV_PY := .venv-train/bin/python
MONITOR := tools/training_monitor.py
LOGDIR := artifacts/logs/monitor
LOG := $(LOGDIR)/monitor.log
PIDFILE := $(LOGDIR)/monitor.pid
ARGS := --interval 300 --tick 20 --port $(PORT)

.PHONY: start stop restart retrain status logs kill-port

start: kill-port
	@mkdir -p $(LOGDIR)
	@echo "arrancando: $(VENV_PY) $(MONITOR) $(ARGS)"
	@bash -c 'nohup $(VENV_PY) $(MONITOR) $(ARGS) </dev/null >>$(LOG) 2>&1 & echo $$! > $(PIDFILE); disown %1 2>/dev/null || true'
	@echo "desacoplado de esta terminal (puedes cerrarla sin matar el monitor)"
	@sleep 3
	@$(MAKE) -s status

# Fuerza un reentreno sobre los datos actuales (borra el marcador de
# "ya entrenado" y rearranca). Útil para comprobar que el pipeline se mueve.
retrain: stop
	@echo "AVISO: reentrena el BASELINE HISTORICO TF-IDF (#T-train-real);"
	@echo "       el modelo de producto es el job del daemon 'train-decision'."
	@rm -f artifacts/runs/training-monitor/last_train_data_mtime.txt
	@echo "marcador borrado: el próximo arranque reentrenará"
	@sleep 1
	@$(MAKE) -s start ARGS="--interval 300 --tick 20 --train --port $(PORT)"

stop:
	@if [ -f $(PIDFILE) ]; then kill `cat $(PIDFILE)` 2>/dev/null && echo "parado pid `cat $(PIDFILE)` (pidfile)" || true; rm -f $(PIDFILE); fi
	@$(MAKE) -s kill-port
	@-pkill -f "training_monitor.py" 2>/dev/null && echo "restos training_monitor eliminados" || true

restart: stop
	@sleep 1
	@$(MAKE) -s start

kill-port:
	@if lsof -ti :$(PORT) >/dev/null 2>&1; then \
		echo "liberando puerto $(PORT): `lsof -ti :$(PORT) | tr '\n' ' '`"; \
		kill `lsof -ti :$(PORT)` 2>/dev/null || true; sleep 1; \
		if lsof -ti :$(PORT) >/dev/null 2>&1; then \
			echo "SIGTERM no bastó, SIGKILL"; \
			kill -9 `lsof -ti :$(PORT)` 2>/dev/null || true; sleep 1; \
		fi; \
	else echo "puerto $(PORT) libre"; fi

status:
	@echo "--- puerto $(PORT) ---"
	@(lsof -i :$(PORT) -sTCP:LISTEN 2>/dev/null || echo "(nadie escucha en $(PORT))")
	@echo "--- procesos ---"
	@(ps aux | grep "[t]raining_monitor.py" || echo "(ningún training_monitor vivo)")
	@echo "--- dashboard ---"
	@$(VENV_PY) -c "import urllib.request; print('HTTP', urllib.request.urlopen('http://127.0.0.1:$(PORT)/', timeout=10).status)" 2>&1 | tail -n 2
	@echo "--- últimos totales ---"
	@$(VENV_PY) -c "import json; s=json.load(open('artifacts/runs/training-monitor/state.json')); print(f\"{s['trained_examples']:,} / {s['total_examples']:,} ({s['pct_trained']}%) · pendiente {s['pending_examples']:,} · {s['total_size']} · params≈{s['baseline_params_est']:,} · run {s['last_run']} · {s['ts']}\")" 2>&1 | tail -n 2

logs:
	@tail -n 50 $(LOG) 2>/dev/null || echo "(sin log todavía en $(LOG))"

# Generador de esquemas de decisión (#T-gen-schemas). Ya no es un bucle
# infinito desacoplado: la tirada acaba cuando su cuota está llena, así que
# corre en primer plano y termina. Un run largo es un job del daemon.
GENLOGDIR := artifacts/logs/genloop
GENLOG := $(GENLOGDIR)/gen.log
GENARGS := --per-cell 8

gen:
	@mkdir -p $(GENLOGDIR)
	@$(VENV_PY) tools/data_gen_loop.py $(GENARGS) 2>&1 | tee -a $(GENLOG)

gen-gate:
	@$(VENV_PY) tools/data_gen_loop.py --gate

GENDIV := artifacts/data-prefetch/gen-schemas/diversity.json

gen-status:
	@echo "--- última tirada ---"
	@$(VENV_PY) -c 'import json,sys;d=json.load(open(sys.argv[1]));print(json.dumps({k:d[k] for k in ("generated_at","rows","unique_skeletons","unknown_rate")},indent=2))' $(GENDIV) 2>/dev/null || echo "(sin tirada todavía: make gen)"

gen-logs:
	@tail -n 50 $(GENLOG) 2>/dev/null || echo "(sin log todavía en $(GENLOG))"

# --- Stack neuronal (#T-torch-stack) -------------------------------------
# torch + transformers viven en .venv-train; los pesos NO están en git.
.PHONY: torch-env torch-weights torch-smoke torch-test

torch-env:            ## reinstala el entorno de entreno desde el lock
	$(VENV_PY) -m pip install --require-hashes -r requirements-train.txt

torch-weights:        ## descarga/verifica Ettin-68M y ModernBERT-base
	$(VENV_PY) -m model.weights fetch
	$(VENV_PY) -m model.weights verify

torch-smoke:          ## forward pass real en MPS y CPU + gate.json
	$(VENV_PY) tools/smoke_torch.py

torch-test:
	$(VENV_PY) -m unittest model.test_torch_stack model.test_bakeoff
