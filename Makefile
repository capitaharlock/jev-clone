PYTHON ?= python
DEVICE ?= cpu
HF_REPO ?= capitaharlock/asimovia-decision-v1
CHECKPOINT := checkpoints/asimovia-decision-v1
WEIGHTS_SHA256 := 00c0fd3bf3c7a3b19f586edf35406e0befde419ed63c49e071abe8331c7799dd

.PHONY: install test lint format weights serve bench report plot

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

weights: $(CHECKPOINT)/model.safetensors

$(CHECKPOINT)/model.safetensors:
	curl -fL -o $@.part https://huggingface.co/$(HF_REPO)/resolve/main/model.safetensors
	echo "$(WEIGHTS_SHA256)  $@.part" | shasum -a 256 -c -
	mv $@.part $@

serve: weights
	$(PYTHON) -m asimovia.serve --model $(CHECKPOINT) --device $(DEVICE)

bench: weights
	$(PYTHON) -m asimovia.bench rules
	$(PYTHON) -m asimovia.bench own --model $(CHECKPOINT) --device $(DEVICE)
	$(PYTHON) -m asimovia.bench.report

report:
	$(PYTHON) -m asimovia.bench.report

plot:
	$(PYTHON) docs/plot_history.py
