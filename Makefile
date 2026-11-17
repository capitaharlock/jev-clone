PYTHON ?= python
DEVICE ?= cpu
HF_REPO ?= capitaharlock/asimovia-decision-v1
CHECKPOINT := checkpoints/asimovia-decision-v1

.PHONY: install test weights serve bench report plot

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m unittest discover -s tests -t .

weights: $(CHECKPOINT)/model.safetensors

$(CHECKPOINT)/model.safetensors:
	curl -fL -o $@ https://huggingface.co/$(HF_REPO)/resolve/main/model.safetensors
	@echo "expected sha256 00c0fd3bf3c7a3b19f586edf35406e0befde419ed63c49e071abe8331c7799dd"
	@shasum -a 256 $@ 2>/dev/null || sha256sum $@

serve: weights
	$(PYTHON) -m serve.server --model $(CHECKPOINT) --device $(DEVICE)

bench: weights
	$(PYTHON) -m bench.run rules
	$(PYTHON) -m bench.run own --model $(CHECKPOINT) --device $(DEVICE)
	$(PYTHON) -m bench.report

report:
	$(PYTHON) -m bench.report

plot:
	$(PYTHON) docs/plot_history.py
