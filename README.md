# Asimovia Decision Model

An original decision-model research project: architecture, training,
evaluation, weights and runtime.

The product goal is a model designed here and trained from random
initialization on traceable datasets. It encodes shared state once, scores
dynamic options without autoregressive decoding and returns typed calibrated
decisions.

Third-party models such as JEv, TinyJev and Qwen are local teachers,
references and data tools. Their code and weights are not part of the model
published by this project. The earlier TinyJev fine-tuning path is preserved
as a licensed fallback experiment, not presented as our model.

Current work starts with a small encoder and pointer head trained entirely
here, then scales only when measured capacity demands it.

```bash
.venv-train/bin/python -m training.python.own_train \
  --episodes path/to/train.jsonl \
  --out artifacts/checkpoints/own/run-001
```

See [project lineage](.meshkore/docs/project-lineage.md) and
[the active initiative](.meshkore/roadmap/initiatives/own-decision-model.md).
