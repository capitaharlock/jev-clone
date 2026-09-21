---
id: T-torch-stack
title: Stack neuronal real — torch, transformers y pesos descargados
status: active
priority: high
owner: unassigned
category: model
initiative: decision-rebuild
depends_on: []
created: 2026-09-21
updated: 2026-09-21
---

# Stack neuronal real — torch, transformers y pesos descargados

Hallazgo A: **no hay PyTorch en el proyecto**. `.venv-train` contiene solo
numpy, scipy, scikit-learn, joblib, cloudpickle y narwhals; cero `import
torch` en todo el repo. `model/bakeoff.py` compara "proxies" (encoders hash
char-ngram a dim 256/1024/4096) con pesos aleatorios, y Ettin-68M,
ModernBERT-base, LFM2.5-230M y NeoBERT-250M figuran como `pending_weights`:
nunca se han descargado ni ejecutado.

Sin esta task no existe ni el camino a `#T-pointer-head`.

Trabajo:

1. `torch` + `transformers` + `tokenizers` + `safetensors` en `.venv-train`,
   con `requirements-train.txt` fijado por versión y hash. Recordatorio del
   histórico: el entrenamiento debe correr con `.venv-train/bin/python`;
   el `python3` del sistema no tiene ni sklearn.
2. Descargar de verdad **Ettin-68M** y **ModernBERT-base**, con sus
   `tokenizer.json`, a `artifacts/weights/<model>/`, y registrar
   sha256 + licencia + fecha en `.meshkore/docs/source-register.md`.
   Los pesos NO entran en git (ver `#T-repo-clean`).
3. Smoke test: `encode_state` de una fila real del schema V1 produce un
   tensor de la forma esperada en MPS y en CPU, con la misma salida hasta
   tolerancia; tiempo medido y anotado.
4. Retirar de `model/bakeoff.py` la ruta "proxy aleatorio" o marcarla
   explícitamente como no-comparable, para que no vuelva a entrar en un
   Pareto (lo cierra `#T-bakeoff-real`).

## Verification gate

- `.venv-train/bin/python -c "import torch, transformers; print(torch.__version__)"`
  responde, y `torch.backends.mps.is_available()` es `True` en la M-series.
- Test que carga cada backbone desde `artifacts/weights/` y falla si el
  sha256 no cuadra con el registro.
- El gate escribe `artifacts/gates/T-torch-stack/gate.json` con `pass: true`,
  versiones, shas de pesos y latencia de un forward en MPS y CPU.

## Done when

- `.venv-train` tiene torch + transformers fijados y reproducibles.
- Ettin-68M y ModernBERT-base están descargados, verificados por hash y
  registrados en `source-register.md` con su licencia.
- Ningún artefacto de bake-off puede volver a publicar un proxy aleatorio
  como si fuera un backbone.
