---
id: T-torch-stack
title: Stack neuronal real — torch, transformers y pesos descargados
status: done
priority: high
owner: unassigned
category: model
initiative: decision-rebuild
depends_on: []
created: 2026-09-21
updated: 2026-09-21
commit: 0d905fb
completed_at: 2026-09-21T12:44:04.943Z
resolved_by: A003
resolved_by_conv: roadmap-architect-uwgjq
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

## Resolution

✓ #decision-rebuild #T-halt-contam done · 9 ficheros · commit `88ad4e6`
Se acabó el verde falso: el generador synth-loop está muerto y sus 360 700 filas en cuarentena fuera de git; logiqa/reclor quedan como eval-only por firewall. Auditoría de fugas sobre 2 080 647 filas de train contra 22 752 ítems held-out: exact 0, normalized 0, paraphrase 0. 16/16 tests, dashboard honesto en :8794.

🚀 Opus (claude-code, pid 33119 vivo) → #decision-rebuild #T-optset-sampler · el dataloader que alimenta al pointer head: K opciones dinámicas por fila, hard negatives medidos, filas `unknown` reales y barajado por epoch sobre banking77+massive+huffpost+boolq.

Un solo agente esta oleada: #T-train-real cuelga de este sampler y #T-bakeoff-real de #T-train-real — la cadena es secuencial, y el daemon rechaza despachar fuera de #decision-rebuild mientras siga viva.

<details><summary>#decision-rebuild — estado de las 6 tasks</summary>

- `done`: #T-torch-stack (`0d905fb`), #T-pointer-head (`4506325`), #T-halt-contam (`88ad4e6`).
- `active`: #T-optset-sampler — en curso ahora.
- `next`: #T-train-real (dep: #T-pointer-head ✓ + #T-optset-sampler ⏳), #T-bakeoff-real (dep: #T-train-real).
- Cierre de iniciativa cuando las 6 estén `done`; entonces paso a #honest-eval.
</details>

<details><summary>Integridad — #data-training con 14 tasks</summary>

- El check marca `initiative_too_dense`. No lo toco a media oleada: repartir 14 tasks entre sub-iniciativas reescribe el `initiative:` de cada fichero y movería la card mientras #decision-rebuild está en vuelo.
- Lo propongo al cerrar #decision-rebuild, antes de entrar en el resto de #data-training.
</details>

— T-halt-contam · synth-loop congelado, corpus en cuarentena y firewall train↔benchmark con fuga cero

2.4M tokens
