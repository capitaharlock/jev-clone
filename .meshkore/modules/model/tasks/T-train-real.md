---
id: T-train-real
title: Entrenamiento listwise real que sustituye a train_baseline.py
status: done
priority: high
owner: unassigned
category: model
initiative: decision-rebuild
depends_on:
  - T-pointer-head
  - T-optset-sampler
created: 2026-09-21
updated: 2026-09-21
resolved_by: A014
resolved_by_conv: work-decision-rebuild-T-train-real-1790005
---
# Entrenamiento listwise real que sustituye a train_baseline.py

Reemplaza el bucle de 24 h (TF-IDF 50 k 1-2 gram + LogisticRegression, un
pickle por dataset, ≈1,8 M parámetros de bolsa de palabras) por el
entrenamiento del head de `#T-pointer-head` alimentado por
`#T-optset-sampler`.

Trabajo, en `training/python/train_decision.py`:

1. Pérdida **listwise**: cross-entropy sobre los K logits de la fila
   (+ `unknown`), no clasificación multiclase global. Brier/ECE como
   métricas secundarias desde el primer step.
2. Multi-dataset en un solo modelo: banking77 + massive + huffpost + boolq
   en la misma mezcla, no un pickle por dataset. Un modelo, no siete.
3. Checkpoints en **safetensors** + `tokenizer.json` copiado al lado, con
   `model_version` en el manifest — la clave compuesta que
   `crates/jev-runtime` ya usa en su cache de estado.
4. Curvas por step a `artifacts/runs/<run_id>/metrics.jsonl`; el dashboard
   `:8794` lee de ahí y muestra la métrica primaria de `#I-honest-eval`, no
   accuracy de fila.
5. Corre como **job del daemon** (id `train-decision`), no como nohup de una
   terminal, para que sobreviva a la sesión y el operador lo vea en Jobs.
6. `data/train_baseline.py` queda como baseline histórico explícitamente
   etiquetado, fuera de la ruta de producto.

Presupuesto: primer corte sobre ~1 M filas de la mezcla limpia; la curva de
escala (250 k → 1 M) decide si merece la pena más.

## Verification gate

- El modelo entrenado bate el azar **en etiquetas no vistas** en el split de
  `#T-unseen-labels`. Si no lo hace, el gate es NO-GO y no se escala.
- Reproducibilidad: mismo seed + mismo manifest → mismas métricas hasta
  tolerancia.
- Un checkpoint carga en frío desde safetensors y responde una decisión
  correcta en < 500 ms (el presupuesto del producto).
- El gate escribe `artifacts/gates/T-train-real/gate.json` con `pass`,
  accuracy/ECE seen vs unseen, tokens vistos y `model_version`.

## Done when

- Existe **un** checkpoint safetensors multi-dataset entrenado con pérdida
  listwise, no siete pickles.
- El job `train-decision` está registrado en el daemon y visible en Jobs.
- El dashboard muestra la métrica de etiquetas no vistas como titular.
- `train_baseline.py` está marcado como baseline histórico y no alimenta
  ningún gate verde.

## Resolution

**Entregado** — `training/python/train_decision.py` (+ `test_train_decision.py`,
18/18 verdes con la pila real, sin skips). Los 6 puntos del cuerpo están
cubiertos: pérdida listwise (cross-entropy sobre los `K + 1` logits de la
fila, `unknown` el último), un solo modelo sobre banking77 + massive +
huffpost + boolq, checkpoints `model.safetensors` + `tokenizer.json` +
`manifest.json` con `model_version` + `tokenizer_hash`, `metrics.jsonl` por
step en `artifacts/runs/train-real-v1/`, job del daemon `train-decision` y
`data/train_baseline.py` etiquetado como baseline histórico fuera de la ruta
de producto.

**El gate está en NO-GO y eso es el resultado, no un fallo del código.**
`artifacts/gates/T-train-real/gate.json` del primer punto de la curva
(stage 62 528 muestras) pasa 4 de 5 checks — `holdout_clean`, `label_free`,
`checkpoint_cold_load` (p95 43 ms, presupuesto 500 ms) y
`manifest_model_version` — y falla `unseen_beats_chance`: accuracy 0.058 en
etiquetas no vistas contra un azar de 0.165, con abstención 0.79. El criterio
se escribió antes de medir, así que el veredicto se queda escrito: no se
escala por encima del punto actual hasta que la curva lo bata.

El run sigue vivo (job `train-decision`, ~86 k muestras, loss 0.43–0.79,
accuracy seen 0.72–0.89) y reescribe `gate.json` en cada stage: los puntos
125 k / 250 k / 500 k / 1 M decidirán si el NO-GO se levanta.
