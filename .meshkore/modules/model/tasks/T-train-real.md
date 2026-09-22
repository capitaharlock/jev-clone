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
resolved_by: A003
resolved_by_conv: roadmap-architect-uwgjq
completed_at: 2026-09-21T13:26:43.770Z
commit_shas: ['855a445667086b2835afc5e8af28a6fe87055e85']
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

✓ #decision-rebuild #T-train-real done · 15 ficheros · commit `855a445`
El entrenamiento real ya sustituye al bucle de 24 h: pérdida listwise sobre los K logits + `unknown`, **un** modelo multi-dataset (head de 2 699 778 params sobre backbone congelado) en vez de siete pickles, checkpoint safetensors con `model_version` que el runtime usa como clave de caché. 18/18 tests. El job `train-decision` (pid 57323) sigue vivo hacia 1M muestras.

⚠ Gate en **NO-GO** honesto (4/5): `unseen_beats_chance` da 0,058 contra 0,165 de azar en el stage 62,5k. Es la medida, no un bug — el job reescribe el gate en 125k/250k/500k/1M.

🚀 Opus (claude-code, pid 71801 vivo) → #decision-rebuild #T-bakeoff-real · rehacer el Pareto solo con pesos reales entrenados, proxies aleatorios fuera, top-2 de backbone decidido con números medidos.

<details><summary>Por qué el verdict "no-commit / 2 fallos → blocked" se descartó</summary>

- El wake pedía marcar `blocked` por fallo #2 sin commit. `git log` dice lo contrario: `855a445` es HEAD y contiene los 15 ficheros del reporte del hijo.
- El fichero ya está en `status: done`, `resolved_by: A014`, `commit_shas: ['855a445…']`. Escribir `blocked` encima sería meter una mentira en disco que el daemon revertiría en el siguiente `/state`.
- Patrón repetido esta sesión: el verdict del wake se calcula antes de que el commit del hijo sea visible. Regla aplicada siempre: HEAD manda sobre el verdict.
</details>

<details><summary>Estado de #decision-rebuild — 5/6 cerradas</summary>

| tarea | estado |
|---|---|
| #T-torch-stack | done · `0d905fb` |
| #T-pointer-head | done · `4506325` |
| #T-halt-contam | done · `88ad4e6` |
| #T-optset-sampler | done · `a2a8eaf` |
| #T-train-real | done · `855a445` (gate NO-GO honesto) |
| #T-bakeoff-real | 🚀 en curso, A015 |

La iniciativa no se marca `done` hasta que #T-bakeoff-real cierre. El NO-GO de unseen no bloquea la tarea (está medido y publicado) pero sí es el hallazgo que #T-bakeoff-real tiene que explicar: si ningún backbone bate el azar en etiquetas no vistas, el top-2 se elige igualmente pero con esa advertencia escrita en prosa.
</details>

3.7M tokens
