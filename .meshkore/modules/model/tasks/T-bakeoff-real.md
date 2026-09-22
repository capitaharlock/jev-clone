---
id: T-bakeoff-real
title: Bake-off con pesos reales y retirada de los proxies del Pareto
status: done
priority: medium
owner: unassigned
category: model
initiative: decision-rebuild
depends_on:
  - T-train-real
created: 2026-09-21
updated: 2026-09-21
failed_at: 2026-09-21T13:56:50.962Z
resolved_by: A016
resolved_by_conv: work-decision-rebuild-T-bakeoff-real-1790020
completed_at: 2026-09-21T13:58:00.278Z
commit_shas: ['d5bcf79610f0140c0925431337f770b08f3cddde']
---
# Bake-off con pesos reales y retirada de los proxies del Pareto

`artifacts/gates/T-bakeoff/report.json` compara hoy encoders hash char-ngram
con pesos aleatorios (dim 256/1024/4096) y lista Ettin-68M, ModernBERT-base,
LFM2.5-230M y NeoBERT-250M como `pending_weights`. La honestidad del
artefacto es de agradecer —nunca se inventaron números— pero la decisión de
backbone sigue sin tomarse.

Con `#T-torch-stack` (pesos descargados) y `#T-train-real` (entrenamiento que
funciona), el bake-off se rehace midiendo lo que importa:

- Métrica primaria: accuracy + ECE **sobre etiquetas no vistas**
  (`#T-unseen-labels`), no accuracy de fila.
- Latencia p50/p95 por decisión con K=4 y estado realista, en MPS y CPU,
  dentro del presupuesto 70-500 ms que define el producto.
- Coste: parámetros, memoria en inferencia, tiempo de entreno hasta el corte.
- Pareto limpio: **ningún proxy aleatorio en la gráfica**. Los proxies se
  mueven a una sección "sanity checks" claramente separada o desaparecen.

Alcance mínimo real: Ettin-68M y ModernBERT-base entrenados de verdad.
LFM2.5-230M y NeoBERT-250M entran solo si sus pesos y licencia lo permiten;
si no, se declaran `not_available` con el motivo, como ya se hace.

## Verification gate

- Cada fila del Pareto tiene un `weights_sha256` y un `run_id` de
  `#T-train-real` detrás; una fila sin linaje hace fallar el gate.
- Test que rechaza la publicación si algún backbone del report tiene
  `pending_weights` o `random_init`.
- El gate reescribe `artifacts/gates/T-bakeoff/report.json` con `pass: true`
  y el veredicto top-2 escrito en prosa.

## Done when

- Top-2 de backbone elegido con números medidos, no proxies.
- El report publica accuracy/ECE unseen + latencia p95 + coste por candidato.
- La decisión y su porqué están en `.meshkore/docs/model-card.md`.

## Resolution

✓ task #T-bakeoff-real done. files: 4. commit: d5bcf79. tests: 20/20. gate: pass=true 6/6.

<details><summary>gate artifacts/gates/T-bakeoff/report.json — verified, unmodified</summary>

- `pass: true`; 6/6 checks pass with empty `offenders` (latency_budget, measured_not_imputed, minimum_real_scope, no_proxy_in_pareto, pareto_lineage, top2_stated)
- `top2` = `["modernbert-base","ettin-68m"]`
- `pareto.rows`: 2 rows, both `status: trained`, neither carries pending_weights/random_init/harness_only
- each row's `lineage` names run_id + checkpoint + backbone `weights_sha256` (per-file) + `head_weights_sha256`
- the 3 random-init proxies live only under `sanity_checks.harness_proxies` with `pareto_eligible: false`
</details>

<details><summary>staging — 4 files, no noise</summary>

Staged exactly: `model/bakeoff.py`, `model/test_bakeoff.py`, `artifacts/gates/T-bakeoff/report.json`, `.meshkore/modules/model/tasks/T-bakeoff-real.md` (`status: blocked` → `done`, nothing else touched).

Left uncommitted deliberately — unrelated to the bake-off: `target/`, `.DS_Store`, other tasks' `.md` and gate JSONs (T-halt-contam, T-mix-1m, T-optset-sampler, T-pointer-head, T-torch-stack, T-train-real, T-cloud-api, T-data-p0, T-option-mixer, T-release, T-state-cache, data-prefetch manifest, `.meshkore/team/*`).

Not pushed.
</details>

— T-bakeoff-real · el Pareto del bake-off se reconstruye con pesos entrenados reales (commit d5bcf79)

**Commit** `d5bcf7961` · 4 files · 496k tokens

**Files changed (4):**
- `.meshkore/modules/model/tasks/T-bakeoff-real.md`
- `artifacts/gates/T-bakeoff/report.json`
- `model/bakeoff.py`
- `model/test_bakeoff.py`
