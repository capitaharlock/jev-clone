---
id: T-optset-sampler
title: Sampler de opciones dinámicas y hard negatives sobre el corpus ya convertido
status: done
priority: high
owner: unassigned
category: data
initiative: decision-rebuild
depends_on:
  - T-halt-contam
created: 2026-09-21
updated: 2026-09-21
completed_at: 2026-09-21T12:50:31.811Z
resolved_by: A012
resolved_by_conv: work-decision-rebuild-T-optset-sampler-1789994631
commit_shas: ['a2a8eafbed33f7c370b8ddf0f1b875a88e35af98']
---
# Sampler de opciones dinámicas y hard negatives sobre el corpus ya convertido

El material bueno ya está en disco: los converters (`convert_banking77.py`,
`convert_massive.py`, `convert_huffpost.py`, `convert_logiqa_reclor.py`)
preservan estado, opciones y textos correctamente. Lo que falta es el
*dataloader* que alimenta a `#T-pointer-head` con el contrato correcto.

Hallazgo B, segunda mitad: **HuffPost convierte a 4 candidatos por fila, pero
el trainer entrena sobre las 41 categorías globales**, destruyendo el
contrato de opciones dinámicas. Ese es exactamente el fallo que este sampler
impide que vuelva a ocurrir.

Trabajo, en `data/optset.py`:

1. Por fila, muestrear **K opciones dinámicas** (K variable, no fijo) que
   incluyan siempre la correcta y K-1 distractores.
2. **Hard negatives** por similitud, reutilizando `data/hardneg.py` y
   `leakage.jaccard`: el distractor plausible (`card_arrival` vs
   `card_delivery_estimate` en banking77, categorías hermanas en HuffPost)
   pesa más que el aleatorio. Mezcla explícita fácil/difícil, medida.
3. Filas `unknown`: una fracción del batch donde la respuesta correcta **no
   está** entre las opciones ofrecidas, para entrenar la salida `unknown`
   de verdad y no como umbral.
4. Barajado del orden de las opciones en cada epoch (alimenta el test de
   invariancia de `#T-pointer-head`).
5. Cobertura inicial: banking77, massive, huffpost, boolq — lo ya convertido
   y limpio. Nada de synth-loop, nada de logiqa/reclor (eval-only).

El sampler NO reescribe los jsonl: lee el schema V1 tal cual y construye los
batches. `data/schema.py` se conserva intacto.

## Verification gate

- Test: ninguna muestra emitida tiene el conjunto de opciones igual al
  espacio global de etiquetas del dataset (el bug de HuffPost no puede
  reproducirse).
- Test: la opción correcta aparece en posición uniforme (chi-cuadrado sobre
  10 k muestras) — sin prior posicional como el que dio 0,435 en LogiQA.
- Test: la fracción de filas `unknown` y la mezcla hard/easy son parámetros
  verificados por conteo, no implícitos.
- El gate escribe `artifacts/gates/T-optset-sampler/gate.json` con
  `pass: true`, distribución de K, % hard negatives y % `unknown` por
  dataset.

## Done when

- `data/optset.py` emite batches con K variable, distractores duros y filas
  `unknown`, sobre banking77 + massive + huffpost + boolq.
- Los tres tests del gate pasan y el artifact publica la composición real.
- Está documentado que el trainer nunca vuelve a ver un espacio de etiquetas
  global.

## Resolution

✓ task T-optset-sampler done. files: 3. commit: a2a8eaf. verification: 23/23 tests pass, gate `pass: true` (5/5 checks), end-to-end contract confirmed against the real pointer head.

`data/optset.py` reads the converted V1 jsonl read-only (banking77/massive/huffpost/boolq; `schema.py` untouched) and emits `Sample`s with variable K (3–8, clamped by pool), gold + K-1 hard/easy distractors, 12% gold-absent rows, and per-epoch option reshuffling. Blocked corpora are refused twice: allowlist + `firewall.check_job_allowed` (raises).

<details><summary>data/optset.py — the sampler (688 lines)</summary>

- `SamplerConfig`: k_min/k_max, `hard_fraction`, `unknown_fraction`, `hard_top_ratio`, `cos_weight`, split, seed — every knob the gate reports is a field, not a constant.
- `difficulty()` = `hardneg._embed` char-3gram cosine blended with `leakage.jaccard` over the de-underscored label. Neither alone covers both label styles: cosine catches `PARENTS`/`PARENTING`, Jaccard catches `card_arrival`/`card_delivery_estimate`.
- `DistractorIndex` ranks the pool per gold and cuts it into hard (top 25%, min 4) and easy buckets; the sampler draws `round(hard_fraction × n_distract)` from hard and the rest from easy, then counts what it actually drew.
- Option **set** is seeded by `(seed, row, question)` → stable curriculum; option **order** by `(seed, epoch, row, question)` → reshuffled every epoch, `gold_index` re-derived after the shuffle.
- Emitted contract matches `DecisionEngine.logits(mem, question, options)`; `gold_index == K` is the head's `unknown` slot. `dropped_gold` records the withheld gold so "gold really absent" is checkable, not assumed.
- Stdlib chi-square (`_gammainc_upper` / `chi2_sf`) — no scipy.
</details>

<details><summary>Gate — artifacts/gates/T-optset-sampler/gate.json, pass: true</summary>

16 000 samples (4 000 rows × 4 datasets), 1.6 s.

- `no_global_label_space`: 0 violations. Pool sizes 77/60/41/2; max K = 8 everywhere. **boolq is exempt and says so**: a 2-label pool IS the question, so its option set cannot differ from its global space — the exemption and its reason are in the gate, not hidden.
- `gold_position_uniform`: 14 019 answerable samples, pooled χ²=32.3 df=25 **p=0.190**; per-K p ∈ [0.089, 0.827]. A control test plants "gold always last" and the same function rejects it (p < 1e-6).
- `mix_counted`: 12.38% unknown (target 12%), 55.05% hard negatives overall — 51.7/51.9/51.9% on banking77/massive/huffpost (target 50%), 100% on boolq where only one distractor exists (counted, not targeted).
- `epoch_shuffle`: set stable across epochs 0→1, 87% reordered, gold moves.
- `firewall`: synth-loop / logiqa / reclor all raise.
- K distribution (global): 1:466, 2:3534, 3:2046, 4:1966, 5:2013, 6:1943, 7:2040, 8:1992.
</details>

<details><summary>Verification actually executed</summary>

```
python3 -m unittest data.test_optset -v   → Ran 23 tests, OK
python3 -m data.optset gate               → 5/5 PASS, pass: true
python3 -m unittest data.test_firewall data.test_hardneg data.test_data_schema → 36 tests, OK
ruff check data/optset.py data/test_optset.py → All checks passed!
```
Plus a real end-to-end run on `.venv-train` (torch 2.14 + ettin-68m): `DecisionEngine.logits` on sampler output returned `[K+1]` logits for K=5 and K=2, `cross_entropy` against `gold_target()` finite for both an answerable and an `unknown` row.

Note: `ruff format` would reformat these files, but the repo is not ruff-format clean (hardneg.py, decision_head.py also differ) and `ruff check data/ model/` has 12 pre-existing errors in other files — untouched, none in mine.
</details>

— T-optset-sampler · el dataloader entrega opciones dinámicas con hard negatives y filas `unknown`, y el gate lo demuestra contando

**Commit** `a2a8eafbe` · 3 files · 2.6M tokens

**Files changed (3):**
- `artifacts/gates/T-optset-sampler/gate.json`
- `data/optset.py`
- `data/test_optset.py`
