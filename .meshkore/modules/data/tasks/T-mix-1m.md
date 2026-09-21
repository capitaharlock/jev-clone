---
id: T-mix-1m
title: decision-mix-clean-1m y entreno top-2
status: active
priority: high
owner: unassigned
category: data
initiative: data-training
depends_on:
  - T-prog-gold
created: 2026-09-20
updated: 2026-09-20
failed_at: 2026-09-21T16:48:22.933Z
resolved_by: A027
resolved_by_conv: work-data-training-T-mix-1m-commit-1790012000
---
# decision-mix-clean-1m y entreno top-2

Primer corpus nuevo versionado (§§86, 97, 135): `decision-mix-v1`
objetivo 5 M, primer corte limpio `decision-mix-clean-1m` (~1,5 M
human/expert + 0,5 M NLI + 0,5 M intent/multilingüe + 0,5 M
preference/ordinal + 1,0 M programmatic + 0,7 M grounded synthetic +
0,3 M adversarial/OOD, §86) con guardrails (≤15 % por dataset,
≤30 % por familia, ≤50 % synthetic, ≥20 % human, ≥10 % hard/OOD,
§§65–66) y mix de tipos Choice 55 / Noul 30 / Score 15 (§48).
Entrenar top-2 backbones del bake-off; losses simples: CE+Brier
(+KL teacher opcional), ordinal/RPS para Score, auxiliar
trust/unknown, consistencia de permutación (§95); hard y soft labels
conviven sin sobrescribir el gold humano (§96). Incluye la evaluación
de calidad sintética: baseline vs +50 k; si no mejora held-out, NO
escalar el generador (§128).

Fuentes: data-training §§48, 61–68, 86, 95–97, 111, 128, 135.

## Verification gate

- Requiere PASS de `T-prog-gold`; el mix falla como error si un
  dataset supera el 15 % o Civil domina (§65: "nunca 89 % sin que el
  pipeline lo marque como error").
- Tests: dashboard de diversity index por domain/familia/tipo/K/idioma
  (§65); guardrails verificados por conteo; benchmark-clean fence:
  cero Banking77/HelpSteer2/PubMedQA/Jevals-hashes en train (§§18,
  77); tokens medidos por shard (examples, state/Q/candidate tokens,
  mean K/Q, §63); reproducibilidad por seed+manifest.
- El gate escribe `artifacts/gates/T-mix-1m/gate.json` con `pass: true`,
  composición final, tokens totales y métricas top-2 backbones
  (accuracy, NLL, Brier, ECE, §113); sin ese artifact no arranca
  `#T-mix-5m`. Si el synthetic no aporta, gate NO-GO con fallback
  documentado (corregir prompts, §128).

## Done when

- `decision-mix-clean-1m` publicado con manifests y diversity
  dashboard; Stage 0 (250 k) y Stage 1 (1 M) de la curva (§62) medidos.
- Top-2 backbones entrenados y comparados; veredicto synthetic
  escrito (escala o se corrige).

## Corpus contract (A028, 2026-09-21)

`decision-mix-clean-1m` is ONE recipe, held in `data/mix.py`:
`CLEAN_1M_TARGET = 1_000_000`, `CLEAN_1M_SEED = 20260922`,
`CLEAN_1M_CAP_MARGIN = 0.0001`. `tools/mix_1m/run_mix.py` publishes the
manifest from those three and `train_decision --fence-clean` assembles
from the same three, so both land on one `members_sha256`.

Before this they were two paths: the trainer defaulted its mixture
target to the cap CEILING and planned at `cap_margin=0.0`, so
`--max-samples 1000000` built a 39 981-row mixture and looped it ~25
times while `run.json` reported a 1 M-decision curve.

The loop is now an abort. `--max-samples` is decisions SEEN; if the
realised mixture holds fewer, the trainer raises `MixShortfallError`.
`--allow-repeat` is the only way to loop, and it publishes
`epochs_over_corpus` in `run.json`, `metrics.jsonl` line 0 and the run's
`mix.json`.

The §128 ablation arms lost their 18 % cap exception with the widening:
without `synth-v1` the registry now covers 1.65 cap units, so both arms
build at the §§65-66 caps over 250 000 rows for a 250 000-decision
budget — one pass each, not 6.25.

## Resolution

**Failed — exit 143.**

Now I'll write the converter for the four new sources.Now the full 1M build (this streams ~2.5 GB, so I'll run it in the background):

# architect: 2026-09-21 — A026 widened the registry and the 1M corpus builds
# (gate: 8/10 checks green, measured_rows=1000000). It died on SIGTERM before
# committing; A027 is committing that work. The last two checks (synthetic_value,
# top2_trained) are owned by the live job `mix1m-curve` (pid 11028) — nobody
# relaunches training, the task closes when that curve lands its verdicts.
