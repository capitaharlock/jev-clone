---
id: T-mix-1m
title: decision-mix-clean-1m y entreno top-2
status: done
priority: high
owner: unassigned
category: data
initiative: data-training
depends_on:
  - T-prog-gold
created: 2026-09-20
updated: 2026-09-22
completed_at: 2026-09-22T09:30:45.998Z
resolved_by: A003
resolved_by_conv: roadmap-architect-uwgjq
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

**No tenemos modelo capaz todavía. Y no es cuestión de más días de entrenamiento: es un fallo de arquitectura.**

La instancia de ayer **sí terminó** (`mix1m-curve`, exit 0 a las 01:41 UTC). El corpus de 1M quedó verde 10/10 y se entrenaron los dos backbones sobre 1 000 000 de filas reales. Pero los números dicen esto:

- **swag 0,255** contra azar 0,250 · **synth-v1 0,230** · **prog-gold 0,441** → cero aprendizaje en razonamiento.
- **dbpedia14 0,972 · civil-comments 0,942** → aprendió a emparejar texto, no a decidir.
- **La causa está en `run.json`: `backbone.frozen: true`.** Entrena un pointer head de 2 capas d=256 (~1,6 M params) sobre un encoder congelado de 149 M. Esa cabeza puede hacer similitud léxica; no puede hacer inferencia. Más filas no arreglan eso — el 1M ya se gastó y swag no se movió del azar.
- La métrica de producto (etiquetas no vistas: **0,220 no vistas vs 0,342 vistas, ECE 0,41**) está medida sobre el checkpoint **viejo** de 250k. El release gate sigue **NO-GO, 8 de 12 criterios**.

🚀 A029 → `#data-training` `#T-data-eval` · registrar la curva, **re-medir el gate de etiquetas no vistas sobre los dos modelos de 1M**, y medir un run corto con el backbone **descongelado** para confirmar la hipótesis con cifras.

<details><summary>Estado del sistema — qué hay y qué falta</summary>

**Verde y medido:**
- `#T-mix-1m` cerrada: gate 10/10 — fence limpio sobre 1 000 000 de preguntas (banking77/helpsteer2/pubmedqa vallados, 5 000 ids de Jevals excluidos), guardrails (`civil-comments` 14,99 %, ningún dataset sobre el cap), diversidad, ledger, reproducible.
- `synthetic_value`: veredicto **GO-scale** — el sintético aporta, medido con brazos con/sin sobre 250k.
- `top2_trained`: modernbert-base (3 h 46 min) y ettin-68m (2 h 42 min), 1 época exacta sobre el corpus, `members_sha256` idéntico al manifest.

**Rojo:**
- Release gate NO-GO: `unseen_accuracy_all_min`, `unseen_accuracy_cut_min`, `unseen_above_chance_every_cut`, `seen_unseen_accuracy_drop_max`, `seen_ece_max`, `teacher_cohen_kappa_min`, `coverage_min_at_risk`, `latency_p95_max_ms`.
- Auditoría de coherencia: 14 de 32 gates marcados incoherentes (métricas sin `model_version`).

**Nada entrenando ahora mismo** — solo el monitor en :8794. La GPU está libre.
</details>

<details><summary>¿Cuánto falta para un sistema utilizable?</summary>

No son días de entrenamiento, son estas tres cosas en orden:

1. **Descongelar el backbone** (horas de trabajo + un run de validación). Es la hipótesis que A029 va a medir. Si swag/synth-v1 suben del azar, el camino está claro.
2. **Re-medir el gate de producto** sobre los modelos de 1M — ahora mismo estamos juzgando el sistema por un modelo obsoleto y 4× más pequeño.
3. Solo entonces tiene sentido `#T-mix-5m` (ya lanzada su ingeniería) y `#T-data-eval`.

Gastar días de mps entrenando la cabeza congelada sobre 5M sería repetir el error de ayer a mayor escala: mucho coste, swag clavado en 0,25.
</details>

— T-mix-1m · el corpus de 1M se construye y se entrena de verdad, gate 10/10 verde

1.5M tokens
