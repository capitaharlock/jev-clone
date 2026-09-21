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
