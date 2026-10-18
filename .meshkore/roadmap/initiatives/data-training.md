---
reactivated: 2026-09-22
id: data-training
title: Data training scale-up (decision-mix)
status: done
owner: architect-master
modules:
  - data
created: 2026-09-20
updated: 2026-09-22
completed_at: 2026-09-22T10:18:51.542Z
commit_sha: 0ecbd528efbe71e12376b8de0ecd7814df0ea53b
---
# Data training scale-up (decision-mix)

Ejecuta la estrategia `TMP/JEVCLONE_DATA_TRAINING_STRATEGY_2026-09-20.md`:
convertir el POC actual (~2,24 M filas, ~89 % Civil Comments) en un corpus
de decisiones benchmark-clean, diverso y calibrado. Tres frentes en una
sola cadena serial: (1) fuentes masivas reales con output cerrado
(Tasksource Instruct, P3 muestreado, DocNLI, MASSIVE completo, HuffPost
completo, LogiQA/ReClor); (2) synthetic factory con council local
(Qwen+DeepSeek, roles invertidos) + gold programático (Wikidata);
(3) mixes versionados `decision-mix-clean-1m` → 5 M con curva de scaling
y reportes públicos de generalización, calibración y latencia.

Principios no negociables del doc: diversidad > volumen (§§15, 64–66);
grounded synthetic > self-generation (§36); benchmark-clean excluye
Banking77/HelpSteer2/PubMedQA del entreno (§§17–18, 84, 118);
≤50 % synthetic hasta demostrar ganancia (§110); Civil Comments con
sampling cap (§§27, 126); nunca 50 M antes de que 1 M → 5 M → 10 M
mejore (§§0, 89).

Tasks (secuenciales): `#T-bigsrc` → `#T-massive-huff` →
`#T-synth-factory` → `#T-prog-gold` → `#T-mix-1m` → `#T-mix-5m` →
`#T-data-eval`. Rama backlog: `#T-mix-10m` (10 M/20 M + FLAN amplio +
Dolma, sólo si la curva sigue abierta).

Fuentes: data-training §§19–31 (fuentes), §§32–45 (synthetic/validación),
§§61–69 (capas, mixes, sampler), §§76–83 (evals), §§116–140 (plan de 25
tasks del agente, plegadas aquí en 7 activas + 1 backlog).

## Done when

- `decision-mix-clean-1m` existe versionado con manifests, licencias
  auditadas y cero overlap con Jevals; top-2 backbones entrenados.
- El mix de 5 M sólo existe si 1 M demuestra generalización; el piloto
  de distillation soft sólo escala si el held-out mejora.
- Reportes `JEVALS_REPORT.md`, `GENERALIZATION_REPORT.md`,
  `CALIBRATION_REPORT.md`, `LATENCY_REPORT.md` publicados desde splits
  sellados; ningún gate PASS con Civil > 15 % de un mix.

## Cierre (2026-09-22)

El work-stream de **corpus** está entregado: 12 tasks en `done`, el mix
`decision-mix-clean-1m` construido y fenced, y la curva de escalado medida.
Su último hallazgo es un NO-GO: `#T-mix-5m` registró que 1 M no demuestra
generalización, así que multiplicar filas no compra nada.

Dos tasks salen de aquí porque no son corpus:

- `#T-data-eval` (OOD, calibración y reportes públicos) → `#honest-eval`:
  es evaluación, y su sitio natural es la iniciativa que decide qué significa
  un número verde.
- `#T-mix-10m` (10 M/20 M) → `#train-scaleout`: escalar está bloqueado por el
  NO-GO de `#T-mix-5m`, no por falta de receta.

Lo que falta ahora no son datos: es `#generalization-fix`.
