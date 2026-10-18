---
id: T-massive-huff
title: MASSIVE completo, HuffPost y cierre P0
status: done
priority: high
owner: unassigned
category: data
initiative: data-training
depends_on:
  - T-bigsrc
created: 2026-09-20
updated: 2026-09-20
completed_at: 2026-09-20T13:01:24.091Z
resolved_by: A004
resolved_by_conv: general-09192230
commit_shas: ['4c2f504ab6bca9cab77b7f8d63fd20dbf1d38584', 'd2e26744deaeec749d7b32bafe56c4be49fa1f78']
---
# MASSIVE completo, HuffPost y cierre P0

Cierra el frente "real labels" (§§26–31, 84, 122–126): MASSIVE completo
(>1 M utterances, 52 lenguas; primera expansión EN+ES+FR+DE+PT+IT, el
resto sólo por decisión, §26), HuffPost completo (~210 k; investigar
por qué el local trae 66.510 — subset, split, adapter o mirror —
y corregir o documentar, §§31, 125), y terminar adapters LogiQA 2.0
y ReClor pendientes (§124; enseñar decisión sin generar explicación).
BoolQ se mantiene con decisión de licencia Share-Alike; Civil Comments
se mantiene pero con cap 100–300 k por ciclo (§§27, 84, 126);
NLI/zero-shot-label y DPO pairs entran sólo como research hasta
auditoría de provenance (§§23–24).

Fuentes: data-training §§23–31, 84–85, 122–126.

## Verification gate

- Requiere PASS de `T-bigsrc`; cada fuente ampliada trae manifiesto
  con rows antes/después del filtro y justificación del delta
  (caso HuffPost 66 k vs 210 k resuelto por escrito).
- Tests: adapter MASSIVE conserva intent + descripciones + idioma y
  pasa round-trip ES↔EN + code-switching (§59); LogiQA/ReClor
  convierten state/question/options/gold sin almacenar explicaciones;
  el cap de Civil se verifica contando su % en un mix de prueba
  (falla si > 15 %, §§65–66); BoolQ marcado con su obligación
  Share-Alike en la rama commercial (§99).
- El gate escribe `artifacts/gates/T-massive-huff/gate.json` con
  `pass: true` y matriz fuente×licencia×split; sin ese artifact no
  arranca `#T-synth-factory`.

## Done when

- MASSIVE completo + HuffPost completo + LogiQA/ReClor cerrados en
  parquet canónico con manifests.
- Informe HuffPost 66 k vs 210 k escrito; Civil con cap aplicado y
  verificado; DPO/zero-shot-label en `research-only/` hasta auditoría.
- Layer A (labels humanas/expertas, §61) congelada como base del mix.

## Resolution

Dataset y entrenamiento, ambos en marcha: #T-massive-huff cerrada con gate PASS en #data-training (commits `d2e2674` + cierre) y baseline v4 recién entrenado sobre los 8 sets. El conversor Qwen sigue vivo en fondo (offset 825 en boolq). Siguiente en la cadena: #T-synth-factory.

<details><summary>HuffPost 66k vs 210k — resuelto</summary>

- El adapter recortaba a 8 de 41 categorías y tiraba 134.343 filas. Universo ampliado a las 41 (FOOD heredado queda como distractor 42).
- Resto del delta (19.713) son filas degeneradas de origen (titular/descripción vacíos), verificado por re-escaneo: 0 categorías desconocidas. 181.140 ejemplos convertidos.
</details>

<details><summary>MASSIVE + LogiQA/ReClor — cerrados</summary>

- MASSIVE en 6 locales (en-US/es-ES/fr-FR/de-DE/pt-PT/it-IT): 99.126 ejemplos.
- LogiQA 84.935/84.976: el dump mezcla 4 formas internas (MC + 2 NLI); las NLI van a booleano yes/no con la convención de bigsrc. 4 truncadas + 37 degeneradas, documentado.
- ReClor 6.138/6.138: el split test trae gold oculto (`-1`) → `answer: unknown`, nunca entrena. Sin campos de explicación (escaneo de nombres de campo, no de prosa).
- Cap Civil 15 % verificado en ambos sentidos; matriz fuente×licencia×split en el gate.
</details>

<details><summary>Baseline v4 — 8 sets, métricas</summary>

- boolq 0,68 · helpsteer2 0,32 · civil 0,93 · huffpost 0,62 (163k train) · banking77 0,85 · massive 0,79 (69k) · logiqa 0,44 (64k) · reclor 0,25 (4,6k, esperado: set pequeño y duro).
- Log en `artifacts/runs-baseline-v4.log`, métricas en `artifacts/runs/20260920T125723Z/metrics.json`. El trainer ahora absorbe logiqa+reclor y excluye `unknown`.
</details>

<details><summary>Verificación y estado continuo</summary>

- fmt + clippy + 60 tests Rust + 19 suites Python en verde antes del cierre.
- Entrenamiento continuo: v4 corre al cierre de cada tanda de datos nuevos; el Qwen sigue aumentando en fondo sin bloquear. Di: qué tarea sigue y la ejecuto en este mismo flujo.
</details>

**Commit** `4c2f504ab` (+1) · 13 files · 28M tokens

**Files changed (13):**
- `.meshkore/modules/data/tasks/T-massive-huff.md`
- `artifacts/data-prefetch/huffpost.jsonl`
- `artifacts/data-prefetch/logiqa.jsonl`
- `artifacts/data-prefetch/manifest.json`
- `artifacts/data-prefetch/massive.jsonl`
- `artifacts/data-prefetch/reclor.jsonl`
- `artifacts/gates/T-massive-huff/DATA_AUDIT.md`
- `artifacts/gates/T-massive-huff/gate.json`
- `data/adapters.py`
- `data/convert_logiqa_reclor.py`
- `data/convert_massive.py`
- `data/test_massive_huff.py`
- `data/train_baseline.py`
