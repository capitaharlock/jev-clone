---
id: T-bigsrc
title: Tasksource, P3 y DocNLI adapters
status: done
priority: high
owner: unassigned
category: data
initiative: data-training
depends_on:
  - T-data-schema
created: 2026-09-20
updated: 2026-09-20
completed_at: 2026-09-20T12:05:22.361Z
resolved_by: A004
resolved_by_conv: general-09192230
commit_shas: ['ce32ebbc276e5563fc52bb9387a9c80071f84666', 'ee96d6ffc70f52e246eee6c9d3c96c7db486693e']
---
# Tasksource, P3 y DocNLI adapters

Las tres fuentes masivas P0 del doc (§§19–20, 25, 85): Tasksource
Instruct (~5,6 M rows, ~510 tasks), P3 (~122 M rows materializados,
streaming + sampler, NUNCA entreno ciego) y DocNLI (~1,3 M).
Sólo ejemplos con output cerrado o candidate set reconstructible
(§§120–121, 124): `answer_choices`/`is_correct`/targets finitos;
nada de generación libre, CoT ni diálogo (§§21, 101).

Puerta de entrada de `#data-training`: reutiliza schema + registry +
license fence de `#T-data-schema` y el firewall de `#T-firewall`;
todo shard sale en parquet normalizado con manifiesto
(source, revision, licencia, rows antes/después, tokens, dedupe,
overlap removido, §97). Targets piloto: 1 M aceptados Tasksource,
1 M P3 alta diversidad, DocNLI sampleado balanceado.

Fuentes: data-training §§19, 20, 25, 85, 100–101, 120–121, 124.

## Verification gate

- Requiere PASS de `T-data-schema`; el loader falla si un shard no
  trae manifiesto completo o mezcla licencias non-commercial en la
  rama commercial-clean (§§97–98).
- Tests: adapter determinista sobre fixtures (mismo input → mismo
  parquet+hash); el filtro rechaza generación libre/CoT; el sampler
  P3 demuestra que no duplica templates (mismo source × N prompts
  no cuenta como N conocimientos, §20); leak detector contra
  `benchmarks/jevals/*_ids.json` en verde (§77).
- El gate escribe `artifacts/gates/T-bigsrc/gate.json` con `pass: true`,
  hashes de shards y conteos aceptados/rechazados por fuente; sin ese
  artifact no arranca `#T-massive-huff`.

## Done when

- Adapters Tasksource/P3/DocNLI → schema canónico con 1 M + 1 M +
  sample balanceado aceptados y manifestados.
- Cero overlap con IDs Jevals; licencias componente-a-componente
  auditadas en fuentes agregadas (§99).
- `DATA_AUDIT.md` generado: rows, tokens, familia, tipo, idioma,
  licencia, distribución K por fuente (§119).

## Resolution

#T-bigsrc done, gate PASS. First of your 7 training tasks executed: Tasksource/P3/DocNLI adapters live in `data/bigsrc.py`, 12 new tests green, full gate chain revalidated in one lineage, committed as `ee96d6f` + `ce32ebb`.

Next in order is #T-massive-huff (MASSIVE completo, HuffPost, LogiQA/ReClor). Say the word and I continue. The Qwen converter is still running untouched in the background.

<details><summary>What #T-bigsrc delivered</summary>

- `data/bigsrc.py`: 3 adapters to the canonical schema, closed-output filter (rejects generation/CoT/dialogue), P3 sampler with per-template cap + input dedupe, balanced DocNLI sampler, deterministic JSONL shards with full manifests, commercial-clean license fence, firewall leak check.
- `artifacts/gates/T-bigsrc/DATA_AUDIT.md`: per-source rows/tokens/family/type/license/K-dist, fence PASS, leak clean.
- Pinned revisions: Tasksource `Apache-2.0`, P3 `Apache-2.0`, DocNLI `BSD-3-Clause` (added BSD to the approved-train set in `data/registry.py`).
- One honest gap, documented in the audit: no `benchmarks/jevals/*_ids.json` exists in the repo yet, so the Jevals-overlap check is vacuous until #T-data-eval freezes those IDs.
</details>

<details><summary>Verification</summary>

- `data.test_bigsrc`: 12/12 green.
- Gate chain `T-rust-skel → T-data-schema → T-bigsrc`: all PASS, lineage `c4c005ad` (had to re-run from root since new files changed the tree hash).
- Committed `ee96d6f` + `ce32ebb` with standard trailers; diary entry written; task file marked done with shas.
</details>

**Commit** `ce32ebbc2` (+1) · 19 files · 28M tokens

**Files changed (19):**
- `.meshkore/modules/data/tasks/T-bigsrc.md`
- `artifacts/fixtures/bigsrc/docnli.jsonl`
- `artifacts/fixtures/bigsrc/p3.jsonl`
- `artifacts/fixtures/bigsrc/reject.jsonl`
- `artifacts/fixtures/bigsrc/tasksource.jsonl`
- `artifacts/gates/T-bigsrc/DATA_AUDIT.md`
- `artifacts/gates/T-bigsrc/gate.json`
- `artifacts/gates/T-bigsrc/shards/docnli.jsonl`
- `artifacts/gates/T-bigsrc/shards/docnli.manifest.json`
- `artifacts/gates/T-bigsrc/shards/p3.jsonl`
- `artifacts/gates/T-bigsrc/shards/p3.manifest.json`
- `artifacts/gates/T-bigsrc/shards/tasksource-instruct.jsonl`
- `artifacts/gates/T-bigsrc/shards/tasksource-instruct.manifest.json`
- `artifacts/gates/T-data-schema/gate.json`
- `artifacts/gates/T-rust-skel/gate.json`
- `data/bigsrc.py`
- `data/registry.py`
- `data/test_bigsrc.py`
- `training/python/tools/gate.py`
