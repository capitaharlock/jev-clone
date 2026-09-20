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
completed_at: 2026-09-20T12:04:03Z
resolved_by: A004
resolved_by_conv: general-09192230
commit_shas: ['ee96d6ffc70f52e246eee6c9d3c96c7db486693e']
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
