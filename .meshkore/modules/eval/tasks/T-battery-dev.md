---
id: T-battery-dev
title: Batería de desarrollo — 400 casos representativos, cinco familias, ES y EN
status: done
priority: high
owner: developer
category: eval
initiative: honest-eval
depends_on:
  - T-episode-contract
created: 2026-09-25
updated: 2026-09-26
completed_at: 2026-09-26T13:20:26.952Z
resolved_by: A048
resolved_by_conv: work-honest-eval-T-battery-dev-1790450100
commit_shas: ['3fb2eea86e356154447ac922b477cbd684963e0c']
---
# Batería de desarrollo — 400 casos representativos, cinco familias, ES y EN

Paso 1 del piloto. Hoy el proyecto no tiene ningún corte que represente las
decisiones del operador: BANKING77 mide 77 intenciones de banca, no preguntas
binarias sobre un estado ni comparaciones de producto. Sin este corte, «70 %» no
tiene sujeto.

400 casos de **desarrollo** —el corte contra el que se elige todo, y que por eso
se puede mirar tantas veces como haga falta— conformes a `#T-episode-contract`:

- Las **cinco familias** de decisión, con reparto declarado antes de construirlo.
- **ES y EN**, ambos con n suficiente para leer cada familia por separado.
- **K = 2 / 3 / 8**, el régimen del producto.
- Pares contrafactuales agrupados: original + variante decisiva + paráfrasis de
  control.

La **mezcla y los pesos se definen antes de evaluar nada**, y quedan versionados.
Un corte cuya composición se ajusta después de ver un resultado ya no mide.

Protocolo de información equivalente: cada modelo comparado recibe un **único
formato**, elegido sin consultar el test final. Si un modelo necesita hipótesis
explícitas y otro no, la diferencia de formato se documenta y se justifica; no se
elige el formato que más favorece a uno de ellos.

## Verification gate

- Test: los 400 casos pasan el validador de `#T-episode-contract`.
- Test: no hay `variant_group` compartido con la batería sellada de
  `#T-battery-sealed`.
- Test: el reparto real por familia × idioma × K coincide con la mezcla declarada
  (tolerancia escrita antes).
- Los desacuerdos de revisión están resueltos o el caso está fuera; no quedan
  casos ambiguos sin decidir.

## Done when

- Los 400 casos existen, versionados con su sha, revisados y con gold
  justificable caso a caso.
- La mezcla y los pesos están escritos y fechados **antes** de la primera
  medición.
- El formato de evaluación de cada modelo candidato está fijado y documentado.

## Resolution

✗ #T-battery-dev delivered: 400 cases, gate GO, committed `3fb2eee` — but the `eval/` suite run surfaced 1 pre-existing failure unrelated to this task (verbatim below).
400/400 pass the episode-v1 validator; family×lang×K matches the declared mix exactly; gold rule holds 400/400; 140 counterfactual groups; sealed-intersection pending (no sealed cut yet, prefix rule enforced by test).
Tests: battery 9/9, `data/` 420 passed + 6 skipped, ruff clean. `eval/`: 391 passed, 1 failed (pre-existing, no code path to my files).
Zero training/GPU used (all prose hand-authored, validator CPU-only, slowest step 108 s of pytest). Local commit only, no push. Integrity-hint fixes left untouched (other tasks' refs, outside this brief).

<details><summary>Files committed in 3fb2eee — 6 new + 1 task file</summary>

- [data/battery_dev.jsonl](/Users/ricartjuncadella/Documents/Prj/asimovia/jev-clone/data/battery_dev.jsonl) — 400 episodes, seeds 26001–26400
- [data/battery_dev_manifest.json](/Users/ricartjuncadella/Documents/Prj/asimovia/jev-clone/data/battery_dev_manifest.json) — mix declared and dated 2026-09-26, before any case
- [data/battery_dev.py](/Users/ricartjuncadella/Documents/Prj/asimovia/jev-clone/data/battery_dev.py) — auditor: gold rule, mix, namespace, sealed check, `--gate`
- [data/test_battery_dev.py](/Users/ricartjuncadella/Documents/Prj/asimovia/jev-clone/data/test_battery_dev.py) — 9 tests
- `artifacts/gates/T-battery-dev/gate.json` (verdict GO) + `gold_slots.json` (140 rule-derived slots)
- `.meshkore/modules/eval/tasks/T-battery-dev.md` — status next → done

</details>

<details><summary>Gate measured values (artifacts/gates/T-battery-dev/gate.json)</summary>

- battery sha `4d77217d7bb6`, schema episode-v1, manifest battery-dev-v1/2026-09-26
- n 400, valid 400, invalid 0, dupes 0; gold violations 0; shape 0; distribution 0; namespace 0; sealed `absent`, shared 0
- Mix: 5 families × 80, ES/EN 200/200, K 160/120/120; per cell K2 16 / K3 12 / K8 12; 120 triples + 20 pairs

</details>

<details><summary>Pre-existing eval failure (verbatim, not mine)</summary>

- `eval/test_release_gate.py::TestPublishedVerdict::test_the_numbers_are_the_ones_t_unseen_labels_published`
- `E AssertionError: 0.292141 != 0.304943` at `eval/test_release_gate.py:227`
- Evidence it predates me: `eval/release_gate.py` contains no `glob`/`listdir`/`battery` reference and never reads my files; my commit adds only new files. The worktree was already dirty with other agents' modifications before this turn.

</details>

**Commit** `3fb2eea86` · 7 files

**Files changed (7):**
- `.meshkore/modules/eval/tasks/T-battery-dev.md`
- `artifacts/gates/T-battery-dev/gate.json`
- `artifacts/gates/T-battery-dev/gold_slots.json`
- `data/battery_dev.jsonl`
- `data/battery_dev.py`
- `data/battery_dev_manifest.json`
- `data/test_battery_dev.py`
