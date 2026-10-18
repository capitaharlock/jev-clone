---
id: T-data-p0
title: P0 dataset adapters
status: done
priority: high
owner: unassigned
category: data
initiative: data-foundation
depends_on:
  - T-firewall
created: 2026-09-19
updated: 2026-09-20
completed_at: 2026-09-19T22:52:05.491Z
resolved_by: A004
resolved_by_conv: general-09192230
commit_shas: ['e830f0ff79b3ddbe0b3e66bf67102268f15f3abe', '2cf997920a0b3b7da07c498149aaed2c70f13782']
---
# P0 dataset adapters

HuffPost (con decisión documentada sobre su sesgo de titulares, §148),
Banking77, BoolQ, Civil Comments (V1: ramas boolean/choice; su rama
`score` diferida a V2) y HelpSteer2, más mixed loader con sampling.
Transformaciones gratuitas: shuffle/inserción de opciones, renombres
semánticos, typo-noise.

Mapeo V1: HuffPost solo smoke (`headline+description` → categoría),
Banking77 para labels semánticas dinámicas, BoolQ para choice binario,
Civil Comments para preguntas booleanas con slices de sesgo y HelpSteer2
como cinco preguntas choice 0–4/pairwise sobre atributos de una respuesta.
No se entrena un LM ni se predicen tokens.

Fuentes canónicas, mirrors, licencias y cautelas están en
`.meshkore/docs/source-register.md`. En particular, HuffPost se descarga del
original `rmisra/news-category-dataset` o del mirror `khalidalt/HuffPost`; el
id inexistente `HuggingFaceHug/HuffPost` queda descartado. Cada adapter fija
revisión y hash en su dataset card.

Fuente: plan §§29–40 (P0), 44.1, 148.

## Verification gate

- Requiere PASS de `T-firewall`; todo adapter pasa primero schema, licencia y
  contaminación, nunca sólo un smoke de parsing.
- Golden fixtures por dataset cubren conversión, splits, IDs, opción correcta,
  `unknown`, unicode y casos corruptos; el loader mixto se prueba con 2–42
  opciones y 1–N preguntas.
- Dos ejecuciones con la misma seed producen el mismo orden, batch y manifest;
  ningún ID cruza train/calibration/test.
- Un smoke-training CPU pequeño debe reducir loss sin NaN y el gate
  `T-data-p0` publica conteos, distribuciones y hashes.

## Done when

- 5 adapters convierten a schema universal con tests de dataset.
- Mixed loader reproduce el mismo batch con semilla fija.
- HuffPost queda marcado como smoke, no como evidencia de razonamiento, y no
  entra en pesos comerciales sin resolver su fence de procedencia.
- BoolQ conserva obligaciones CC-BY-SA y todos los datasets conservan la
  atribución exigida en manifests/model card.

## Resolution

#T-data-p0 lista con gate PASS: 5 adapters a schema universal con revisiones fijadas, transforms con semilla y mixed loader determinista. Cadena T-rust-skel → T-data-schema → T-firewall → T-data-p0 revalidada en un linaje (commits `2cf9979`, `e830f0f`).
<details><summary>data/adapters.py + fixtures + tests — qué incluye</summary>

- Adapters HuffPost (smoke §148), Banking77, BoolQ, Civil y HelpSteer2; filas corruptas rechazadas, `unknown`/unicode cubiertos.
- Pre-check schema+licencia+contaminación; fence: banking77/civil/helpsteer2 train-clear, huffpost y boolq gated con obligaciones en card.
- Mixed loader: mismo seed → mismo orden/batches/manifest; smoke CPU reduce loss sin NaN; summary en `artifacts/gates/T-data-p0/summary.json`.
- 12 tests nuevos (32/32 verdes); gate con paso `python-adapters`.
</details>

Siguiente en cola: rama eval (#T-recon → #T-calib) o modelo, según dependencias — digo cuál al seguir.

— T-data-p0 · 5 adapters P0, mixed loader determinista y gate PASS

**Commit** `e830f0ff7` (+1) · 15 files · 5.1M tokens

**Files changed (15):**
- `.meshkore/modules/data/tasks/T-data-p0.md`
- `artifacts/fixtures/p0/banking77.jsonl`
- `artifacts/fixtures/p0/boolq.jsonl`
- `artifacts/fixtures/p0/civil.jsonl`
- `artifacts/fixtures/p0/helpsteer2.jsonl`
- `artifacts/fixtures/p0/huffpost.jsonl`
- `artifacts/gates/T-data-p0/gate.json`
- `artifacts/gates/T-data-p0/summary.json`
- `artifacts/gates/T-data-schema/gate.json`
- `artifacts/gates/T-firewall/gate.json`
- `artifacts/gates/T-firewall/raw_results.json`
- `artifacts/gates/T-rust-skel/gate.json`
- `data/adapters.py`
- `data/test_adapters.py`
- `training/python/tools/gate.py`
