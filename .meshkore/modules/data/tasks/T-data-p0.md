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
completed_at: 2026-09-20T00:00:00.000Z
resolved_by: A004
resolved_by_conv: general-09192230
commit_shas: ['2cf9979']
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

#T-data-p0 ejecutada con gate PASS en el linaje `02414ff6` (commit `2cf9979`): 5 adapters a schema universal con revisiones fijadas, transforms gratuitas con semilla, mixed loader determinista y summary con conteos/distribuciones/hashes en `artifacts/gates/T-data-p0/summary.json`. Cadena revalidada T-rust-skel → T-data-schema → T-firewall → T-data-p0, todo PASS.
<details><summary>data/adapters.py + fixtures + tests — qué incluye</summary>

- Adapters HuffPost (smoke §148), Banking77, BoolQ, Civil Comments y HelpSteer2 con cards fijadas (revisión inmutable + SHA-256) y rechazo de filas corruptas.
- Pre-check schema+licencia+contaminación por adapter; fence: banking77/civil/helpsteer2 train-clear, huffpost y boolq gated con obligaciones documentadas.
- Mixed loader con pesos y semilla: mismo seed → mismo orden, batches y manifest; IDs disjuntos entre splits; smoke CPU reduce loss sin NaN.
- 12 tests nuevos (32/32 verdes con los suites previos); gate extendido con paso `python-adapters`.
</details>

— T-data-p0 · 5 adapters P0, mixed loader y gate PASS (commit 2cf9979)
