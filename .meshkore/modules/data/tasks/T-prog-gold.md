---
id: T-prog-gold
title: Gold programático, hard negatives y packs multi-Q
status: done
priority: high
owner: unassigned
category: data
initiative: data-training
depends_on:
  - T-synth-factory
created: 2026-09-20
updated: 2026-09-21
commit_shas:
  - 487a6ab8afe5daf43c0aaf9b3e4c8aec97e8fda4
resolved_by: A004
resolved_by_conv: general-09192230
---
# Gold programático, hard negatives y packs multi-Q

La fuente de labels más fiable (§60): generador Wikidata (CC0, >120 M
items) donde el GRAFO da el gold y los teachers sólo redactan
pregunta/descripciones — nunca deciden la correcta (§§35, 104).
Piloto 100 k: entity/property, boolean relacional, hard candidates del
mismo tipo (§129). Encima: hard-negative miner por embedding nearest
labels + adversario teacher (§§46, 130); sampler de cardinalidad K
variable con mix 35/25/20/10/6/4 (§§47, 131); packer multi-question
3–10 Q por state para entrenar encode-once (§§8, 58, 132); transforms
de consistencia (shuffle, paráfrasis, contexto irrelevante) con peso
controlado y `parent_example_id` (§105); pares contrafactuales y
contexto adversario para robustez (§§56–57).

Fuentes: data-training §§8, 35, 46–47, 56–60, 104–105, 129–132.

## Verification gate

- Requiere PASS de `T-synth-factory`; el gold Wikidata se verifica
  por query al grafo, no por voto teacher (test: gold alterado a
  mano → el checker lo rechaza, §104).
- Tests: hard negatives del mismo taxón (falla si propone
  "database/volcano" a color favorito, §46); distribución K del mix
  de prueba dentro del mix orientativo; packs con ≥3 Q por state y
  mismo split para todo el grupo; robustez: inserción de 2 k tokens
  irrelevantes no mueve top-1 más allá del umbral (§57).
- El gate escribe `artifacts/gates/T-prog-gold/gate.json` con
  `pass: true`, 100 k Wikidata + conteos de miner/K/packs; sin ese
  artifact no arranca `#T-mix-1m`.

## Done when

- 100 k Wikidata deterministas + miner + sampler K + packer
  funcionando con Layers B (programmatic) y D (adversarial/OOD, §61).
- Quality score por example (human 1.0, program 1.0, 2-teacher 0.8,
  1-teacher 0.55, unverified → reject, §67) y difficulty
  (easy 25 / medium 50 / hard 25, §68) implementados en el sampler.

## Resolution

Pilot completo y gate en verde (`artifacts/gates/T-prog-gold/gate.json`,
`pass: true`): **100 004 preguntas** deterministas construidas desde Wikidata
sobre 5 taxones (city 2 789, film 1 125, book 1 835, mountain 2 627,
human 653), 22 568 packs multi-Q (3–8 preguntas por state, ningún padre
partido entre splits) y 13 533 pares contrafactuales.

<details><summary>Qué verifica el gate — 15 checks, todos en verde</summary>

- **Gold verificable**: cada respuesta se deriva del grafo y se vuelve a
  comprobar contra él (`every_gold_verifies`); una respuesta manipulada o un
  gold manipulado se rechazan (`tampered_answer_rejected`,
  `tampered_gold_rejected`).
- **Hard negatives del mismo taxón** (`distractors_same_taxon`): un
  distractor de otra familia semántica no cuenta como negativo.
- **Distribuciones dentro de guía**: mezcla de K (desviación ≤ 0,02),
  dificultad easy 0,25 / medium 0,50 / hard 0,25 y share de transformaciones
  ≤ 0,35.
- **Calidad y limpieza**: quality score medio 1,0 (suelo 0,55, 0 rechazos por
  validador), dedup al 0,9 con 9,87 % rechazado, firewall de benchmarks
  limpio y un solo split por grupo.
- **Robustez**: insertar 2 k tokens irrelevantes no mueve el top-1 más allá
  del 5 %.

Coste: 48,9 s de build en CPU. El artefacto desbloquea `#T-mix-1m`.
</details>
