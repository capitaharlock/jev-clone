---
id: T-prog-gold
title: Gold programático, hard negatives y packs multi-Q
status: blocked
priority: high
owner: unassigned
category: data
initiative: data-training
depends_on:
  - T-synth-factory
created: 2026-09-20
updated: 2026-09-20
failed_at: 2026-09-20T18:32:20.196Z
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

**Failed — exit 143.**

En marcha #T-prog-gold en #data-training: tests propios 14/14 en verde, pilot lanzado a 100k preguntas Wikidata (fetch city en curso). Al terminar verifica gate PASS y cierra con commit.

<details><summary>Estado — qué hay y qué falta</summary>

- Implementación ya existente: `tools/prog_gold/` (fetch, graph, miner, sampler_k, packer, transforms, pipeline) + `data/test_prog_gold.py` 14 tests OK vía unittest.
- Sin caché Wikidata (`artifacts/data-raw/wikidata/` vacía, store 0 entidades): el pilot hace fetch SPARQL + labels, construye records, escribe shards en `artifacts/prog_gold/v1/` y gate en `artifacts/gates/T-prog-gold/gate.json`.
- Al completarse: comprobar `pass:true`, preguntas ≥100k, histogramas K/dificultad, y cerrar task + commit en el linaje actual.
</details>Pilot #T-prog-gold en #data-training sigue corriendo (fetch Wikidata en curso, sin errores). Aviso al terminar con el gate y el cierre.

<details><summary>Estado — sin cambios que reportar</summary>

- Sin salida nueva del pilot; sigue en fase de fetch.
- Al completarse: verificar `artifacts/gates/T-prog-gold/gate.json` (`pass:true`, ≥100k preguntas), cerrar task y commit.
</details>⟦anchor⟧ {"i":"data-training","t":"T-prog-gold"}

Pilot #T-prog-gold en #data-training sigue en fetch Wikidata (caché aún vacío, primera query en curso). Sin errores; aviso al terminar con gate y cierre.

<details><summary>Detalle — progreso del fetch</summary>

- `artifacts/data-raw/wikidata/` creada pero aún sin payloads: el pilot sigue en el primer taxón (`city`).
- Wikidata SPARQL + labels suele tardar varios minutos por taxón con reintentos; el build de 100k y el gate vienen después en el mismo proceso.
</details>

28M tokens
