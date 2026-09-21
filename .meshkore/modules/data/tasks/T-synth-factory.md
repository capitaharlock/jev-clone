---
id: T-synth-factory
title: Synthetic factory V1 (teacher council, piloto 50k)
status: done
priority: high
owner: unassigned
category: data
initiative: data-training
depends_on:
  - T-massive-huff
created: 2026-09-20
updated: 2026-09-20
---

# Synthetic factory V1 (teacher council, piloto 50k)

Construye `tools/data_factory/` (§69: source_streamer, task_proposer,
teacher_router, solver, adversary, judge, validator, deduper,
license_tracker, shard_writer, metrics) con el council Qwen↔DeepSeek:
proposer/solver independientes, roles invertidos al 50 % para evitar
teacher signature, tercer juez sólo en desacuerdo (§§37–41).
Fuente de states: FineWeb-Edu en streaming (§§32, 127); grounded >
ungrounded, Qwen nunca inventa el mundo desde cero (§36).
Output estricto JSON state/questions/candidates/gold, sin CoT (§43);
validator determinista (IDs únicos, gold miembro, 2 ≤ K ≤ 255, sin
leakage, límites de tokens, §§44, 103–104); dedupe exacto SHA-256 +
semántico, splits por grupo de parent (§§45, 105–106, 133).
PILOTO 50 k aceptados, no 5 M (§127); Qwen sólo donde aporta valor:
descripciones, paráfrasis, distractores, ambigüedad (§§101–102).

Fuentes: data-training §§32–45, 69–72, 101–106, 127, 133.

## Verification gate

- Requiere PASS de `T-massive-huff`; el factory corre contra endpoint
  local OpenAI-compatible/Ollama/MLX/vLLM configurable, nunca contra
  un UI no reproducible (§42).
- Tests: validator rechaza fixtures corruptos (gold fuera de
  candidatos, K=1, duplicados, leakage estado↔target); el acuerdo
  proposer/solver se mide y los desacuerdos van a adjudicación o
  reject (§37); dedupe elimina near-duplicates semánticos y el split
  por grupo pasa test de leakage (paráfrasis nunca en split distinto,
  §106); roles invertidos verificados por metadata de teacher.
- El gate escribe `artifacts/gates/T-synth-factory/gate.json` con
  `pass: true`, 50 k aceptados, tasa de reject y acuerdo inter-teacher;
  sin ese artifact no arranca `#T-prog-gold`.

## Done when

- `tools/data_factory/` corre end-to-end (stream → queue → teachers →
  validación → dedupe → shards) y produce 50 k aceptados.
- Cero CoT almacenado; toda variante registra `parent_example_id`.
- Tasa de acuerdo y de reject publicadas como baseline de calidad.
