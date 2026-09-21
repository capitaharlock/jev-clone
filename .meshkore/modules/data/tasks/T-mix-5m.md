---
id: T-mix-5m
title: Escala a 5M y distillation soft piloto
status: active
priority: high
owner: unassigned
category: data
initiative: data-training
depends_on:
  - T-mix-1m
created: 2026-09-20
updated: 2026-09-20
---

# Escala a 5M y distillation soft piloto

SÓLO si 1 M demuestra generalización (§§87, 136): escalar a 5 M
añadiendo diversidad (más familias, buckets K, idiomas, long context,
OOD, desacuerdos teacher — no duplicar, §87). En paralelo, piloto de
distillation sobre 100 k high-value (§137): soft targets por acuerdo
de N teachers o N stochastic samples — NUNCA un único número
verbalizado como verdad (§52: métodos A/B/C; loss CE(gold) +
λ·KL(teacher‖student)); stop cada 500 k (micro-checkpoint + eval
fija; sin ganancia → cambiar fuente/familia/dificultad, §73);
ablations real-only / +programmatic / +synthetic / full para saber
si el factory aporta (§112); curva 250 k→1 M→5 M→10 M con quality,
NLL, Brier, OOD y latencia (§111). Calibración final con labels
reales held-out, no copiando a Qwen (§53).

Fuentes: data-training §§52–53, 62, 73–75, 87, 111–112, 136–137.

## Verification gate

- Requiere PASS de `T-mix-1m` con ganancia demostrada; si 1 M no
  mejora, esta task registra NO-GO y no genera 5 M (§§89, 136).
- Tests: curva de scaling graficada con los mismos evals fijos;
  ablation muestra el delta de cada capa; soft targets verificados
  por construcción (acuerdo/samples, no verbalización única);
  active generation posterior: el student-in-the-loop propone
  confusiones X↔Y validadas por otro teacher (§§74–75).
- El gate escribe `artifacts/gates/T-mix-5m/gate.json` con `pass: true`,
  curva, deltas de ablation y veredicto de continuar a 10 M; sin ese
  artifact no arranca `#T-data-eval`.

## Done when

- Mix 5 M (si la curva lo justifica) + piloto soft 100 k evaluado.
- Curva 250 k→1 M→5 M y ablations publicadas; decisión 10 M tomada
  con datos, no con intuición.
