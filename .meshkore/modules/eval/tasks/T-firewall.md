---
status: done
id: T-firewall
title: Eval firewall and stress suite
status: done
priority: high
owner: unassigned
category: eval
initiative: eval-trust
depends_on:
  - T-data-schema
created: 2026-09-19
updated: 2026-09-19
completed_at: 2026-09-19T22:44:27.011Z
resolved_by: A004
resolved_by_conv: general-09192230
commit_shas: ['099ba929306abaf782fc4740e50742e13d21fe26']
---
# Eval firewall and stress suite

Registry de benchmarks versionados e inmutables (MMLU-Pro, GPQA,
ARC, MuSR, RewardBench…) con scanner de contaminación en CI: entrenar
con ellos está prohibido. Stress suite propia: label rename, opaque-ID
trap, shuffle, hard siblings, state irrelevante, evidencia al final,
contradicciones, mismatch multilingüe, typos. Criterios GO/NO-GO del
§128 y release criteria del §170.

CLINC150/OOS queda como holdout de transferencia semántica y OOD, no como
train, hasta resolver su licencia. El scanner cubre hashes exactos y similitud
contra prompts/outputs de teachers para detectar paráfrasis contaminadas.

Fuente: plan §§42, 64–65, 120, 128, 147, 167–170.

## Verification gate

- Requiere PASS de `T-data-schema`.
- Fixtures canarios de exact match, texto normalizado y paráfrasis aproximada
  deben ser detectados; fixtures limpios no pueden generar falsos positivos por
  encima del umbral documentado.
- Cada benchmark queda fijado por revisión/hash y un test prueba que ni sus
  inputs ni outputs pueden entrar en train, teachers o hard negatives.
- La stress suite tiene semillas y umbrales versionados; el gate conserva raw
  results y PASS/NO-GO en `artifacts/gates/T-firewall/gate.json`.

## Done when

- Contamination scanner bloquea datasets que tocan benchmarks.
- Stress suite corre por release con umbrales GO/NO-GO.
- Ningún modelo se llama "calibrated" sin métricas held-out + OOD.
- Un dataset completo no visto demuestra semantic label transfer o dispara
  NO-GO para la hipótesis de decision foundation model.

## Resolution

Cadena revalidada y extendida: #T-data-schema re-PASS bajo el runner nuevo y #T-firewall implementado con gate PASS en el mismo linaje (`f0d7e0e`). Commit `099ba92` en main.
<details><summary>data/firewall.py + fixtures + tests — qué incluye</summary>

- Registry de 8 benchmarks (MMLU-Pro, GPQA, ARC, MuSR, RewardBench2, SimpleQA, OpenBookQA + holdout CLINC150-OOS) fijados por revisión inmutable y hash del corpus canario; barrera que bloquea train/teachers/hard-negatives.
- Scanner sobre `LeakageDetector` (exacto + normalizado + paráfrasis): 16/16 canarios detectados, 0 falsos positivos en 8 limpios.
- Stress suite versionada con 9 perturbaciones y semilla fija; veredicto GO (detección 1.0, FP 0.0; raw results en `artifacts/gates/T-firewall/`).
- Gate extendido con paso `python-firewall`; re-ejecución completa T-rust-skel → T-data-schema → T-firewall, todo PASS.
</details>

Siguiente en cola: #T-data-p0 (adapters P0, fija revisiones reales) — lo tomo si dices sigue.

— T-data-schema · revalidado con gate PASS bajo el runner acumulativo
— T-firewall · firewall de contaminación, stress suite y gate PASS (commit 099ba92)

**Commit** `099ba9293` · 10 files · 3.5M tokens

**Files changed (10):**
- `.meshkore/modules/data/tasks/T-data-schema.md`
- `.meshkore/modules/eval/tasks/T-firewall.md`
- `artifacts/fixtures/firewall/canary-exact.jsonl`
- `artifacts/fixtures/firewall/canary-normalized.jsonl`
- `artifacts/fixtures/firewall/canary-paraphrase.jsonl`
- `artifacts/fixtures/firewall/clean-general.jsonl`
- `artifacts/gates/T-rust-skel/gate.json`
- `data/firewall.py`
- `data/test_firewall.py`
- `training/python/tools/gate.py`
