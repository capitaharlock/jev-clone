---
title: Coverage matrix
updated: 2026-09-20
owner: architect-master
---

# Coverage matrix — `jev-clone`

Cada requisito del plan maestro (180 §) y del stack (41 §) mapea a una
task o a un `defer`. Consolidación deliberada: las 30 iniciativas I0–I29
del plan se comprimen en 5 work-streams. Las fuentes, licencias y
transformaciones están normalizadas en `source-register.md`.

## Execution order and progressive gates

La V1 tiene cinco iniciativas visibles que forman **un único paquete de
ejecución**. Para una corrida integral, **Run All debe seleccionar las cinco**:
`local-runtime`, `data-foundation`, `eval-trust`, `decision-model` y
`cloud-release`. No se debe seleccionar sólo cloud o datos, porque las
dependencias científicas precedentes estarían honestamente fuera de alcance.
Sólo `T-rust-skel` no tiene predecesor. Cada task siguiente requiere el PASS
anterior y conserva su linaje de artefactos.

| Order | Initiative | Task | Gate purpose |
|---:|---|---|---|
| 01 | local-runtime | T-rust-skel | Workspace, CI, fixtures y gate runner |
| 02 | data-foundation | T-data-schema | Schema, registry y license fence |
| 03 | eval-trust | T-firewall | Firewall de contaminación y stress suite |
| 04 | data-foundation | T-data-p0 | Adapters P0 y loader reproducible |
| 05 | eval-trust | T-recon | Baselines locales y harness de latencia |
| 06 | decision-model | T-bakeoff | Selección top-2 de backbone |
| 07 | decision-model | T-shared-state | Arquitectura encode-once y `unknown` |
| 08 | decision-model | T-option-mixer | Invariancia al orden de opciones |
| 09 | data-foundation | T-hardneg | Hard negatives y OOD/`unknown` |
| 10 | decision-model | T-curriculum | Entreno reproducible y ablations |
| 11 | decision-model | T-distillation | Teacher mix y prueba A/B de distillation |
| 12 | data-foundation | T-gold | Gold ES+EN y split sellado |
| 13 | eval-trust | T-calib | Calibración, abstención y GO/NO-GO |
| 14 | local-runtime | T-local-infer | Paridad PyTorch↔Candle local |
| 15 | local-runtime | T-state-cache | Cache, batching y benchmark warm |
| 16 | local-runtime | T-quant-onnx | Export/quantización sólo si gana medida |
| 17 | cloud-release | T-cloud-api | API y Docker reproducible local |
| 18 | cloud-release | T-release | Bundle verificable de research release |

Gate command: `python training/python/tools/gate.py --task <task-id>`. It must
write `artifacts/gates/<task-id>/gate.json` with `pass: true`, commit, upstream
gate hash, config/data/model hashes, environment and raw-result references.
The next task must refuse to start if that artifact is missing, red, stale or
belongs to another lineage. A failed scientific target records NO-GO and takes
the documented fallback; failing tests or integrity checks never get bypassed.

`T-dist-train` is the only optional branch. It unlocks after `T-v1-train`,
remains backlog, and no required task depends on it; V1 therefore needs one
terminal and one computer only.

## Sections (plan maestro)

| Source | Requirement | Coverage |
|---|---|---|
| §1 Producto | Motor state→distribuciones, sin generación | T-shared-state, T-release |
| §2 Targets | V0/V1/V2 (smoke→útil→interesante) | T-bakeoff, T-calib |
| §4 Jev-likes | Laya, Verdict, kev, jevlike, open-jev, NanoJev | T-recon |
| §5 Backbones | Carrera Ettin/ModernBERT/NeoBERT/LFM2.5; mmBERT solo si aporta señal | T-bakeoff, source-register |
| §6 Exp cero | Selección de backbone con mismo budget | T-bakeoff |
| §§7–12 Arquitectura | State encoder, Q/O repr, cross-attn, pointer head | T-shared-state |
| §13 Opciones | Interacción listwise (DeepSets/Set Transformer) | T-option-mixer |
| §14 Orden | Invariancia al orden | T-option-mixer |
| §15 Tipos | V1: choice (+boolean binario); score/extract/multiselect → V2 | T-data-schema, T-shared-state, T-curriculum, T-cloud-api |
| §§16–17 Abstención | unknown explícito, incertidumbre 2 niveles | T-shared-state, T-hardneg, T-calib |
| §18 Cascada | Fallback a LLM en casos difíciles | defer: V2, tras OOD sólido |
| §19 Early exit | Salida temprana por estabilidad | defer: V2 (I20 del plan) |
| §§20–22 Fast paths | Bi-encoder, late interaction, branch attention | defer: V2, si el POC lo pide |
| §23 Losses | CE+Brier+KL+permutación+ordinal | T-curriculum |
| §24 RLCD | Supervised primero; RL solo con necesidad real | T-curriculum |
| §§25–26 Calibración | Temp scaling, conformal/selective risk | T-calib |
| §27 Schema | Schema universal de datos | T-data-schema |
| §28 Governance | Licencias, hashes, leakage | T-data-schema, T-firewall |
| §§29–41 Datasets P0+P1 | HuffPost→HelpSteer2, MASSIVE, CLINC/OOS, LogiQA, ReClor; ANLI restringido | T-data-p0, T-distillation, T-gold, source-register |
| §43 Dataset matrix | Fuente, licencia, transformación y train/eval-only | T-data-schema, source-register |
| §42 Firewall | Benchmarks: no entrenar | T-firewall |
| §§44–47 Sintético | Hard negatives, teachers, presupuestos €; Qwen local primero | T-hardneg, T-distillation |
| §§123–125 / single-machine | Un equipo por run; cola opcional con 3 workers; DDP solo CUDA homogéneo si gana medido | T-curriculum, T-dist-train (backlog, opcional) |
| §§48–49 Gold | Human gold set, acuerdo inter-anotador | T-gold |
| §§50–53 Curriculum | Stages, sampling, cardinalidad, multi-Q | T-curriculum |
| §54–55 Scratch | Contrastivo extra, from-scratch | defer: research (solo con gap) |
| §§56–62 Hardware/latencia | M4/M5/RTX, breakdown, p50/p95/p99 | T-recon, T-local-infer, T-state-cache |
| §§63–65 Métricas | Calidad + métricas propias + stress | T-calib |
| §66 Baselines | Baselines 0–7 obligatorias | T-recon |
| §§68–70 Repo/tests/repro | Layout, testing, reproducibilidad | T-rust-skel |
| §§85, 107–111 Long/mutable state | Buckets 512/2K/4K/8K, position stress, invalidación | T-shared-state, T-state-cache |
| §§99 Shortcut learning | Label/source/position/template leakage | T-firewall, T-option-mixer |
| §§125–127 Scheduler/ablations | Cola local, successive halving, ablations | T-curriculum, T-dist-train |
| §§128–134 GO/M0–M5 | Criterios y milestones 0–5 | T-calib (GO), T-release (M5) |
| §135 No-hacer | Lista de prohibiciones del primer mes | T-calib |
| §§136–147 Riesgos | 11 riesgos + register | T-calib |
| §170 Release criteria | Qué exige llamarse "calibrated" | T-calib, T-release |
| §171 Naming | `jev-clone` solo interno hasta nombre comercial | context, T-release |
| §179 Sources | Registro canónico y revisiones/hash por artifact | source-register, T-data-schema |

## Rules (stack)

| # | Rule | Coverage |
|---|---|---|
| 1 | Rust owns product; Python solo research | T-rust-skel, T-local-infer |
| 2 | SafeTensors como contrato | T-rust-skel, T-local-infer |
| 3 | Candle primero; Burn solo fallback | T-rust-skel, T-local-infer |
| 4 | `encode_state` separado de `decide` | T-shared-state, T-state-cache |
| 5 | Misma semántica en todos los backends | T-local-infer |
| 6 | ONNX/TensorRT/CoreML solo tras profiling | T-quant-onnx |
| 7 | Sin kernels custom sin perfil | T-quant-onnx |
| 8 | Paridad PyTorch↔Candle bloquea release | T-rust-skel, T-local-infer |
| 9 | Contenedor inferencia sin Python | T-cloud-api |
| 10 | Cloud portable (adapter fino, no lock-in) | T-cloud-api |
| 11 | No loguear contenido de states | T-cloud-api, T-release |

## Explicit deliverables

| Deliverable | Coverage |
|---|---|
| Bake-off con Pareto publicado | T-bakeoff |
| Shared-state V1 + benchmark §177 | T-shared-state, T-state-cache |
| Calibration report por release | T-calib |
| Research release + model card | T-release |
| API pública documentada | T-cloud-api |
| MoE / from-scratch tiny / decoder→encoder | defer: research P3 |
| Compresión latente de state | defer: V2, tras V1 sólido |
| Long-state | T-shared-state mide 512/2K/4K/8K; compresión solo si el perfil lo exige |
| Tres dispositivos | T-dist-train paraleliza runs; no promete DDP heterogéneo ni 3× sin benchmark |
| Multi-región / routing global | defer: post-comercial |
| gRPC / MessagePack / unix sockets | defer: solo si JSON limita |
