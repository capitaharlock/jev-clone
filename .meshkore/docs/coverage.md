---
title: Coverage matrix
updated: 2026-09-21
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
| — | decision-model | T-prefetch | Prefetch único + conversión temprana de todas las fuentes |
| — | decision-model | T-train-base | Baseline TF-IDF+LR sobre sets convertidos, v2 absorbe Qwen-aug |

Gate command: `python training/python/tools/gate.py --task <task-id>`. It must
write `artifacts/gates/<task-id>/gate.json` with `pass: true`, commit, upstream
gate hash, config/data/model hashes, environment and raw-result references.
The next task must refuse to start if that artifact is missing, red, stale or
belongs to another lineage. A failed scientific target records NO-GO and takes
the documented fallback; failing tests or integrity checks never get bypassed.

`train-scaleout` is the only optional branch (backlog initiative, 3 tasks:
`T-scaleout-gate` → `T-dist-train` → `T-scaleout-data`). It unlocks after
`T-curriculum` (done) via its own gate, and no required task depends on it;
V1 therefore needs one terminal and one computer only.

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
| §§123–125 / single-machine | Un equipo por run; cola opcional con 3 workers; DDP solo CUDA homogéneo si gana medido | T-curriculum, #train-scaleout: T-scaleout-gate, T-dist-train, T-scaleout-data (backlog, opcional) |
| §§48–49 Gold | Human gold set, acuerdo inter-anotador | T-gold |
| §§50–53 Curriculum | Stages, sampling, cardinalidad, multi-Q | T-curriculum |
| §54–55 Scratch | Contrastivo extra, from-scratch | defer: research (solo con gap) |
| §§56–62 Hardware/latencia | M4/M5/RTX, breakdown, p50/p95/p99 | T-recon, T-local-infer, T-state-cache |
| §§63–65 Métricas | Calidad + métricas propias + stress | T-calib |
| §66 Baselines | Baselines 0–7 obligatorias | T-recon |
| §§68–70 Repo/tests/repro | Layout, testing, reproducibilidad | T-rust-skel |
| §§85, 107–111 Long/mutable state | Buckets 512/2K/4K/8K, position stress, invalidación | T-shared-state, T-state-cache |
| §§99 Shortcut learning | Label/source/position/template leakage | T-firewall, T-option-mixer |
| §§125–127 Scheduler/ablations | Cola local, successive halving, ablations | T-curriculum, T-dist-train (#train-scaleout) |
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
| Tres dispositivos | #train-scaleout: T-scaleout-gate (activación/inventario x3), T-dist-train (cola x3, paraleliza runs), T-scaleout-data (volumen); no DDP heterogéneo ni 3× sin benchmark |
| Data-training §§19–31, 120–126 | Fuentes masivas P0+P1, HuffPost/MASSIVE completos, cierre LogiQA/ReClor | #data-training: T-bigsrc, T-massive-huff |
| Data-training §§32–47, 127–133 | Synthetic factory council, Wikidata programático, hard negatives, K, multi-Q | #data-training: T-synth-factory, T-prog-gold |
| Data-training §§48–53, 61–68, 86–87, 95–97, 111–112, 135–137 | Mixes 1M→5M, guardrails, losses, distillation soft, scaling | #data-training: T-mix-1m (corpus y trainer comparten UNA receta, `data.mix.CLEAN_1M_TARGET/_SEED/_CAP_MARGIN`; `--max-samples` > mezcla realizada = `MixShortfallError`, repetir exige `--allow-repeat` y publica `epochs_over_corpus`), T-mix-5m (**NO-GO §§89/136**: 250 k→1 M no mejora — unseen-label cae −0.167/−0.190 y 9/9 cortes de razonamiento quedan dentro de su azar por CI95; `artifacts/gates/T-mix-5m/gate.json` + `SCALING_NOGO.md`; la regla "ningún GO con los evals en el azar" vive en `tools/mix_5m/nogo.py`) |
| Data-training §§54–55, 76–83, 113–115, 138–140 | OOD, calibration split, Jevals clean room, 4 reportes | #data-training: T-data-eval |
| Data-training §§21–22, 34, 50–51, 87–89 | v2/v3 10M/20M, FLAN amplio, Dolma (sólo si la curva abre) | #data-training: T-mix-10m (backlog) |
| Multi-región / routing global | defer: post-comercial |
| Monitor 24/7 terminal + dashboard (totalizadores, state.json) | #data-training: T-train-monitor (`tools/training_monitor.py`, `artifacts/runs/training-monitor/state.json`; v2: latido `--tick`, train en fondo con venv) |
| Teacher barato + datasets intencionales de decisión (email-triage v1, juez/labeler) | #data-training: T-teacher-intent `done` (`data/intent/email_triage.py` 5k filas, `data/teacher_client.py` con caché; piloto 500 acuerdo 0,92, coste 0,00 €; train 20260920T223639Z acc=1,0; 14 casos forward-testing) |
| gRPC / MessagePack / unix sockets | defer: solo si JSON limita |

## Auditoría 2026-09-21 — plan de corrección

Fuente: `.meshkore/docs/audit-2026-09-21.md`. Cada hallazgo mapea a una task.
Ningún hallazgo queda sin dueño; los tres `#I` nuevos son
`#decision-rebuild`, `#honest-eval` y `#oss-release` (este último en `backlog`,
porque publicar antes de tener modelo sería publicar un baseline léxico).

| Hallazgo | Severidad | Coverage |
|---|---|---|
| A · El modelo en entrenamiento no es el del plan (sin torch, sin Candle, proxies aleatorios) | BLOQUEANTE | #decision-rebuild: T-torch-stack `done` (torch 2.14 + transformers 5.17 en `.venv-train` con `requirements-train.txt` hash-pinned; ettin-68m y modernbert-base descargados y verificados por sha256 en `artifacts/weights/`; `model/encoder.py::encode_state` hace forward real MPS+CPU, gate `artifacts/gates/T-torch-stack/gate.json`; los proxies aleatorios del bake-off quedan `not_comparable`), T-bakeoff-real |
| B · El trainer descarta preguntas y opciones (`X = state`, `y = answer`, label space fijo) | BLOQUEANTE | #decision-rebuild: T-pointer-head `done` (`model/decision_head.py`: encode-once `H_s`, 2 capas de cross-attention, pointer scorer sobre el embedding del TEXTO de cada opción, `unknown` como logit aprendido dentro del softmax; sin `clf.classes_` ni matriz `num_labels×d`; K variable sin padding; gate `artifacts/gates/T-pointer-head/gate.json` pass=true, 2 699 778 params, KL de permutación 1.0e-07, p50 37.573 ms/fila con K=4), T-optset-sampler, T-train-real `done` (`training/python/train_decision.py`: un solo modelo sobre banking77+massive+huffpost+boolq con cross-entropy listwise sobre los `K + 1` logits de la fila, checkpoint safetensors + tokenizer.json + `model_version`, `metrics.jsonl` por step, job del daemon `train-decision`; gate `artifacts/gates/T-train-real/gate.json` **pass=false / NO-GO** en el primer stage: unseen 0.058 vs azar 0.165 — 4 de 5 checks verdes, no se escala hasta que la curva bata el azar) |
| C · acc=1,0000 es contaminación train/test (190 esqueletos, split `i % 10`) | BLOQUEANTE | #decision-rebuild: T-halt-contam · #honest-eval: T-split-domain `done` (`eval/splits.py`: agrupa por esqueleto normalizado + dominio + idioma y parte POR GRUPO, splits sellados en `artifacts/splits/<name>/{train,test}.ids` con seed y sha256 por fichero; auditoría retroactiva dataset a dataset en `artifacts/gates/T-split-domain/audit.json`; el escáner AST de `i % 10` da 0 splits por índice en el repo; contraste sobre el corpus en cuarentena: acc 1,0000 → **0,3503** (logloss 0,0019 → 1,6820) con las mismas filas y el mismo modelo; los números históricos que se apoyaban en `i % 10` quedan marcados inválidos (2026-09-21) en 93 artefactos y en el dashboard) |
| D · Q-W-E-N aporta casi cero originalidad (5 plantillas en 11 h, un solo dominio) | ALTO | #data-training: T-gen-schemas, T-gen-loop `done` (`tools/gen_schemas/loop.py` + `tools/data_gen_loop.py --loop`: loop continuo sobre `gen-schemas/v1` con gate POR LOTE —novelty cruzada contra el corpus existente, balance dominio/idioma, cobertura de K, hard negatives, tasa `unknown`— y lotes rechazados a `rejected/`, nunca a `corpus/`; criterio de parada fijado antes de medir: diversidad marginal < 250 esqueletos nuevos/hora de reloj sobre 4 rondas, o 3 lotes consecutivos rechazados; `run.json` + índice append-only de textos normalizados lo hacen reanudable sin regenerar; gate `artifacts/gates/T-gen-loop/gate.json`) |
| E · Mezcla dominada por ruido (civil-comments 74,6 %) | ALTO | #data-training: T-corpus-rebalance |
| F · La evaluación no mide la propiedad del producto (sin unseen, kappa 0 en verde) | ALTO | #honest-eval: T-unseen-labels, T-release-gate (`.meshkore/docs/release-criteria.md` v1 fechado 2026-09-21 y escrito ANTES de medir: 12 umbrales, cada uno con su línea de por qué ese número; `eval/release_gate.py` los lee del markdown —ninguna constante de calidad en el código— y publica `artifacts/gates/T-release-gate/gate.json` con `criteria_sha` y veredicto **NO-GO** (8/12 fallan sobre los números reales de #T-unseen-labels; kappa contra el profesor no lo mide ningún artefacto y por eso cuenta como fallado, no omitido). `eval/gate_rules.py` aplica C1/C2/C3 a todo `artifacts/gates/**` como ERROR duro: 14 de 31 gates históricos quedan **marcados** con `coherence-invalid.json` —no borrados—, `T-release/release.json` entre ellos por publicar `cohen_kappa 0.0` junto a un `cold_verdict: GO`. El criterio es una PROPUESTA: `signed_by: pending-operator`, y el gate emite veredicto igual, marcado `criteria_signed: false`) |
| G · LogiQA y ReClor se entrenan en vez de estar en firewall | MEDIO | #decision-rebuild: T-halt-contam |
| H · El repositorio no es publicable (9 145 ficheros de `target/`, sin README ni inferencia local) | APARCADO | #oss-release (backlog): T-repo-clean, T-readme-card, T-candle-infer |

Se conserva sin tocar lo que la auditoría dio por bueno: `data/schema.py`,
los `convert_*.py`, `data/firewall.py` + `data/leakage.py`,
`crates/jev-runtime` y `start-all.sh` (que deja de arrancar el generador
viejo).

### Orden de ejecución del plan de corrección

| Orden | Initiative | Task | Desbloquea |
|---:|---|---|---|
| 01 | decision-rebuild | T-halt-contam | Para la contaminación; sin esto nada se mide bien |
| 01' | decision-rebuild | T-torch-stack | Paralelo: sin torch no hay camino al head |
| 02 | decision-rebuild | T-pointer-head | **El núcleo**: la opción se puntúa por su texto |
| 02' | decision-rebuild | T-optset-sampler | Paralelo: K dinámica, hard negatives, `unknown` |
| 03 | honest-eval | T-split-domain | Split por esqueleto/dominio, splits sellados |
| 04 | decision-rebuild | T-train-real | Entrenamiento listwise, un checkpoint safetensors |
| 05 | honest-eval | T-unseen-labels | Métrica primaria: accuracy + ECE en etiquetas no vistas |
| 06 | decision-rebuild | T-bakeoff-real | Top-2 de backbone con pesos reales |
| 07 | honest-eval | T-release-gate | Criterio de release escrito antes de medir |
| 08 | data-training | T-gen-schemas | Generador de esquemas de decisión con presupuesto |
| 08' | data-training | T-gen-loop | Loop continuo autolimitado por el gate de diversidad |
| 09 | data-training | T-corpus-rebalance | Guardrails de mezcla como error duro |
| 10 | oss-release | T-repo-clean | Repo clonable, historial sin blobs |
| 11 | oss-release | T-candle-infer | Candle Metal/CUDA con paridad contra Python |
| 12 | oss-release | T-readme-card | Quickstart de 3 comandos + model card real |

## Hallazgo I (2026-09-22) — la generalización cae al escalar

| Hallazgo | Severidad | Lo entrega |
|---|---|---|
| I · La accuracy sobre etiquetas no vistas **decrece** al añadir datos (0,239 → 0,050 de 250 k a 1 M) y queda por debajo del azar (0,165), mientras la accuracy dentro del corpus se mantiene alta (dbpedia14 0,98). `#T-release-gate` da NO-GO con 10/12 criterios fallados. Más datos no lo arreglan: `#T-mix-5m` ya registró el NO-GO al escalado | CRÍTICO | #generalization-fix: T-antiscale-diag (ablación de los 5 ejes), T-unfreeze-backbone (frozen vs last-n vs full sobre el mismo 1 M), T-gen-objective (dropout de etiquetas, episódico, contrastivo, penalización de prior), T-labelspace-div (diversidad de espacios de etiquetas: medirla y generarla) |

`#oss-release` queda **aparcada en `backlog`** (decisión del operador,
2026-09-22): fuera de la ecuación hasta que haya GO. Publicar hoy sería
publicar un modelo por debajo del azar en la propiedad que lo define.

### Orden actualizado

| Orden | Initiative | Task | Desbloquea |
|---:|---|---|---|
| 13 | generalization-fix | T-antiscale-diag | Nombra la causa de la caída con un número |
| 14 | generalization-fix | T-unfreeze-backbone | Capacidad entrenable donde está el lenguaje |
| 14' | generalization-fix | T-gen-objective | Incentivo a comparar en vez de recordar |
| 14'' | generalization-fix | T-labelspace-div | La mitad de datos: inventario de espacios de etiquetas y, si faltan, generarlos |
| 15 | honest-eval | T-data-eval | OOD, calibración y los 4 reportes (repointed desde #data-training) |
| 16 | honest-eval | T-release-gate | Re-evaluación sobre el checkpoint arreglado + firma del operador |
| — | oss-release | T-repo-clean → T-candle-infer → T-readme-card | Backlog: fuera de la secuencia hasta un GO |
