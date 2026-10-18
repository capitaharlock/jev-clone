---
title: Coverage matrix
updated: 2026-09-25
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
| 13 | generalization-fix | T-antiscale-diag | Nombra la causa de la caída con un número — hecho salvo el eje 3: eje dominante **4 · calibración vs ranking, 75,4 %** (`artifacts/gates/T-antiscale-diag/gate.json`) |
| 14 | generalization-fix | T-gen-objective | Incentivo a comparar en vez de recordar — **priorizada primera** por el eje dominante de `#T-antiscale-diag` |
| 14' | generalization-fix | T-unfreeze-backbone | Capacidad entrenable donde está el lenguaje — segunda; vuelve a primera si el eje 3 (cabeza ×2/×4, job `antiscale-wide`) aplana la pendiente |
| 14'' | generalization-fix | T-labelspace-div | La mitad de datos: inventario de espacios de etiquetas y, si faltan, generarlos |
| 14'''' | generalization-fix | T-lever-stack | Apilar las dos palancas que sí aplanan la pendiente (cabeza d512 + objetivo prior); `last-n` del backbone quedó NO-GO en `#T-unfreeze-backbone` |
| 14''' | train-scaleout | T-metal-throughput | Coste, no métrica: una llamada de cabeza por batch en vez de una por fila — ×2,7 en la config de entreno real y GPU Metal al 97 % a B=128 (`artifacts/gates/T-metal-throughput/gate.json`) |
| 15 | honest-eval | T-data-eval | OOD, calibración y los 4 reportes (repointed desde #data-training) |
| 16 | honest-eval | T-release-gate | Re-evaluación sobre el checkpoint arreglado + firma del operador |
| — | oss-release | T-repo-clean → T-candle-infer → T-readme-card | Backlog: fuera de la secuencia hasta un GO |

## Replanificación 2026-09-23 — dónde está la métrica y qué queda

Estado medido, no estimado. Mejor checkpoint del proyecto sobre el gate
independiente de `#T-unseen-labels` (n=5624, azar 0,166): **unseen 0,2879**,
seen 0,4864, umbral de release **0,50**. Sobre banking77 —el corte comparable
con el benchmark público Jevals— **0,237 contra 0,166 de azar**: el producto
apenas se despega del azar en la tarea que lo define.

Tres ejes de modelo medidos y agotados o casi: capacidad de cabeza (d512 aplana
−67 % de la caída), objetivo (prior-penalty, −54 %), backbone entrenable
(**NO-GO**). Ninguno la elimina. Queda vivo el eje de datos y falta toda
referencia externa. De ahí el reparto de abajo.

| Orden | Initiative | Task | Qué decide |
|---:|---|---|---|
| 17 | generalization-fix | T-lever-stack | Techo del eje modelo: ¿las dos palancas suman? (job `lever-stack-d512-prior`) |
| 18 | generalization-fix | T-labelspace-div | **El eje vivo**: curva de la mezcla de 20 000 mini-taxonomías contra la de 9 |
| 19 | teacher-distill | T-teacher-probe | La distancia real: el profesor puntuado sobre NUESTRO corte unseen, mismas filas. Cliente + probe hechos (29 tests); medición BLOQUEADA hasta saber el proveedor de la key (401 en los 6 endpoints públicos). Mientras tanto `fullspace.json`: **0,0123 a 77 vías en BANKING77, bajo el azar 0,0130**, contra el 0,924 publicado de Jev |
| 20 | teacher-distill | T-teacher-kappa | Cierra `teacher_cohen_kappa_min`, hoy fallado por evidencia ausente |
| 20' | generalization-fix | T-bigk-optsets | La cardinalidad del objetivo: `k_max=8` en todo el corpus y la ventaja sobre azar se disuelve en K=77 (`fullspace.json`) |
| 21 | generalization-fix | T-xlingual-holdout | `cross_lingual_holdout` falla por construcción en todos los runs: it-IT/pt-PT nunca se excluyeron |
| 22 | teacher-distill | T-teacher-labelspaces | Condicionada al veredicto de #18: taxonomías del profesor, no filas |
| 23 | honest-eval | T-data-eval → T-release-gate | Re-evaluación y firma, cuando haya algo que firmar |

## Replanificación 2026-09-24 — arranca la fase 2 del entreno

`eval.fullspace` midió el régimen que el producto promete y el único comparable
con una cifra publicada: BANKING77 con sus **77 etiquetas**. **0,0123 con azar
0,0130**. La fase 1 entrenó y midió con 3–8 opciones muestreadas y backbone
congelado —un régimen barato, elegido para levantar el sistema end-to-end— y en
él el modelo aprende una preferencia local, no un ranking del espacio. La
transición está documentada en `.meshkore/docs/fase-2-espacio-completo.md`.

Consecuencia de planificación: `#generalization-fix` cierra su alcance (fase 1)
y deja de ser la cabeza de la cadena; sus veredictos valen en su régimen y no
se heredan (regla R4). Nace `#full-space-training`, que cambia el objetivo de
entreno en vez de ajustar hiperparámetros, y `#honest-eval` recibe el ajuste
del approach de testing: cardinalidad completa como métrica primaria.

| Orden | Initiative | Task | Qué decide |
|---:|---|---|---|
| 24 | full-space-training | T-fullspace-objective | **La task central**: pérdida sobre el espacio entero (in-batch + log-Q + caché de claves de etiqueta). Brazo a 62 k antes de cualquier 1 M |
| 25 | full-space-training | T-bigk-optsets | Lado de datos del mismo cambio: el corpus entrega el espacio entero cuando existe, no 8 opciones (repointed desde #generalization-fix) |
| 26 | honest-eval | T-eval-cardinality | Ordena el testing: cardinalidad completa como métrica primaria, `beats_chance` obligatorio en `gate_rules`, un solo scoreboard, tres cortes contradictorios reconciliados |
| 27 | full-space-training | T-encoder-finetune | Reabre el NO-GO de `#T-unfreeze-backbone` bajo el objetivo nuevo; incluye el brazo `full` que nunca se corrió |
| 28 | full-space-training | T-jev-parity | La referencia externa que faltaba desde el día 1: mismas 3 080 filas, K=77, diferencias de protocolo declaradas. No gasta saldo |
| 29 | full-space-training | T-labelspace-factory | Espacios de etiquetas (no paráfrasis) con el Qwen local; `episodic-div` 20 000 mini-taxonomías ya en disco es el primer material |
| 30 | teacher-distill | T-teacher-auth | Endpoint identificado (`api.typesafe.ai`, Bearer); la key aportada da 401. Desbloquea #T-teacher-probe y #T-teacher-kappa |
| — | generalization-fix | T-labelspace-div, T-xlingual-holdout | Lo único que sigue vivo allí: la curva de diversidad en marcha y el holdout cross-lingual, que falla en cualquier objetivo |

## Reordenación 2026-09-24 — tras la revisión externa

Una auditoría externa independiente leyó plan, código y artefactos antes del
primer run de fase 2 (`.meshkore/docs/revision-externa-2026-09-24.md`, hallazgos
verificados contra el árbol). Cambia el **orden** y añade dos tasks: la vía
exacta del objetivo es más barata que la muestreada y va antes, y antes que
ambas se mide qué información recibe el modelo en las opciones — al profesor se
le dan definiciones y 24 ejemplos, a nosotros identificadores crudos.

| Orden | Initiative | Task | Qué decide |
|---:|---|---|---|
| 31 | honest-eval | T-eval-cardinality (ampliada) | Añade: `beats_chance` con la accuracy real (hoy usa el ranking forzado), corte de desarrollo separado del test con registro de consultas, frecuencia de predicciones y matriz de confusión |
| 32 | full-space-training | **T-option-text** (entregada 2026-09-24) | Tres brazos a K=77 sobre el checkpoint actual, sin entrenar: identificadores / descripciones / con ejemplos. Los tres se quedan en el azar — el denominador sigue siendo la primera hipótesis |
| 33 | full-space-training | T-bigk-optsets (ahora activa) | La **vía exacta**: `cross_entropy` ya normaliza sobre las columnas que recibe, así que el espacio enumerable completo no necesita fórmula nueva. Coste de la cabeza a K=8/41/77 publicado |
| 34 | full-space-training | **T-fullspace-objective** (entregada 2026-09-24) | La **vía muestreada**: pérdida especificada con ecuación y tests, distribución real de etiquetas por batch medida, coste a cientos y decisión escrita sobre la cabeza, y el brazo a 62 k con veredicto pre-registrado |
| 35 | full-space-training | T-encoder-finetune | Corregido: la acumulación de gradiente no conserva los negativos in-batch; memoria medida en la configuración real antes de prometer el run |
| 36 | oss-release | **T-serve-engine** (nueva) | Servidor sin `Engine`, cargador Rust que ignoraría el encoder afinado, y colisión de la clave de caché en `/v1/choice`. Bloquea servir el modelo de fase 2, no lo explica |

## Entregado 2026-09-24 — `#T-option-text` (etapa 2 de la fase 2)

Qué información recibe el modelo, puesta a precio a K=77. No entrena nada:
un forward por brazo sobre el checkpoint que ya existía.

| Entregable | Dónde |
|---|---|
| Tres brazos + un diagnóstico, mismas filas selladas, mismo checkpoint, K=77, con aciertos, IC 95 %, azar y abstención en **un solo** artefacto | `eval/optiontext.py`, `artifacts/gates/T-option-text/optiontext.json` |
| Presupuesto de contexto por brazo (R8): tokens pedidos/retenidos, ejemplos completos, posición de la consulta — medido con el tokenizador real, no estimado | `eval/optiontext.py: budget`, `option_budget` |
| Las 77 definiciones del profesor, byte a byte, con commit inmutable, sha256, licencia y su salvedad; el cargador rechaza un fichero que se haya editado en local | `data/taxonomies/banking77-jev/`, `data/taxonomy.py`, `data/test_taxonomy.py` |
| BM25 del profesor (palabras + pares adyacentes, ≤24, ≤4 por clase) sobre las filas de train **menos** el corte sellado | `eval/optiontext.py: BM25`, `support_pool` |
| Una sola consulta al corte reservado, registrada en el gate de la task **y** en el registro R7 del repo | `artifacts/gates/T-option-text/test-queries.json`, `artifacts/gates/T-eval-cardinality/test-queries.json` |

Resultado: los cuatro brazos se quedan dentro del intervalo del azar en el
corte de desarrollo (A 9/1 000 = 0,0090 [0,0047, 0,0170]; B 14/1 000 = 0,0140
[0,0084, 0,0234]; C 13/1 000 = 0,0130 [0,0076, 0,0221]; azar 0,012987) y la
ventaja de B **no reproduce** en el corte reservado (A 38/3 080 = 0,0123,
B 37/3 080 = 0,0120). El presupuesto añade lo que faltaba: con 24 ejemplos el
estado pide 487 tokens y retiene 256 — sólo el 49,6 % de las demostraciones
entra entera, y en el orden del profesor la consulta sobrevive en 1 fila de
1 000. La información de entrada queda **descartada** como causa;
`#T-fullspace-objective` conserva la primera prioridad.

## Entregado 2026-09-24 — `#T-eval-cardinality` (etapa 1 de la fase 2)

El protocolo de medición, que bloqueaba el resto de la fase. Nada de esto
entrena: reordena el testing.

| Entregable | Dónde |
|---|---|
| Regla **C4**: una `accuracy` sin `chance`, sin cardinalidad y sin IC 95 % al lado hace fallar el gate, verde o rojo; la cita ajena declarada se exime | `eval/gate_rules.py`, `eval/test_gate_rules.py` |
| `beats_chance` se decide con la accuracy real y `ranking_beats_chance` con el ranking forzado (abstención total → `beats_chance: false`) | `eval/fullspace.py`, `eval/test_fullspace.py` |
| El barrido de K se publica con conteos e IC y **sin narrativa de progresión**: K=8 y K=40 baten su azar, K=5, K=20 y K=77 no, y están desordenados | `eval/fullspace.py: sweep_reading` |
| Un comando, una tabla: primaria + diagnósticos + paridad + coste + frecuencia de predicción + confusión + muestra de errores | `eval/scoreboard.py`, `artifacts/gates/T-eval-cardinality/scoreboard.json` |
| Corte de desarrollo sellado (1 000 filas del split train) distinto del test reservado (3 080), sin solape, con registro de consultas que arranca con las 7 ya hechas | `eval/cuts.py`, `artifacts/splits/T-eval-cardinality-*`, `artifacts/gates/T-eval-cardinality/test-queries.json` |
| `eval.unseen` y el stage eval del trainer etiquetados `role: diagnostic` con su `n` y su K en el encabezado | `eval/unseen.py`, `training/python/train_decision.py` |
| Todo checkpoint nuevo publica `cardinality_regime` (R4) en `run.json` y en su manifest | `training/python/train_decision.py` |
| Qué corte contesta qué pregunta y cuál manda | [cortes-de-evaluacion.md](cortes-de-evaluacion.md) |

Primer resultado del scoreboard sobre `leverstack-d512-prior-ettin-68m-s20260922`:
en el corte de desarrollo, 9/1 000 = 0,0090 con IC [0,0047, 0,0170] y azar
0,0130 — el mismo cuadro que el corte reservado, indistinguible del azar. El
diagnóstico nuevo añade lo que ninguna accuracy decía: sólo 26 de las 77
etiquetas se predicen alguna vez y `lost_or_stolen_phone` se lleva el 33,9 % de
las filas (uniforme sería 1,3 %). Causa no medida: `#T-option-text` y
`#T-bigk-optsets`.

## Entregado 2026-09-24 — `#T-fullspace-objective` (etapa 4 de la fase 2)

Degradada por el NO-GO de `#T-bigk-optsets`: deja de ser la continuación de
la hipótesis de cardinalidad y pasa a cerrar la especificación y el coste
que `#T-encoder-finetune` necesita, más la vía de los espacios no
enumerables. Por eso los done-when de **especificación y medida** van
primero y el brazo de entreno último.

| Entregable | Dónde |
|---|---|
| La pérdida escrita con ecuación: modo exacto verificado en **valor y gradiente** contra el softmax completo, y modo muestreado con su estimador | `training/python/fullspace_loss.py`, `artifacts/gates/T-fullspace-objective/loss-spec.json` |
| Las propiedades del estimador por **enumeración exacta**, no Monte Carlo: `E[R̂]=R` bajo propuesta y bajo inclusión, y el sesgo de usar `m·Q` sobre un conjunto deduplicado, medido | `training/python/test_fullspace_loss.py` |
| La pérdida muestreada **subestima** (Jensen, `log` cóncava), no sobreestima — el signo se mide, y el veredicto declara que una `train_loss` más baja no es una mejora | `test_the_loss_estimator_underestimates` |
| Colisiones y falsos negativos entre espacios: filtro sobre el POOL antes de muestrear (quitar un duplicado ya extraído rompe el insesgamiento) y lo indetectable, declarado | `fullspace_loss.filter_pool` |
| Cuándo se omite `log Q` y por qué — tres motivos, ninguno más; y `prior_penalty` **no** es una corrección log-Q, medido | `fullspace_loss.OMISSIONS`, `test_prior_penalty_is_not_a_log_q_correction` |
| Distribución real del cargador: **100 % de los batches son de un solo dataset**, mediana de 5 etiquetas ofrecidas únicas. «Cientos por batch» retirado | `tools/fullspace_batches.py`, `artifacts/gates/T-fullspace-objective/batch-composition.json` |
| El contrafactual de batches mixtos (mediana 90,5 ofrecidas, 12 espacios) y la tasa de falso negativo por masa (97,7 % de filas con colisión de oro) que obliga al filtro | mismo artefacto |
| Coste a **cientos** con la cabeza actual y la ablación de scoring independiente, cada medida en su subproceso; K=8/41/77 reutilizados de `#T-bigk-optsets`, no repetidos | `tools/fullspace_cost.py`, `artifacts/gates/T-fullspace-objective/cost.json` |
| Decisión escrita sobre la cabeza: se conserva o se cambia, y si se cambia el brazo se etiqueta CAMBIO DE ARQUITECTURA | `artifacts/gates/T-fullspace-objective/head-decision.md` |
| El brazo: batches mixtos + negativos in-batch entre espacios con `log π` (inclusión, no `m·Q`), cabeza intacta, régimen declarado en `run.json` | `training/python/inbatch.py`, `training/python/train_decision.py --in-batch-negatives` |
| Veredicto **pre-registrado** antes del run, con la regla 6 que impide publicar como veredicto un brazo corrido sobre batches de un solo dataset | `artifacts/gates/T-fullspace-objective/verdict.md` |

## 2026-09-25 — Reorientación: el plan de recuperación sustituye la vía del denominador

Fuente: `.meshkore/docs/plan-recuperacion-2026-09-24.md` (reanálisis del operador,
con prueba de mecanismo reproducible en `.meshkore/docs/evidence/probe-mechanism-2026-09-24.json`).

**Hecho medido que cierra la vía anterior:** `#T-fullspace-objective` 0/1000 con
87,7 % de abstención y 9/1000 forzando elección; `#T-bigk-optsets` 9/1000 en brazo
y control; azar a K=77 = 12,99/1000. El IC95 % del brazo forzado contiene el azar.

**Hecho medido que corrige el diagnóstico:** las puntuaciones **siguen al texto**
de la opción (permutar mueve <5 × 10⁻⁷; intercambiar textos intercambia
puntuaciones). El fallo es la insensibilidad al cambio decisivo del estado y de la
pregunta, no la posición.

### Retirado del alcance activo

| Elemento | Estado | Por qué |
|---|---|---|
| `#full-space-training` | `superseded` → `initiatives/log/` | Su tesis (denominador) está falsificada por su propio gate; plan §§2, 9 |
| `#generalization-fix` | `superseded` → `initiatives/log/` | Las cuatro palancas de fase 1 están medidas y ninguna aplana la pendiente |
| `#T-labelspace-factory` | `cancelled` → `modules/data/log/` | 2.000 espacios con plantilla determinista = diversidad de nombres, no de razonamiento (§4.2) |
| `#T-labelspace-div` | `cancelled` → `modules/data/log/` | Su inventario ya dio el hallazgo (9 taxonomías = 83,1 %); la curva es más volumen de la misma plantilla |
| `#T-xlingual-holdout` | `cancelled` → `modules/data/log/` | ES/EN pasan a ser ejes de primera clase de la batería, no un parche del holdout |
| `#T-teacher-labelspaces` | `cancelled` → `modules/data/log/` | Misma apuesta de plantillas, con dependencia de una credencial que da 401 |
| `#T-data-eval` | `cancelled` → `modules/data/log/` | Escrita contra la mezcla de 5 M y el clean room de Jevals; se rehace en `#T-battery-calib` |
| `#T-encoder-finetune` | `cancelled` → `modules/model/log/` | Sus brazos exigían «el objetivo ganador» de fase 2, que cerró en NO-GO |
| `#teacher-distill` | `active` → `backlog` | §8: no bloquear la recuperación por la credencial de TypeSafe; Qwen local asume el papel de referencia |
| `#T-fullspace-objective` | `done` con NO-GO registrado | El gate existe y es concluyente; las dos correcciones de narrativa y de matemática se reasignan |

### Cobertura del plan de recuperación

| § del plan | Requisito | Task que lo entrega |
|---|---|---|
| §5, §7.3 | Scorer compartido `z_i = f(estado, pregunta, opción_i)`, CE por pregunta | `#T-ce-scorer` |
| §5 | Contrato de contexto comparativo (puntuar sin ver las demás pierde «mejor») | `#T-episode-contract` + `#T-ce-scorer` |
| §5 | Punto de partida NLI multilingüe; ModernBERT-zeroshot con la advertencia BANKING77 | `#T-ce-scorer`, `#T-preflight-refs` |
| §9 | Retirar la afirmación de insesgadez Horvitz–Thompson de `fullspace_loss.py` | `#T-ce-scorer` |
| §6 | Episodio completo con evidencia, familia, idioma y grupo de variantes | `#T-episode-contract` |
| §6 | Cinco familias, atributos en el estado, gold por regla en lo numérico | `#T-episode-gen` |
| §6 | Contrafactuales: hecho / pregunta / descripción cambian el gold; paráfrasis no | `#T-counterfactuals` |
| §6 | Verificador separado, muestra humana estratificada, embudo medido | `#T-episode-verify` |
| §6 | Splits por familia/entidad/espacio/grupo + corte privado | `#T-episode-splits` |
| §7.1 | 400 dev + 600 sellados, cinco familias, ES/EN, K=2/3/8 | `#T-battery-dev`, `#T-battery-sealed` |
| §7.1 | Diagnóstico separado a K=20/77 | `#T-battery-sealed` |
| §7.2 | Referencias antes de entrenar: Qwen local, NLI sin ajustar, checkpoint actual | `#T-preflight-refs` |
| §7.3 | Mecánica validada por sobreajuste de 32–64 casos (>95 %) | `#T-ce-mechanics` |
| §7.4 | 5.000 → 20.000 decisiones verificadas, encoder ajustable, evaluación intermedia | `#T-ce-finetune` |
| §7.5 | Segunda semilla, apertura única del test sellado, ≥70 % macro con IC inferior ≥70 % | `#T-ce-confirm`, `#T-release-gate` |
| §7 métricas | Suite mínima; ranking y abstención por separado; azar por K; n por cruce | `#T-battery-metrics` |
| §7 métricas | Calibración en dev verificada en test, risk–coverage, ECE/NLL/Brier por familia | `#T-battery-calib` |
| §2, §8 | Corregir las lecturas de gate que sus propios números contradicen | `#T-battery-metrics` |
| §7.6 | 100 k dirigidos a errores, destilación a estado compartido, cuantización, paridad Rust | `#T-ce-distill`, `#T-ce-cascade` (backlog) |
| §8 | Jev como comparación externa cuando exista acceso válido, sin bloquear | `#teacher-distill` (backlog) |
| §10 | Motor en el servidor, encoder afinado, clave de caché V1 — sobre el checkpoint elegido | `#T-ce-cascade` (backlog) |

### Orden de ejecución del piloto

Tres iniciativas activas, cada una con una entrada **sin dependencias** para que
seleccionarlas por separado en Run All no deje una cola muerta; el DAG completo se
resuelve seleccionando las tres.

| Order | Initiative | Task | Gate purpose |
|---:|---|---|---|
| 01 | episodic-data | T-episode-contract | Contrato de episodio + validador (entrada) |
| 01 | honest-eval | T-battery-metrics | Suite de métricas + corrección de narrativas (entrada) |
| 01 | cross-encoder-pilot | T-ce-scorer | Scorer compartido y medición sin entrenar (entrada) |
| 02 | honest-eval | T-battery-dev | 400 casos de desarrollo |
| 03 | episodic-data | T-episode-gen | Episodios de las cinco familias |
| 04 | honest-eval | T-preflight-refs | Puerta: Qwen / NLI / control pointer |
| 05 | episodic-data | T-counterfactuals | Variantes que cambian el gold |
| 06 | episodic-data | T-episode-verify | Verificador separado y embudo |
| 07 | honest-eval | T-battery-sealed | Test sellado + diagnóstico K=20/77 |
| 08 | cross-encoder-pilot | T-ce-mechanics | Sobreajuste 32–64 (>95 %) |
| 09 | episodic-data | T-episode-splits | Splits sin fuga de variantes |
| 10 | cross-encoder-pilot | T-ce-finetune | 5.000 → 20.000 verificadas |
| 11 | honest-eval | T-battery-calib | Calibración y abstención aparte |
| 12 | cross-encoder-pilot | T-ce-confirm | Segunda semilla + apertura única |
| 13 | honest-eval | T-release-gate | Criterio v2 firmado y GO/NO-GO |

## Entregado 2026-09-26 — `#T-ce-scorer` (entrada del piloto, `#cross-encoder-pilot`)

| Requisito de la task | Dónde está |
|---|---|
| Scorer compartido `z_i = f(ESTADO, PREGUNTA, RESPUESTA_i)`, softmax sobre los K, una salida escalar y ninguna neurona por etiqueta (auditado, no afirmado) | `model/ce_scorer.py` (`CrossEncoderScorer`, `scalar_output_report`), `artifacts/gates/T-ce-scorer/contract.json` |
| Formato de hipótesis **fijado** y documentado, ES y EN | `model.ce_scorer.HYPOTHESIS` / `PREMISE`; viaja en los dos artefactos |
| Los dos checkpoints cablearon y puntúan **sin entrenar**: `minilmv2-l6-mnli-xnli` 17/28 (IC95 % 0,424–0,764) y `modernbert-zeroshot-v2` 18/28 (0,458–0,793), azar 0,333 | `eval/ce_nograd.py`, `artifacts/gates/T-ce-scorer/nograd.json` |
| Advertencia BANKING77 de ModernBERT-zeroshot en el registro de pesos, no sólo en prosa | `model.weights.SCORERS[...]["contaminated_benchmarks"]` |
| Contrato de contexto comparativo por familia, probado en las **dos** direcciones | `model.ce_scorer.COMPARATIVE_CONTEXT`, `check_comparative_context`, `model/test_ce_scorer.py` |
| Permutación (<1e-5 sobre probabilidades realineadas) e intercambio de textos conservando ids, sobre el juguete determinista **y** sobre los pesos reales | `check_permutation_invariance`, `check_text_follows_id`, `nograd.json.contract_checks` |
| §9: `fullspace_loss.py` deja de reclamar insesgadez sin su condición; la precondición violada por `CrossBlock` está escrita y testeada | `training/python/fullspace_loss.py`, `test_the_spec_states_the_precondition_the_published_head_violates` |

**Lo que la cifra NO es:** 28 sondas escritas a mano, no la batería de 400 de
`#T-battery-dev`. El desglose sí es información accionable para `#T-ce-finetune`:
extracción 8/8 y descripciones 3/4 con ModernBERT, pero comparación de atributos
1/6 y prioridades 2/4 — un checkpoint NLI sin ajustar no resuelve aritmética ni
reglas de desempate, que es lo que el plan §5 avisaba de no dar por hecho.

## Entregado 2026-09-26 — `#T-battery-metrics` (entrada del piloto, `#honest-eval`)

| Requisito de la task | Dónde está |
|---|---|
| UNA función de reporte que todo gate del piloto invoca; nadie calcula métricas a mano por run | `eval/metrics_suite.py::report`, sello `jev.metrics.v1` |
| Accuracy forzada Y accuracy con abstención, nunca una sola; ranking y abstención se evalúan aparte | secciones `ranking` y `abstention`, publicadas lado a lado |
| Cobertura y precisión entre las respondidas | `abstention.coverage`, `abstention.precision_among_answered` |
| Macro por familia (la media global pasa con una familia a cero) | `by_family`, `macro` (+ `worst_family`); test `test_macro_sees_the_family_a_global_mean_hides` |
| Azar por K, escrito al lado de cada cifra | `chance.by_k` + `chance` de cada figura; `FIGURE_KEYS` obligatorio |
| Éxito conjunto por `variant_group`: 0 cuando sólo se acierta una mitad | `counterfactual_section()`, `test_one_half_of_a_pair_right_scores_zero` |
| Invariancia a permutaciones (control de `#T-option-text`) distinguida del seguimiento del estado y la pregunta | `permutation_invariance` / `tracking`, con `distinct_from` cruzado y `controls_are_distinct` |
| NLL, Brier y calibración, con temperatura ajustada en dev y **verificada** en test | `calibration_section()`; ajustar y verificar en el mismo corte es error C5 |
| Los pesos softmax son relativos a los candidatos ofrecidos, nunca probabilidad absoluta de verdad | `SOFTMAX_SEMANTICS`, `FORBIDDEN_KEYS` |
| Un veredicto al que le falte cualquier métrica obligatoria, el azar de su K o el n de su corte hace FALLAR al runner | regla **C5** en `check()` + `eval/gate_rules.py`; `MandatoryMetricTest` |
| Un `reading` que contradiga sus propios números FALLA, no avisa | regla **C6** en `reading_errors()` + `eval/gate_rules.py`; `ReadingCoherenceTest` |
| Las tres lecturas erróneas corregidas en disco, con la frase equivocada conservada | `artifacts/gates/T-fullspace-objective/gate.json` (`/verdict`, `/upstream_verdict`), `artifacts/gates/T-bigk-optsets/gate.json` (`/verdict`), campo `reading_corrected.was` |
| Los gates históricos que incumplen quedan marcados, no borrados | `coherence-invalid.json` (`scan --mark`), inalterado: C5 y C6 no marcan ninguno más |

Cómputo: stdlib puro, sin torch, sin GPU, sin checkpoint — este gate mide el
REPORTE, no un modelo, y no publica ninguna accuracy nuestra. C6 dispara sobre
los 3 `reading` que la task nombra cuando se restauran las frases originales (4
defectos) y sobre **nada más** en las 52 carpetas de `artifacts/gates/`.
44 tests nuevos (38 en `eval/test_metrics_suite.py`, 6 en `eval/test_gate_rules.py`).

## En curso 2026-09-27 — `#T-episode-gen` (piloto de 2 000, `#episodic-data`)

| Requisito de la task | Dónde está |
|---|---|
| Cinco familias en ES y EN con el reparto DECLARADO antes de generar | `declared_split()`; el `manifest.json` se escribe con `status: planned` y `planned_split` ANTES de la primera petición a Qwen |
| Gold numérico por regla determinista, reproducible por semilla | `comparison_gold()` / `priority_gold()`; `rule_gold_flips_on_perturb` 11/11 en el gate |
| Piloto ≥2 000 publicado con manifest, semilla y versión de generador | job canónico `datagen`, salida `artifacts/episodes-qwen/pilot-2k` (2 100 = 2 000 + 5 % de holgura) — **en vuelo**, ver abajo |
| La prosa de Qwen no puede romper el contrato que el episodio debe cumplir | el tramo de `evidence` viaja en la petición (`KEEP:`) y se verifica al volver con `EC.evidence_is_fragment()`; si no sobrevive, se publica la prosa local, no un rechazo |
| El volumen se firma con la cifra MEDIDA, nunca con una prometida (C7) | `_volume_check()` lee `n_published`, `seed`, `generator_version` y el sha256 del `manifest.json` del propio directorio; sin manifest publicado se queda en `null` |
| Un lote no puede ser más laxo que una petición suelta | `_teacher_verdict()` compartido; `test_batch_12_publishes_exactly_what_batch_1_publishes` |
| Un item que falte en una respuesta agrupada es rechazo con motivo, no un hueco desplazado | `_numbered_lines()`, `test_a_skipped_item_is_none_not_a_shift`, `test_a_missing_item_rejects_with_reason_not_silently` |

Cómputo: el generador pedía **una petición a Ollama por episodio** y
`qwen3.6:27b-mlx` cuesta ~19 s de sobrecarga fija por petición contra ~3 s de
cómputo (medido 2026-09-27) — 18 h de GPU para el piloto, con concurrencia 4
dando sólo ×1,5. Agrupando 16 episodios por petición baja a ~13 s/episodio
(prosa 6,0 + profesor 7,2 medidos a lote 12). El primer tramo real destapó que
el camino `--prose qwen` nunca se había ejercitado: 46 de 100 episodios caían
con `evidence is not a fragment of the state` porque Qwen reescribía el tramo
que el contrato exige literal, y la familia de extracción caía entera a prosa
local por una comparación sensible a mayúsculas. Arreglado y medido contra el
modelo real: 20 publicados, 0 rechazos, 18 con prosa de Qwen, **11,0
s/episodio → ETA ~6,5 h**. El piloto NO cabe en un turno: la task sigue
`active` con `outcome: partial` y el volumen SIN firmar hasta que el job
termine. 17 tests nuevos en `data/test_episode_gen.py` (29 en total).

## Entregado 2026-09-26 — `#T-episode-contract` (entrada del piloto, `#episodic-data`)

| Requisito de la task | Dónde está |
|---|---|
| Esquema versionado (`episode-v1`) con `state`, `question`, `candidates[]` (id opaco + texto), `answer` / `acceptable_answers` / `preference`, `evidence`, `family`, `lang`, `origin`, `generator_seed`, `generator_version`, `variant_group`; sha en cada manifest (`manifest_record`) | `data/episode_contract.py`, `artifacts/gates/T-episode-contract/contract.json` |
| Validador que rechaza CON MOTIVO los 4 casos: sin `evidence`, sin `variant_group`, IDs no opacos, ID de dataset por pregunta | `validate()`, `data/test_episode_contract.py::RejectionTest` |
| Cinco familias como enumeración cerrada, idéntica a `model.ce_scorer.FAMILIES` (test que falla si divergen) | `FAMILIES`, `test_families_are_closed_and_match_ce_scorer` |
| Contrato de contexto comparativo escrito POR FAMILIA y explícito (comparación y prioridad exigen bloque con todos los candidatos; el resto juzga de una en una) | `COMPARATIVE_CONTEXT` + `COMPARATIVE_WHY`, igual valor que `model.ce_scorer` |
| `canonical_question()` / un id de dataset como `question` es INVÁLIDO, no degradado (misma cola `-\d+$` que el trainer + prefijos de dataset) | `is_dataset_id_question()` |
| Fixture a mano de 24 episodios ES+EN (12 grupos × 2, cada caso con contrafactual) que pasa el validador entera | `data/episode_fixture.jsonl` |
| Dos episodios del mismo `variant_group` nunca caen en cortes distintos (reparto por grupo + auditoría; inyección deliberada falla) | `assign_split()`, `split_by_group()`, `check_episodes_no_group_split()` |

Cómputo: validador + repartidor stdlib-only sobre la fixture (milisegundos);
sin entreno, sin inferencia, sin GPU. 10/10 checks medidos en
`artifacts/gates/T-episode-contract/contract.json` (`status: measured`).

## Cerrado 2026-09-26 — `#T-ce-mechanics` (gate FIRMADO, 0,958 de sobreajuste)

| Requisito de la task | Dónde está | Estado |
|---|---|---|
| Conjunto de 32–64 casos inequívocos, semilla fija y comando reproducible | `training/python/ce_overfit.py` (48 casos, huella `aadbb6a235c2361e`, semilla 20260926) | hecho |
| Gold donde el trainer cree: resuelto por id, y recalculado por la regla de la familia sobre los números del texto renderizado (36/48; los 12 de inferencia son a mano y el artefacto lo dice) | `validate_set`, `numbers_from_text`, `gold_indices`; tests de permutación **con teeth** | hecho |
| El tokenizado no corta el estado: `truncation="only_first"`, ventana 512, y un par que no cabe **para** el run | `model.ce_scorer.encode_pairs`, `length_report`, `test_ce_overfit.py` | hecho |
| Máscara de candidatos: relleno a probabilidad exactamente 0, K=3 igual sola que mezclada con K=4, sin gradiente a columnas que no existen | `stack_scores`, `test_a_k3_row_scores_the_same_alone_as_batched_with_k4` | hecho |
| Formato de hipótesis **idéntico** en entreno y en evaluación, comparando los strings de los dos caminos de verdad | `test_train_and_eval_render_the_same_strings` (entreno vs `eval.ce_nograd.measure_one` con juguete) | hecho |
| Formato **congelado** en un único sitio importable, y ningún segundo sitio en el árbol | `model.ce_scorer.flatten_pairs` / `encode_pairs` / `format_fingerprint()` = `b215e3003cc60c0f`; `test_no_second_hypothesis_template_lives_in_the_tree` | hecho |
| >95 % de sobreajuste sobre los 48, y el mismo checkpoint sin ajustar claramente por debajo | `artifacts/gates/T-ce-mechanics/overfit.json` (`status: measured`, `gate.pass: true`): 46/48 = **0,958** vs 12/48 = **0,250**, distancia **0,708** | **medido** |

**El gate está FIRMADO** con pesos reales en MPS: 46/48 = 0,958 de sobreajuste
(umbral `>0,95`), el mismo checkpoint sin ajustar 12/48 = 0,250 (umbral
`<=0,60`), distancia 0,708 (umbral `>=0,35`), y los cuatro chequeos de tubería
(gold, máscara, formato entreno/eval, truncado) en PASS sobre el run real. 60
épocas en 117,33 s, semilla 20260926, checkpoint `minilmv2-l6-mnli-xnli`, huella
del conjunto `aadbb6a235c2361e`. Reproducible con el campo `command` del
artefacto:

    PYTHONPATH=. .venv-train/bin/python -m training.python.ce_overfit overfit --device mps --seed 20260926 --epochs 60 --lr 2e-05 --decisions-per-batch 8 --eval-every 5 --weights minilmv2-l6-mnli-xnli

Esa cifra **no** mide generalización: se entrena y se mide sobre los mismos 48
casos, a propósito — la primera con significado externo es `#T-ce-finetune`
contra `#T-battery-dev`.

## Cerrada 2026-09-27 — `#T-preflight-refs` (cuatro columnas medidas, GO)

| Requisito de la task | Dónde está | Estado |
|---|---|---|
| Runner de las tres referencias, ejecutable: Qwen local por elección estructurada entre ids válidos, el scorer NLI sin ajustar por entailment y los dos checkpoints pointer como control | `eval/preflight_refs.py` (`qwen_column`, `nli_column`, `pointer_column`, `run`) | hecho |
| Mismo corte, mismas filas, mismo protocolo; el runner **falla** si los n no coinciden | `same_rows()` → `ProtocolMismatch`; las cuatro columnas medidas comparten `rows_sha256` `8e8ccea5…8535c3fb`, n=400 | hecho |
| Protocolo de información equivalente de `#T-battery-dev`: un formato por modelo, fijado antes de medir | `protocol()`, huella `b215e3003cc60c0f` de `model.ce_scorer`, contrato comparativo por familia | hecho |
| Toda cifra por `eval.metrics_suite.report()`, cero aritmética a mano; C7 pasa sobre el gate | `report_for()` + `MS.require()`; `eval.gate_rules check artifacts/gates/T-preflight-refs` → `pass: true`, 0 errores, `hand_computed: 0`, 4 informes `jev.metrics.v1` | hecho |
| Ninguna referencia consulta el test sellado | `assert_not_sealed()`, `load_cut()` exige prefijo `dev-`; `data/battery_sealed.jsonl` existe en disco y sigue sin abrirse | hecho |
| Tests en segundos, sin torch en el camino caliente | `eval/test_preflight_refs.py`, 49 tests en 8,3 s; `NoModelTest` lo comprueba en un intérprete limpio | hecho |
| Tabla de referencias (accuracy forzada, con abstención, macro por familia, azar por K, IC95 %, n) | `artifacts/gates/T-preflight-refs/refs-{nli-nograd,qwen-local,pointer-control-*}.json`, 4 columnas sobre un solo corte | **medido** |
| Puerta de Qwen resuelta | `QWEN_GATE_RULE` escrita antes de medir; **ABRE**: forzada 0,965 IC [0,942 · 0,979] vs azar 0,3375, macro 0,965, las 5 familias despejan su propio azar | **medido** |
| El control pointer reproduce el fallo de `probe-mechanism-2026-09-24.json` | los dos checkpoints: forzada 0,330 / 0,3275 con azar 0,3375; 0,964 / 0,929 de los 140 grupos con el mismo slot en las dos mitades; conjunto 2/140 y 1/140 vs azar conjunto 0,0826, intervalos enteros por debajo | **medido — reproduce** |
| Checkpoint de partida del ajuste, decidido con cifras | `STARTING_POINT_RULE`: **`nli-nograd`** (minilmv2-l6-mnli-xnli `@0a71e92a`), forzada 0,6225 IC [0,574 · 0,669], conjunto contrafactual 0,343 (48/140) sobre 0,0826. Los dos pointer quedan fuera: quien reproduce la invariancia es el control | **medido** |

**El gate está firmado: `verdict: PASS`, `resolved: true`, ninguna casilla en
`null`.** El veredicto de la task es **GO** — Qwen responde la batería, así que el
enunciado, los datos y el formato no están rotos y `#T-ce-finetune` puede
arrancar desde `nli-nograd`. El contaminado con BANKING77 es
`modernbert-zeroshot-v2`, que **no** es el elegido: la cifra publicada no
arrastra ese caveat.

**Lo que acota el 0,965 de Qwen y viaja con él:** su control de permutación
`#T-option-text` **falla** — 3 de 24 filas (muestra fijada por sha antes de
medir) cambian de elección al invertir el orden de los candidatos, flip_rate
0,125. El chooser muestrea a temperatura 0,7, así que ese 12,5 % acota juntos
sensibilidad al orden y ruido de muestreo sin separarlos. Se publica al lado del
veredicto (`the_qwen_gate.order_stability_beside_the_verdict`) y en cada fila de
la tabla, con `enters_the_rule: false`: la regla se escribió antes de que
existiera una medición, y añadirle una condición tras ver este fallo sería
elegirla por el resultado.

Una nota que el runner hace explícita: la sección de calibración de la suite
exige una temperatura ajustada en un corte y **verificada** en otro, y la del
producto es `#T-battery-calib` (depende de `#T-ce-finetune`, no existe). El
runner ajusta una dentro del corte de desarrollo por mitades de `variant_group`
—ninguna pareja contrafactual se reparte— y lo dice en el propio informe: no es
la calibración del producto.

Las columnas medidas quedan selladas con sus filas en
`artifacts/gates/T-preflight-refs/columns/`, así que la tabla y el gate se
rehacen sin cargar un modelo (`eval.preflight_refs table`).

## #laya-teardown — la referencia externa (2026-09-27)

| Requisito | Quién lo entrega | Estado |
|---|---|---|
| Tabla de diferencias arquitectónicas Laya vs pointer head, y cuál se prueba primero | `#T-laya-archdiff` | **entregado 2026-09-27** — `docs/laya-archdiff.md`; presupuesto 48/192 medido con su `build_sequence`: techo de formato 0,948 a K=77, no 0,425 (`artifacts/gates/T-laya-archdiff/token_budget.json`) |
| Laya sin ajustar sobre la batería privada + contrafactuales, con IC95 % | `#T-laya-baseline` | **medido**: forzada 0,595 [0,546 · 0,642], conjunto 50/140 = 0,357 [0,283 · 0,439]; `artifacts/gates/T-laya-baseline/gate.json` |
| BANKING77 a K=77 de Laya reproducido en nuestro arnés (`eval/fullspace.py`) | `#T-laya-baseline` | **medido**: 0,379 [0,362 · 0,396], azar 0,013, n 3 080 — su 0,425 publicado **no reproduce**; `artifacts/gates/T-laya-baseline/fullspace.json` |
| Declaración por dataset de si estaba en la mezcla publicada de Laya | `#T-laya-baseline` | **hecho**: gate `mix_declared_per_dataset`, citando `BENCHMARKS.md` / `README.md` / `bench_apps.py` |
| Temperatura por (tipo, K) contra nuestra global, en ECE y acc@50 % cobertura | `#T-laya-objective` | **SIN MEDIR** |
| `proper_reward` (log + esférica + RPS) comparada con nuestro CE listwise | `#T-laya-objective` | **SIN MEDIR** |

## Reordenación 2026-09-27 — un solo objetivo: aprender y mejorar cada día

Decisión del operador. Guía completa en `guia-un-solo-objetivo.md` (orden de
ejecución, reglas, bucle diario, comandos). Resumen de lo que cambia:

| Qué | Antes | Ahora |
|---|---|---|
| `oss-release`, `shared-state-distill`, `train-scaleout` (+ 10 tasks) | backlog | **archivadas** en `initiatives/log/` y `modules/<m>/log/`; nadie las despacha |
| `#T-release-gate` | blocked | backlog: no hay release en el objetivo; el veredicto lo publica `#T-ce-confirm` |
| `#laya-teardown` | next | **active** (P1): cota externa y qué copiar |
| `#teacher-distill` | backlog | **next** (P2): Jev como vara de medir y profesor; `#T-teacher-auth` blocked hasta key válida del operador |
| `#data-flywheel` | — | **nueva, active** (P1): `#T-ingest-laya` (typed-decisions, Jev 0,727 / Laya 0,766), `#T-ingest-public` (≥ 8 fuentes), `#T-jev-soft-targets` |
| `#daily-learning-loop` | — | **nueva, next** (P2, arranca con el GO de `#T-ce-finetune`): `#T-loop-trainer`, `#T-loop-scoreboard`, `#T-loop-nightly`, `#T-loop-error-mining`, `#T-loop-rl-jev` |
| `#T-episode-scale` | — | nueva (episodic-data): 2 000 → 20 000 con cuota diaria y dedup |
| `#T-ce-finetune` | spec | guía ejecutable: trainer hoy, smoke de 20 min con el piloto parcial, cifra oficial con splits, regla de escalado escrita |
| piloto `datagen` | muerto a 1 394 | `--resume` en `data.episode_gen` (tests), relanzado 12:40 |

| Requisito | Quién lo entrega | Estado |
|---|---|---|
| Trainer `ce_finetune.py` con `eval none` = 0,6225 | `#T-ce-finetune` | **SIN MEDIR** |
| Smoke ≈ 20 min: dev > nli-nograd con IC | `#T-ce-finetune` | **SIN MEDIR** |
| 5 000 → decisión sobre 20 000 | `#T-ce-finetune` | **SIN MEDIR** |
| typed-decisions convertido, test eval-only | `#T-ingest-laya` | **medido** (2026-09-27): 6 000 + 2 000 episodios, 0 rechazos, gate PASS |
| ≥ 8 fuentes públicas, volumen consumible | `#T-ingest-public` | **SIN MEDIR** |
| `history.jsonl` con ≥ 3 filas y dashboard | `#T-loop-scoreboard` | **SIN MEDIR** |
| 7 noches con regla de promoción | `#T-loop-nightly` | **SIN MEDIR** |
| Jev reproduce ≈ 0,73 en typed-decisions test | `#T-teacher-probe` | **bloqueado por credencial** |

## Entregado 2026-09-27 — `#T-laya-archdiff` (`#laya-teardown`, sólo lectura y CPU)

Documento `docs/laya-archdiff.md`: once diferencias Laya vs pointer head vs cross-encoder, con
qué hace cada sistema, si es estilo o candidata a explicar 0,425 vs 0,0123 en BANKING77 K=77, y
qué costaría medirla aquí. Una sola elegida para probar primero (D1, las opciones dentro de la
secuencia del encoder), con predicción escrita antes de medir: cross-encoder `nli-nograd` SIN
ajustar sobre el corte de desarrollo de BANKING77 a K=77; GO si el límite inferior del IC95 %
≥ 0,05 (≈ 4× azar 0,013), a ejecutar dentro de `#T-laya-baseline` sobre las mismas filas.

| Requisito | Medido | Estado |
|---|---|---|
| Presupuesto 48/192 de Laya sobre las 77 etiquetas, con su propio `build_sequence` y el tokenizador de ModernBERT | cap 4 tokens/opción incl. `[MASK]` → 3 de texto; 47/77 intactas, 73/77 distinguibles, 3 grupos de colisión (7 etiquetas); techo de formato 0,948 (uniforme) / 0,952 (frecuencia de train) | **medido** — `artifacts/gates/T-laya-archdiff/token_budget.json` |
| Ese presupuesto en nuestro scorer (`length_report`) | no existe: par más largo 62 tokens (MiniLM) / 60 (ModernBERT-zs) sobre 200 filas, 0 recortes en 512; con contexto comparativo 437/463, 0 recortes | **medido** — ídem |
| Coste de una pasada K=77 del scorer en CPU | 63,0 pares/s sobre 308 pares → ≈ 63 min el test (237 160 pares), ≈ 20 min el corte dev | **medido (ritmo), estimado (total)** |
| Ablación 192 vs 512 en el checkpoint de Laya; mmBERT; reproducción del 0,425 | — | **SIN MEDIR** → `#T-laya-baseline` |


## #data-flywheel — typed-decisions convertido (2026-09-27)

| Requisito | Quién lo entrega | Estado |
|---|---|---|
| typed-decisions train/test como `episode-v1` con manifest, revisión y sha | `#T-ingest-laya` | **medido**: 6 000 + 2 000 episodios, 0 rechazos, `artifacts/gates/T-ingest-laya/gate.json` PASS |
| Test `eval-only` con barrera que impide que entre en una mezcla | `#T-ingest-laya` | **medido**: `assert_trainable` + `NEVER_TRAINABLE` + `firewall.BENCHMARKS`, con tests |
| `eval/` lee el test por sha para compararlo con Jev 0,727 / Laya 0,766 | `#T-ingest-laya` | **hecho**: `eval.cuts.external_cut("typed-decisions", "test")`; la comparación la publica `#T-loop-scoreboard` |
| Accuracy nuestra sobre typed-decisions test | `#T-loop-scoreboard` | **SIN MEDIR** |

## Entregado 2026-09-27 — `#T-laya-baseline` (`#laya-teardown`, inferencia en CPU, sin ajustar)

Runner `eval/laya_baseline.py` (`.venv-laya`, `laya==0.3.20`; checkpoints por commit pinado
`laya@55cf4c4e` para `en`, `laya-multilingual@e4e9ddf2` para `es`), mismas 400 filas que el
preflight por `same_rows()` (`8e8ccea5…`), formato forzado `choice` con id opaco + texto.

| Requisito | Medido | Estado |
|---|---|---|
| Pares contrafactuales, éxito conjunto (140 grupos) | 50/140 = 0,357 [0,283 · 0,439], azar 0,083; en 0,429, es 0,286; seguimiento 74/140 | **medido** — empata con nli-nograd (0,343), gana al pointer (1–2/140), pierde con Qwen (0,879) |
| Batería completa forzada, macro por familia, por idioma | 238/400 = 0,595 [0,546 · 0,642], azar 0,3375; macro 0,595; en 0,660, es 0,530; comparación de atributos 0,325 y prioridad 0,363 no despejan el azar (las mismas dos de nli-nograd) | **medido** — empata con nli-nograd (0,6225), gana al pointer (0,33), pierde con Qwen (0,965) |
| Control de permutación (opciones al revés, 400 filas) | 128/400 cambian de id (0,32), delta máx. 0,889 | **medido — FALLA**: Laya lee la posición; nli-nograd 0,0 |
| BANKING77 K=77 en `eval.fullspace` (3 080 filas test, checkpoint inglés, presupuesto por defecto) | 0,379 [0,362 · 0,396], azar 0,013, n 3 080 — su 0,425 publicado **no reproduce** | **medido** — `fullspace.json`, lectura del corte reservado en el ledger |
| Mezcla publicada por dataset | batería y BANKING77 fuera; spam/phishing/RAG/triage/AG News/BoolQ dentro | **declarado** con cita de fichero |
| Barrido de cardinalidad; `nli-nograd` a K=77 (predicción D1 de `#T-laya-archdiff`); `laya-typed-decisions` | — | **SIN MEDIR** → `#T-laya-objective` |

## Medido 2026-09-27 — `#T-ce-finetune` sección B (smoke) y `#episodic-data`

| Requisito | Quién lo entrega | Estado |
|---|---|---|
| Piloto de 2 000 episodios con Qwen | `#T-episode-gen` | **done**: 2 092 publicados, `artifacts/gates/T-episode-gen/gen.json` |
| Verificador separado + cuarentena + embudo | `#T-episode-verify` | **medido**: acuerdo 0,9355 [0,924 · 0,945], 1 840 verificados / 252 cuarentena; **muestra humana pendiente del operador** |
| Trainer con `eval none` = 0,6225 | `#T-ce-finetune` | **hecho**: 249/400 exacto, mismas filas |
| Smoke: dev > nli-nograd con IC, sin caer familia/idioma | `#T-ce-finetune` | **NO-GO** en 4 variantes: contrafactual +0,121 [0,043 · 0,200] y forzada +0,055 [0,015 · 0,098] en la primera, pero comparación de atributos y prioridad caen bajo el control; atributos no se aprende ni en el holdout del piloto (0,574 → 0,574). No se escala; brazo siguiente lo decide el operador |

## Reordenación 2026-09-27 (tarde) — se escala sin parar: el bucle infinito

Decisión del operador tras el smoke. El manual de operación es
`bucle-infinito.md`. `#T-ce-finetune` pasa a done: su sección C queda absorbida
por el bucle. `#daily-learning-loop` pasa a active.

| Requisito | Quién lo entrega | Estado |
|---|---|---|
| Qwen 3.8 medido en dev contra el 3.6 (0,965) | `#T-qwen38-ref` | **done · GO**: 400/400 filas, forzada **0,9825** [0,9643 · 0,9915] contra 0,9650; diferencia pareada +0,0175 IC95 % [0,0000 · 0,0375] → *no peor* (`better: false`, y se dice así). Ninguna familia cae; pasa el control de permutación a 0,7 y a 0, que el 3.6 fallaba. El productor se queda en `qwen3.8:27b-mlx` |
| Generador por regla de atributos/prioridad con volumen ilimitado | `#T-numeric-gen` | **medido** (`data/rule_variety.py`, 75 tests sin GPU): lote 1 de **50 000** en 6,7 s (7 441 ep/s, 0 rechazos, 0 inválidos), 15 atributos, 49 combinaciones de formato numérico, 7 formas de pregunta (superlativo, ordinal, umbral, empate con desempate; 2 y 3 criterios encadenados, umbral+cadena), K 2/3/8 y ES/EN. Gate en `artifacts/gates/T-numeric-gen/gate.json` |
| Contrafactuales por grupo: número y criterio/orden mueven el gold, la paráfrasis no | `#T-numeric-gen` | **medido**: 12 500 grupos, 12 500/12 500 en las tres variantes, y el gold de las 50 000 filas se recomputa desde `rule_trace` |
| Posición del gold uniforme (χ² escrita antes) y sin fuga contra dev **y** sellado | `#T-numeric-gen` | **medido**: χ² 0,00 (K=2), 4,57 (K=3), 10,91 (K=8) contra críticos 6,64 / 9,21 / 18,48 a α=0,01; 0 coincidencias de vocabulario y 0 de paráfrasis en las dos baterías (2 440 y 3 660 textos bloqueados, muestra de 800) |
| Prueba de valor: smoke con 5 000 por regla + el piloto verificado, predicción escrita antes | `#T-numeric-gen` | **done · NO-GO**, y es el resultado útil: `attribute_comparison` **no se mueve** con 5× su propio volumen — 0,5625 → 0,5625 en el holdout del piloto (Δ 0,000 [−0,172 · 0,172]) y 0,3706 → 0,4025 en las 564 filas del holdout entero ([−0,014 · 0,080]). En la misma corrida `description_classification` 0,558 → **1,000** y `extraction_paraphrase` 0,739 → **1,000**: el trainer y los datos funcionan. **El cuello es el backbone** → `#T-backbone-ladder` |
| **Cuello identificado con medida, no con opinión** | `#T-numeric-gen` → `#T-backbone-ladder` | **done**: la predicción estaba escrita antes de mezclar (`value_proof_prediction`) y acertó en atributos. **Falló en la otra mitad**: predecía que `priority_decision` subiría y se queda en 0,6118 → 0,6118 en el holdout del piloto (n=85) y CAE en dev (0,3750 → 0,3250) — la subida del piloto (0,62 → 0,78) no se reproduce. Las dos familias numéricas se comportan igual, que refuerza el diagnóstico en vez de debilitarlo. Se salta la espera del bucle (`docs/bucle-infinito.md` §4, r = 3) |
| Restricción de ventana medida antes que la conclusión | `#T-numeric-gen` | **medido**: con K=8 y 3 atributos por opción, 419 de 17 328 pares pasaban de 512 tokens (máx. 570, tokenizador de minilmv2-l6); con `rule_variety.MAX_STATE_ATTRS = {2:4, 3:3, 8:2}`, máx. 482 y 0 recortados. El techo del backbone salió en la longitud antes que en el acierto |
| Mezcla multi-fuente con tope de repetición y `consumed_ids` | `#T-loop-trainer` | **SIN MEDIR** |
| Marcador con ciclo, escalón, brazo y delta contra el control original | `#T-loop-scoreboard` | **SIN MEDIR** |
| Productor sin fin 24 h sin intervención | `#T-episode-scale` (`data.stream`) | **SIN MEDIR** |
| 10 ciclos reales con cambio de escalón y de brazo | `#T-loop-nightly` (`training.python.loop`) | **SIN MEDIR** |
| Dos peldaños de backbone medidos (control + smoke) | `#T-backbone-ladder` | **SIN MEDIR · ACTIVA**, es ahora la prioridad: la recibe del NO-GO de `#T-numeric-gen`. Peldaños candidatos mDeBERTa-v3-base (≈ 280 M) y XLM-R-large / ModernBERT-large (≈ 400 M) frente a los 107 M de hoy |
| Scorer listwise con permutación dentro de la tolerancia | `#T-listwise-format` | **SIN MEDIR** |
| Dev v2 y columna de regresión | `#T-dev-rotation` | **SIN MEDIR** |
| Hitos H1–H4 (Jev 0,727, Laya 0,766, dev ≥ 0,70, ≥ 0,85) | bucle → `#T-ce-confirm` | **SIN MEDIR** |
