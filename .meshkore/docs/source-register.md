---
title: Source register — models and datasets
updated: 2026-09-19
owner: architect-master
---

# Source register

Registro operativo para saber qué bajar, desde dónde, con qué licencia y
para qué experimento. La revisión final se fija por `revision`/commit y SHA-256
en el manifest de #T-data-schema; una URL viva nunca basta para reproducir un
run. Esto no sustituye una revisión legal.

## Reglas de admisión

- `train`: licencia y procedencia aprobadas; atribución y share-alike se
  conservan en dataset/model cards.
- `eval-only`: jamás entra en prompts de teacher, hard negatives, selección de
  hiperparámetros ni training.
- Mirrors facilitan la descarga, pero la card debe enlazar también la fuente
  original. Todo artefacto guarda repo, revisión, hash y transformación.
- Ningún dataset se redistribuye mezclado por defecto; se publican adapters y
  manifests salvo que cada licencia permita expresamente redistribuirlo.

## Backbones del bake-off

| Modelo | Fuente canónica | Licencia declarada | Papel |
|---|---|---|---|
| Ettin 17M–400M | `jhu-clsp/ettin-encoder-{17m,32m,68m,150m,400m}` en [Hugging Face](https://huggingface.co/jhu-clsp/ettin-encoder-150m) | MIT | Familia principal para curva Pareto |
| ModernBERT 149M | [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) | Apache-2.0 | Baseline long-context |
| NeoBERT 250M | [chandar-lab/NeoBERT](https://huggingface.co/chandar-lab/NeoBERT) | MIT | Candidato secundario; requiere `trust_remote_code` auditado |
| LFM2.5 Encoder 230M | [LiquidAI/LFM2.5-Encoder-230M](https://huggingface.co/LiquidAI/LFM2.5-Encoder-230M) | LFM Open v1.0 | Candidato multilingüe; uso comercial condicionado a revisar la licencia vigente |

## Datos P0

| Dataset | Fuente de descarga | Licencia declarada | Transformación V1 | Uso |
|---|---|---|---|---|
| HuffPost News Category | Original: [Kaggle `rmisra/news-category-dataset`](https://www.kaggle.com/datasets/rmisra/news-category-dataset); mirror operativo: [`khalidalt/HuffPost`](https://huggingface.co/datasets/khalidalt/HuffPost) | El mirror declara CC0; derechos/procedencia del contenido HuffPost requieren fence propio | `state=headline+description`; pregunta de categoría; subconjunto dinámico de labels | Train solo para smoke de dynamic-choice; no prueba razonamiento ni licencia comercial final |
| Banking77 | [`PolyAI/banking77`](https://huggingface.co/datasets/PolyAI/banking77) y [repo original](https://github.com/PolyAI-LDN/task-specific-datasets) | CC-BY-4.0 | Utterance → intent entre labels dinámicas | Train; núcleo de semantic label transfer |
| BoolQ | [`google/boolq`](https://huggingface.co/datasets/google/boolq) y [repo original](https://github.com/google-research-datasets/boolean-questions) | CC-BY-SA-3.0 | Passage+question → `yes/no` | Train con obligaciones SA documentadas; boolean V1 |
| Civil Comments | [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/civil_comments) / [`google/civil_comments`](https://huggingface.co/datasets/google/civil_comments) | CC0-1.0 | Comment → preguntas booleanas de toxicidad; scores continuos se reservan para V2 | Train; medir slices de identidad y sesgo |
| HelpSteer2 | [`nvidia/HelpSteer2`](https://huggingface.co/datasets/nvidia/HelpSteer2) | CC-BY-4.0 | Prompt+respuesta → cinco preguntas choice con opciones 0–4; comparación pairwise cuando existan pares | Train; calibración/preferencia, no lenguaje generativo |

## Transferencia, OOD y P1

| Dataset | Fuente | Licencia declarada | Decisión |
|---|---|---|---|
| CLINC150/OOS | [`DeepPavlov/clinc150`](https://huggingface.co/datasets/DeepPavlov/clinc150) + fuente original registrada en la card | Sin licencia clara en el mirror | `eval-only` hasta revisión; holdout de semantic transfer y OOD |
| MASSIVE | [`AmazonScience/massive`](https://huggingface.co/datasets/AmazonScience/massive) | CC-BY-4.0 | Train P1 para ES+EN y transferencia multilingüe |
| LogiQA 2.0 | [`datatune/LogiQA2.0`](https://huggingface.co/datasets/datatune/LogiQA2.0) | MIT declarada en la card | Train P1 tras pasar license/leakage fence |
| ReClor | [`sxiong/ReClor`](https://huggingface.co/datasets/sxiong/ReClor) | MIT declarada en la card | Train P1 tras reservar un holdout y revisar procedencia |
| ANLI | [`facebook/anli`](https://huggingface.co/datasets/facebook/anli) | No comercial según el plan maestro; verificar por revisión | Research/eval-only; no pesos comerciales |

## Firewall de evaluación

MMLU-Pro, GPQA, SimpleQA, MuSR, RewardBench 2, ARC Challenge y OpenBookQA
son `eval-only`. Sus URLs canónicas están en el §179 del plan maestro; la
revisión exacta y hashes se fijan en #T-firewall. La lista se niega por hash y
por similitud antes de aceptar datos sintéticos o de teacher.

## Veredicto de adecuación

La mezcla es adecuada como punto de partida para un modelo de decisión porque
cubre labels dinámicas, booleanos, preferencia, OOD, dominio y transferencia.
No es suficiente por sí sola: HuffPost solo valida mecánica; la hipótesis de
"decision foundation model" depende del holdout completo, hard negatives,
datos programáticos/de dominio y gold humano definidos en el roadmap.
