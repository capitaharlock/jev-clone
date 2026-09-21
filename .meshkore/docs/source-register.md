---
title: Source register — models and datasets
updated: 2026-09-21
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

## Local prefetch cache (#T-prefetch)

Todas las fuentes se precargan una sola vez al arrancar el trabajo de modelo
y se convierten al schema universal en cuanto están en disco, sin esperar al
fin del proyecto. No redescargar una revisión ya cacheada.

- HF hub cache: `~/.cache/huggingface/hub` (snapshots `models--<org>--<name>`,
  `datasets--<org>--<name>`; la revisión exacta por fichero la fija
  `artifacts/data-prefetch/manifest.json`).
- Convertidos (schema universal, splits train/calibration/test/ood):
  `artifacts/data-prefetch/<id>.jsonl` (`huffpost`, `banking77`, `boolq`,
  `civil-comments`, `helpsteer2`); P1 como dump normalizado `<id>.raw.jsonl`
  (`massive` solo locales `en-US`+`es-ES`, `logiqa20`, `reclor`).
- Backbones en cache: `jhu-clsp/ettin-encoder-{17m,32m,68m,150m,400m}`,
  `answerdotai/ModernBERT-base`, `chandar-lab/NeoBERT`,
  `LiquidAI/LFM2.5-Encoder-230M`.
- `DeepPavlov/clinc150` se precarga pero es `eval-only`: snapshot sin
  conversión a train, jamás entra en un manifest de entreno.
- Conversor: `data/prefetch.py` (adapters P0 de `data/adapters.py`; fence +
  firewall se revalidan en `T-data-p0`, el prefetch no los sustituye) más
  conversores snapshot (stdlib, sin datasets/pandas):
  `data/convert_huffpost.py` (News_Category_Dataset_v2.json del snapshot,
  universo P0 de 8 categorías, split train/test 90/10 determinista por
  sha1),
  `data/convert_banking77.py` (CSVs crudos de GitHub en una sola llamada —
  el adapter construye el universo de la llamada, por filas fallaría todo),
  `data/convert_massive.py` (stremea en-US/es-ES del tarball S3 sin
  extraer; intent v1, slots en crudo para luego).
- Raw crudos (no redescargar): `artifacts/data-raw/banking77/{train,test}.csv`
  (github PolyAI-LDN), `artifacts/data-raw/massive/amazon-massive-1.1.tar.gz`
  (S3), snapshot HuffPost ya en hub cache.
- Estado 2026-09-20 13:20 UTC: #T-prefetch `done` — manifest 11 jobs
  (6 convertidos a universal + dumps P1 + snapshots). Baseline v2
  publicada (boolq 0,68 / helpsteer2 0,32 / civil 0,93) en
  `artifacts/runs/20260920T110820Z/metrics.json`; v3 en marcha sobre los
  6 sets (`artifacts/runs-baseline-v3.log`). Conversor Qwen sigue en
  fondo (`artifacts/data-qwen/`, ~103 variantes).
- Conversor Qwen en marcha (continuo, `data/qwen_convert.py`,
  `qwen3.6:27b-mlx` vía ollama localhost:11434, `think:false` — con
  thinking el content vuelve vacío): lee `artifacts/data-prefetch/*.jsonl`,
  escribe variantes a `artifacts/data-qwen/<id>.aug.jsonl` con offsets
  reanudables en `_offsets.json`; CLINC150 excluido por fence.
  Venv del prefetch: `/tmp/prefetch-env` (efímero; recrear con
  `python3 -m venv` + `pip install datasets huggingface_hub`).
  Logs: `artifacts/logs/` (ignorados en git).

## Pesos locales verificados (#T-torch-stack)

Descargados de verdad el **2026-09-21** a `artifacts/weights/<id>/` (fuera de
git; el `manifest.json` de cada uno sí se versiona). La revisión está fijada
por commit, no por rama, y `model/weights.py::verify` re-hashea cada fichero
antes de permitir cargarlo: un sha que no cuadra **impide** la carga, no avisa.

| Modelo | Repo @ revisión | Licencia | Fichero de pesos · sha256 | tokenizer.json · sha256 |
|---|---|---|---|---|
| ettin-68m (68,1 M params) | `jhu-clsp/ettin-encoder-68m` @ `ac19ae4bc51093b31c475665ac872a936d056cc2` | MIT | `pytorch_model.bin` · `97ef650cbe35e69c363f28edbbc30be68912130d6a909de3a05714bd7bb1217d` (273,9 MB) | `9fd55248d51d33976b324fc11592e28071da7d41e0e9401dfb7082e30574b7b1` |
| modernbert-base (149,0 M params) | `answerdotai/ModernBERT-base` @ `8949b909ec900327062f0ebf497f51aef5e6f0c8` | Apache-2.0 | `model.safetensors` · `340ac08b74eef0d7bdec2d7981a6a3d4249bf0e6aab60634b72ad02c2b8023a9` (598,6 MB) | `9fd55248d51d33976b324fc11592e28071da7d41e0e9401dfb7082e30574b7b1` |

`config.json`, `tokenizer_config.json` y `special_tokens_map.json` van en el
mismo manifest con su sha. Rehacer la descarga:
`.venv-train/bin/python -m model.weights fetch` (gated por
`JEV_ALLOW_DOWNLOAD`/`JEV_OFFLINE`; `HF_TOKEN` solo haría falta para un repo
gated — estos dos son públicos). Verificar: `... -m model.weights verify`.

LFM2.5-230M y NeoBERT-250M siguen SIN descargar: el primero por licencia
condicionada, el segundo por `trust_remote_code` sin auditar. Ninguno de los
dos entra en un Pareto hasta que se resuelva eso (#T-bakeoff-real).

## Veredicto de adecuación

La mezcla es adecuada como punto de partida para un modelo de decisión porque
cubre labels dinámicas, booleanos, preferencia, OOD, dominio y transferencia.
No es suficiente por sí sola: HuffPost solo valida mecánica; la hipótesis de
"decision foundation model" depende del holdout completo, hard negatives,
datos programáticos/de dominio y gold humano definidos en el roadmap.
