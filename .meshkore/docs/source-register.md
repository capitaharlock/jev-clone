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

## Ensanche del registro para el corpus 1 M (#T-mix-1m)

Los §§65-66 dicen que ningún dataset pasa del 15 % de una mezcla y ninguna
familia del 30 %. Con siete fuentes limpias eso es aritmética, no opinión: el
registro entero admitía **39 970 filas** del millón que pide el §86, porque
`email-triage` (4 000 preguntas) no llega a llenar su 15 % y su déficit se
come el 5 % de holgura que dejan los topes. Ninguna semilla y ningún tamaño
de target mueven ese número — sólo oferta independiente.

Estas cinco fuentes son esa oferta. Ninguna toca la valla de benchmark
§§18/77 (Banking77 / HelpSteer2 / PubMedQA) ni el registro congelado de ids
Jevals, ninguna pide credenciales ni token de HF, y todas son lo bastante
grandes para sentarse **en** el tope en vez de por debajo. Con ellas el techo
sube a 1 213 741 filas y el corpus de 1 M existe.

| Dataset | Fuente de descarga | Licencia declarada | Transformación V1 | Uso |
|---|---|---|---|---|
| dbpedia14 | [`dbpedia_csv.tgz`](https://s3.amazonaws.com/fast-ai-nlp/dbpedia_csv.tgz) (fast.ai NLP mirror; original: DBpedia ontology de Zhang et al. 2015) | CC-BY-SA-3.0 + GFDL | `state=título+abstract`; pregunta de clase entre 14; 45 % en K=4 y 55 % en K 9-13 con distractores de vecindad medida | Train; segunda fuente de la familia `topic`, que es lo que la sube de una unidad de tope a dos |
| snli | [`snli_1.0.zip`](https://nlp.stanford.edu/projects/snli/snli_1.0.zip) (Stanford NLP) | CC-BY-SA-4.0 | `state=premisa+hipótesis`; dos preguntas por par: la relación a 3 vías y un booleano de entailment; los pares sin mayoría (`-`) se descartan | Train; llena la capa §86 `nli`, que no tenía oferta, y el booleano sostiene la cuota Noul §48 |
| goemotions | [`goemotions_{1,2,3}.csv`](https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv) (Google Research) | Apache-2.0 | `state=comentario`; gold = emoción ganadora por votos (≥2 y sin empate); una pregunta K=4 y otra K=12 con los vecinos medidos del gold | Train; el pool de 28 labels es donde un option set ancho es pregunta real y no cara o cruz |
| detox-attack | [comentarios](https://ndownloader.figshare.com/files/7554634) + [anotaciones](https://ndownloader.figshare.com/files/7554637) (Wikipedia Detox, Figshare) | CC0-1.0 | `state=comentario de talk-page`; gold = fracción medida de ~10 anotadores, en cubos `score 0`..`score 4`; el aspecto `third_party_attack` se balancea contra sus ceros | Train; **es la oferta Score del §48**: HelpSteer2 era el único corpus ordinal y está tras la valla, así que Score estaba al 0 % contra un target del 15 % |
| swag | [`train.csv` / `val.csv`](https://raw.githubusercontent.com/rowanz/swagaf/master/data/train.csv) (repo `rowanz/swagaf`) | MIT | `state=startphrase`; cuatro continuaciones, ids posicionales `o0..o3`; `test.csv` va sin label y se salta | Train; la capa §86 `adversarial`, que no tenía oferta ninguna |

Obligaciones share-alike: dbpedia14 (CC-BY-SA-3.0 + GFDL) y snli
(CC-BY-SA-4.0) cargan la misma que BoolQ ya cargaba, y quedan anotadas aquí
al lado de ella. Nada de esto se redistribuye mezclado: se publican el
conversor y el manifest, no las filas.

**Dureza medida, nunca afirmada.** `data.mix.is_hard` cuenta una pregunta
como dura con K ≥ 9. Emitir sets anchos llenos de labels al azar cumpliría la
letra y no enseñaría nada, así que cada fila dura toma sus distractores del
tope del ranking de `data.optset.difficulty` y se lleva puesta la similitud a
la que se construyó (`quality.nn_mean` / `nn_min`) — comprobable después, no
una bandera que el conversor se puso a sí mismo. SWAG es la excepción
declarada: sus distractores ya pasaron filtrado adversarial contra un
ensemble (Zellers et al. 2018), así que eso se anota como propiedad **de la
fuente** (`quality.source = "adversarial filtering (SWAG)"`) y no como una
dificultad que midiéramos aquí.

Descarga y conversión (stdlib, sin `datasets` ni `pandas`):

```
python3 -m data.convert_widen fetch      # qué bajar y a qué ruta
python3 -m data.convert_widen all        # convierte las cinco
```

Los bytes convertidos viven en `artifacts/data-prefetch/` y están
gitignorados como el resto (ver la tabla de #T-repo-clean más abajo). Medido
el 2026-09-21:

| Fichero | MiB | sha256 |
|---|---:|---|
| `artifacts/data-prefetch/dbpedia14.jsonl` | 605.3 | `3dfe940b4ca8e4d6129fe873fbbd7ac5c3133fd04ad67ec9f2a116b6b713e5ba` |
| `artifacts/data-prefetch/snli.jsonl` | 336.1 | `475081b03c9ba6a2580e10da677d00baa66eb5e00c6dd9895c11cf9f7b2e7bfc` |
| `artifacts/data-prefetch/goemotions.jsonl` | 66.6 | `b3ecb12219a30bfacd04b1eac1dae7f47b1199c2565e11278ab6bef9fcee2ab2` |
| `artifacts/data-prefetch/detox-attack.jsonl` | 163.8 | `91e823c38cdc1f03c806a0f3d65c5732a088b6c0d42a7b19056053ec2e0f51f4` |
| `artifacts/data-prefetch/swag.jsonl` | 67.9 | `058867e9dc06101cf6946facbd1cf21cb75ca590851bd06a713be81f222d5f88` |

Oferta limpia que aportan, en preguntas: dbpedia14 560 000 · snli 1 097 590 ·
detox-attack 166 012 · goemotions 87 076 · swag 73 546.

## Artefactos grandes — dónde viven los bytes (#T-repo-clean)

Ningún byte de esta sección está en git, y ninguno vuelve. Lo que git
versiona es esta tabla: ruta, tamaño y sha256, más cómo regenerar o volver a
bajar cada cosa. Es lo que sustituye a los `manifest.json` que hasta
#T-repo-clean viajaban dentro de `artifacts/weights/` y
`artifacts/data-prefetch/` — esos directorios están ahora ignorados enteros.

Medido el 2026-09-21 sobre
`/Users/ricartjuncadella/Documents/Prj/asimovia/jev-clone`:
**44 ficheros, 2.62 GiB**. Los hashes de pesos,
checkpoints y datasets coinciden con los `manifest.json` que los declaraban,
así que la tabla está verificada contra los bytes, no copiada de otro papel.

> **Hoy no hay copia fuera de esta máquina.** No hay bucket, ni LFS (decisión
> deliberada del plan: LFS no, registro sí). Todo lo marcado *regenerable* se
> reconstruye con el comando indicado; lo que no lo es —los checkpoints— sólo
> existe aquí. Mover eso a almacenamiento externo es trabajo pendiente, no
> algo que esta tabla resuelva.

Los pickles por corrida (`artifacts/runs/*/models/*.pkl`, ~250 MiB por run,
decenas de runs) no se registran a propósito: son la salida de
`train_baseline.py`, se regeneran en minutos y no son un artefacto que
nadie deba conservar.

### Backbones descargados

Revisión fijada: `ettin-68m` = `jhu-clsp/ettin-encoder-68m` @
`ac19ae4bc51093b31c475665ac872a936d056cc2` · `modernbert-base` =
`answerdotai/ModernBERT-base` @ `8949b909ec900327062f0ebf497f51aef5e6f0c8`.
`python -m model.weights fetch --id <id>` los vuelve a bajar y
`python -m model.weights verify` los compara contra estos sha256.

| Fichero | MiB | sha256 |
|---|---:|---|
| `artifacts/weights/ettin-68m/pytorch_model.bin` | 261.2 | `97ef650cbe35e69c363f28edbbc30be68912130d6a909de3a05714bd7bb1217d` |
| `artifacts/weights/ettin-68m/tokenizer.json` | 2.0 | `9fd55248d51d33976b324fc11592e28071da7d41e0e9401dfb7082e30574b7b1` |
| `artifacts/weights/modernbert-base/model.safetensors` | 570.9 | `340ac08b74eef0d7bdec2d7981a6a3d4249bf0e6aab60634b72ad02c2b8023a9` |
| `artifacts/weights/modernbert-base/tokenizer.json` | 2.0 | `9fd55248d51d33976b324fc11592e28071da7d41e0e9401dfb7082e30574b7b1` |

### Checkpoints entrenados aquí

No existen en ninguna fuente externa: los produjo un run de esta máquina. Si se pierden, se regeneran reentrenando con el `seed` y el `data_manifest` que lleva cada `manifest.json` — no byte a byte.

| Fichero | MiB | sha256 |
|---|---:|---|
| `artifacts/checkpoints/decision/bakeoff-ettin-68m-v1/stage-000025000/model.safetensors` | 10.3 | `24380159935a154b1fabee940fe57769f5c60b7184b874c8b0aec026ba96aa8c` |
| `artifacts/checkpoints/decision/bakeoff-ettin-68m-v1/stage-000025000/tokenizer.json` | 2.0 | `9fd55248d51d33976b324fc11592e28071da7d41e0e9401dfb7082e30574b7b1` |
| `artifacts/checkpoints/decision/bakeoff-modernbert-base-v1/stage-000025000/model.safetensors` | 11.1 | `b8941294da154c5b055beef1c25995e5e7f0e27800cbee37edaa288caa896c87` |
| `artifacts/checkpoints/decision/bakeoff-modernbert-base-v1/stage-000025000/tokenizer.json` | 2.0 | `9fd55248d51d33976b324fc11592e28071da7d41e0e9401dfb7082e30574b7b1` |
| `artifacts/checkpoints/decision/train-real-v1/stage-000062500/model.safetensors` | 10.3 | `f7dbf57eb4f61749e125141c7d7e694b32e3936a57eb0b1aeae412054933cd72` |
| `artifacts/checkpoints/decision/train-real-v1/stage-000062500/tokenizer.json` | 2.0 | `9fd55248d51d33976b324fc11592e28071da7d41e0e9401dfb7082e30574b7b1` |
| `artifacts/checkpoints/decision/train-real-v1/stage-000125000/model.safetensors` | 10.3 | `6ed13f7ec2759645c864bce5de9c27e9eb418420a0c5deea6aa947ab1b98844d` |
| `artifacts/checkpoints/decision/train-real-v1/stage-000125000/tokenizer.json` | 2.0 | `9fd55248d51d33976b324fc11592e28071da7d41e0e9401dfb7082e30574b7b1` |
| `artifacts/checkpoints/decision/train-real-v1/stage-000250000/model.safetensors` | 10.3 | `7448fa46c0d91a059bb594f95f421d5490a2dab7e09c9a979791757f81a2cc0d` |
| `artifacts/checkpoints/decision/train-real-v1/stage-000250000/tokenizer.json` | 2.0 | `9fd55248d51d33976b324fc11592e28071da7d41e0e9401dfb7082e30574b7b1` |

### Datasets convertidos (prefetch)

Derivados de las fuentes de arriba; `make prefetch` los reconstruye. El sha256 es del `.jsonl` convertido, no del original.

| Fichero | MiB | sha256 |
|---|---:|---|
| `artifacts/data-prefetch/banking77.jsonl` | 6.1 | `abfd60052afa0bbb4a119876dad653a9582d0f580ed35148491a6535735cbaca` |
| `artifacts/data-prefetch/boolq.jsonl` | 9.8 | `d6054f599f484e15a90850ad99d7478e736f522345f11fcbf06c145de3c3e54a` |
| `artifacts/data-prefetch/civil-comments.jsonl` | 1013.1 | `9524ba4638ac20f7838f4b8159db7137fcf984ad16463aff1dd24ea7f191534c` |
| `artifacts/data-prefetch/email-triage.jsonl` | 2.7 | `009b45868b0ecc7add9c564427422977b526335ba69bad03bab152ea5590e25d` |
| `artifacts/data-prefetch/helpsteer2.jsonl` | 73.4 | `5fefac5fe5ec09344ea9658b4ad1fae741a11f1771bb4d01a385958b0b89cab8` |
| `artifacts/data-prefetch/huffpost.jsonl` | 84.4 | `ddcfcb0babb6e5d8b5aaddc4a9e349fa4952836d29df494442ea0453f5d41cfd` |
| `artifacts/data-prefetch/logiqa.jsonl` | 70.9 | `37531b781ad71d772afbad87eb8598fb89283296e0cfe42ec40c58753c543e92` |
| `artifacts/data-prefetch/logiqa20.raw.jsonl` | 66.1 | `1debc735ae24a0f432e422f2135f54b8a890ddc93e98ec15e93de8c8bb45a500` |
| `artifacts/data-prefetch/massive.jsonl` | 38.9 | `c254e88d7011824b946b8c4cafb77e47cc32dc3b072a11cdbeaf9b13b3647e07` |
| `artifacts/data-prefetch/quarantine/synth-loop-20260921.jsonl` | 200.3 | `b118383b09d4778bbae041265928731ac1a94ff4c30da4bd11b984db02c0da76` |
| `artifacts/data-prefetch/reclor.jsonl` | 7.2 | `a5419e66ada782dbb45c4c64b04188f7c27c19736697776de278fd4bb0d30db9` |
| `artifacts/data-prefetch/reclor.raw.jsonl` | 6.5 | `e9e57386b836a7a8d0aad932117e9d1ced9e958283f1ed4c3f36b42e2e122136` |

### Dataset crudo

Tarball original tal cual se descargó.

| Fichero | MiB | sha256 |
|---|---:|---|
| `artifacts/data-raw/massive/amazon-massive-1.1.tar.gz` | 38.4 | `4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577` |

### Aumentos de teacher (Qwen)

Salida de un teacher local; regenerable sólo reejecutando el teacher.

| Fichero | MiB | sha256 |
|---|---:|---|
| `artifacts/data-qwen/banking77.aug.jsonl` | 8.2 | `b39cb760aa411ee2f8a71dc74d0911f2ab9c1bebeca7c9de64fa6ab31c80967f` |
| `artifacts/data-qwen/boolq.aug.jsonl` | 2.4 | `88795b7490c8364c798869bd0a350d30b14c56ac2f9066fff54ca66e379fde7a` |
| `artifacts/data-qwen/civil-comments.aug.jsonl` | 4.5 | `a9cf59d03f276ba6ff3fd86a8bbc25713477ffcbb07c0af9f3ae83a11bd96cc7` |

### Splits sellados

Listas de ids. El `manifest.json` de cada split (sí versionado) ya lleva estos hashes; aquí quedan por si el manifest se pierde.

| Fichero | MiB | sha256 |
|---|---:|---|
| `artifacts/splits/civil-comments/test.ids` | 8.1 | `6b3e4be13f2b3958f6fce3ba150dad29a84fe00a8d9a9ff0113245bb751a03bc` |
| `artifacts/splits/civil-comments/train.ids` | 32.0 | `52ab6a1968762a978b38df1be08705986ddb8fac5849308099a90cf0f7188c14` |
| `artifacts/splits/huffpost/train.ids` | 2.1 | `23850a35eabffe36ceba35ffa5494c907c8de4132fb6fa6a401b8467fe843465` |
| `artifacts/splits/logiqa/train.ids` | 1.4 | `2998b9059c36a48634da2c312ed97aa7c712aaf317e8b6f818ad09fc85783dfc` |
| `artifacts/splits/massive/train.ids` | 1.6 | `43226cc1dc9c116c839186c1ecf4110e76ad544e9a70fa6e04d35481220b2df5` |
| `artifacts/splits/synth-loop-quarantine/test.ids` | 1.0 | `4d384f71d9e8749b666917a4b459fba75b25b6226bacd739e4dc1110a396b41d` |
| `artifacts/splits/synth-loop-quarantine/train.ids` | 6.5 | `6dca6b91f946311f3f58ea7a498641880ad2585988ff81e1deb8eefa63b34788` |

### Sintético v1

Generado por `tools/data_gen_loop.py`; regenerable con su seed.

| Fichero | MiB | sha256 |
|---|---:|---|
| `artifacts/synth/v1/synth-v1-000.jsonl` | 11.1 | `a8df96950215f95fe065b95fe927c771a4fe5ec8d38bd28a3b86645b0841f076` |
| `artifacts/synth/v1/synth-v1-001.jsonl` | 11.2 | `36ed207da7b7b71203bdd32c50ad5b79884196d1ea8c5a9e2e33e69d145a3b16` |
| `artifacts/synth/v1/synth-v1-002.jsonl` | 11.1 | `accec211d3f9c76a555a8370816164daedcd7e16cf136249ddac67222c999011` |
| `artifacts/synth/v1/synth-v1-003.jsonl` | 11.2 | `62791bfe30516363f34241c5279a58e08023f4dd1f8b46821073117d52ea7547` |
| `artifacts/synth/v1/synth-v1-004.jsonl` | 11.1 | `f924ae0ebbf890d210020a20ff49d02446055b282e5319388e39c4836e5b552b` |

### Eval OOD

Particiones OOD de #T-data-eval. Grandes y NO ignoradas por .gitignore: quien haga `git add -A` las mete. El check de tamaño las rechaza.

| Fichero | MiB | sha256 |
|---|---:|---|
| `artifacts/gates/T-data-eval/ood_heldout.jsonl` | 3.9 | `24def0857dd33b12c14b758f2682f9570c9b0bed43e22590e3901e618bab90b6` |
| `artifacts/gates/T-data-eval/ood_train.jsonl` | 42.8 | `fb97cb10323d318d52b29505dd7bce69d2c1fc96abc6cc9e1983ee693504007e` |
