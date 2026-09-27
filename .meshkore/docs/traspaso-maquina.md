# Traspaso de máquina — qué viaja en git y qué hay que rehacer

Escrito el 2026-09-27 al mover el proyecto del MacBook Pro (M-anterior) a un M5.
Vale para cualquier máquina nueva. La conclusión primero: **lo caro e
irrepetible SÍ está en git**; lo que no viaja son bytes derivados, y cada uno
tiene su comando.

## 1. Lo que ya viaja (no hay nada que hacer)

`git clone` + `git checkout main` trae todo el estado *decisivo*:

- Código entero (`data/`, `eval/`, `training/`, `model/`, `tools/`) y sus tests.
- `.meshkore/` versionado: `modules/*/tasks`, `roadmap/initiatives`, `docs/`,
  `public/jobs.yaml` (los seis jobs canónicos con su `command` exacto).
- **Los episodios del piloto verificados con Qwen** —
  `artifacts/episodes-qwen/pilot-2k/verified.jsonl` (4,8 MB, trackeado). Son
  2 100 episodios que costaron horas de profesor: no se regeneran gratis y por
  eso están en git.
- **La columna del profesor** `artifacts/gates/T-qwen38-ref/columns/*.picks.jsonl`
  (244 KB) — las 400 filas medidas fila a fila. Sin esto la comparación 3.6 vs
  3.8 habría que pagarla otra vez.
- `data/battery_dev.jsonl` (la batería de desarrollo) y los manifests de los
  cortes sellados en `artifacts/splits/`.
- Los 455 ficheros trackeados bajo `artifacts/`: cada `gate.json`,
  `manifest.json` y `train_manifest.json`. Eso es lo que hace verificable un
  resultado; los pesos no.

## 2. Lo que NO viaja, y con qué se rehace

| Qué | Tamaño | Cómo se rehace | ¿Bloquea? |
|---|---|---|---|
| `.venv-train/` | 1,1 GB | `python3 -m venv .venv-train` + `make torch-env` (instala desde `requirements-train.txt` con `--require-hashes`) | no |
| `artifacts/weights/` | 1,5 GB | `make torch-weights` (= `model.weights fetch` + `verify`; revisión y sha256 en `source-register.md`) | no |
| `artifacts/data-prefetch/`, `data-raw/` | 3,4 GB | `.venv-train/bin/python -m data.prefetch` | no |
| `artifacts/episodes-rule/batch-0001/episodes.jsonl` | 107 MB | job `datagen`, **6,7 s** — el `manifest.json` lleva el comando y la semilla | no |
| `artifacts/episodic-div/`, `prog_gold/`, `synth/` | 664 MB | ver la regla de cada bloque en `.gitignore` | no |
| `artifacts/checkpoints/*/model.safetensors` y `backbone.safetensors` | 5,2 GB | **no se rehacen**: son runs pasados. El peldaño siguiente entrena con `--init none`, así que no hacen falta | no |
| `artifacts/splits/*/*.ids` | 56 MB | se reconstruyen de semilla + sha256 del manifest | no |
| `.meshkore/log/` (diario) | 128 KB | **cópialo a mano si lo quieres** — está en `.gitignore` | no |
| `.meshkore/credentials/portal-token` | — | lo pone el portal de MeshKore al registrar la máquina | **sí** |
| `.secrets/` (key del profesor externo) | — | a mano; la del 2026-09-23 ya daba 401 | solo para TypeSafe |

## 3. Orden de arranque en la máquina nueva

1. `git clone` y `git checkout main`.
2. `python3 -m venv .venv-train` → `make torch-env` → `make torch-weights` →
   `.venv-train/bin/python -m data.prefetch`. Comprueba el stack con `make torch-smoke`.
3. `ollama pull qwen3.8:27b-mlx` (el productor; es el profesor decidido en
   `#T-qwen38-ref`). Ojo al hábito de `ollama`: deja el modelo cargado en GPU
   después de terminar y mete la máquina en swap — `ollama ps` antes de
   entrenar, `ollama stop <modelo>` para descargarlo.
4. Registrar los seis jobs de `.meshkore/public/jobs.yaml` con `POST /jobs`
   (crea en `running`: encadena `POST /jobs/<id>/stop` si los quieres parados).
5. `pytest data/ eval/` — el verde esperado son 566 de `data/`. Hay **tres
   rojos preexistentes** ajenos a cualquier task nueva, anotados en el diario:
   `eval/test_data_eval.py::test_ood_abstention_separates`,
   `eval/test_release_gate.py::…test_the_numbers_are_the_ones_t_unseen_labels_published`
   y el censo del piloto de `test_gate_rules.py`.
6. Job `datagen` para rehacer el lote por regla (segundos), y a por
   `#T-backbone-ladder`.

## 4. Dónde se retoma el hilo

`#T-backbone-ladder` (`active`, iniciativa `daily-learning-loop`). El peldaño 1
(MiniLMv2-L6, 107 M) está medido y su techo es real: 5 000 decisiones de la
familia que falla no movieron `attribute_comparison`
(`artifacts/gates/T-numeric-gen/gate.json`). El peldaño 2 es
`mDeBERTa-v3-base-xnli` (≈ 280 M) y lo primero que se publica de él es su
**control sin ajustar**, antes de entrenar nada. El `## Done when` de la task
lleva los cinco criterios.
