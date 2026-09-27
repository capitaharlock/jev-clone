---
id: T-ingest-public
title: Datasets públicos de decisión a episode-v1 — el adaptador genérico y las primeras ocho fuentes
status: active
priority: high
owner: unassigned
category: data
initiative: data-flywheel
depends_on:
  - T-ingest-laya
created: 2026-09-27
updated: 2026-09-27
---

# Datasets públicos de decisión a episode-v1 — el adaptador genérico y las primeras ocho fuentes

## Contexto

El repo ya tiene adaptadores al esquema universal antiguo (`data/adapters.py`,
`data/convert_*.py`, `data/schema.py`) con fence de licencia
(`data/registry.py`) y firewall de contaminación (`data/firewall.py`). Lo que
falta es un puente **esquema universal → `episode-v1`** y una lista de fuentes
elegida por lo que enseña a decidir, no por tamaño. Laya se preentrenó con una
mezcla parecida (AG News, BoolQ, spam, phishing, relevancia RAG, triage,
MASSIVE, XNLI; `TMP/laya/BENCHMARKS.md`, `README.md` §Benchmarks) y por eso
sale cerca de chance en typed-decisions zero-shot pero aprende rápido: el
volumen externo da cobertura de dominio, no sensibilidad. La sensibilidad la
dan los episodios con contrafactuales de `#episodic-data`.

## Fuentes candidatas (todas en Hugging Face; licencia POR VERIFICAR en cada card antes de convertir)

| Fuente | HF id | Tipo de decisión | Idioma | Uso previsto |
|---|---|---|---|---|
| AG News | `fancyzhx/ag_news` | 4 categorías | en | train |
| BoolQ | `google/boolq` | sí/no con pasaje (**ya en source-register**, CC-BY-SA) | en | train |
| DAIR Emotion | `dair-ai/emotion` | 6 emociones | en | train (Laya lo tiene held-out: útil para comparar) |
| SST-5 | `SetFit/sst5` | ordinal 5 | en | train (ordinal → tipo `score`) |
| MASSIVE | `AmazonScience/massive` | intención, 60 etiquetas (**ya en source-register**) | es+en (+49) | train ES/EN |
| XNLI | `facebook/xnli` | entailment 3 vías | es+en | train — es la familia «inferencia y negación» |
| PAWS-X | `google-research-datasets/paws-x` | paráfrasis sí/no | es+en | train — control de «paráfrasis no cambia el gold» |
| toxic-chat | `lmsys/toxic-chat` | moderación binaria | en | train (Laya held-out, 0,53: dominio duro) |
| deepset prompt-injections | `deepset/prompt-injections` | binaria | en | train |
| Enron spam | `SetFit/enron_spam` | binaria | en | train |
| COPA (SuperGLUE) | `super_glue` config `copa` | causa/efecto, 2 opciones | en | train — decisión con estado corto |
| LogiQA 2.0 / ReClor | `datatune/LogiQA2.0`, `sxiong/ReClor` (**ya en source-register**, P1) | lectura + 4 opciones | en | train tras holdout |
| HelpSteer2 | `nvidia/HelpSteer2` (**ya en source-register**) | 5 preguntas ordinales | en | train (`score`) |
| BANKING77 | `PolyAI/banking77` | 77 intenciones | en | **eval-only** (es el benchmark de Jev y nuestro corte unseen) |

Ocho fuentes convertidas es el mínimo del Done; la tabla se amplía en el
propio fichero conforme se convierten. Español: prioriza MASSIVE, XNLI y
PAWS-X para no dejar el idioma atrás.

## Qué hacer, paso a paso

1. **Puente** `data/episode_bridge.py`: función `universal_to_episode(row,
   seed) -> dict | reject` que toma una fila del esquema universal y produce
   `episode-v1`: `state` = pasaje/texto; pregunta = plantilla por dataset
   (escrita a mano, en el idioma de la fila, **al menos tres variantes por
   dataset elegidas por semilla** para que la pregunta no sea un atajo);
   candidatos con `id` opaco barajado y `text` = nombre de etiqueta **con
   descripción corta** cuando exista (si no, el nombre); `family:
   external/<dataset>/<tipo>`; `origin`; `variant_group: <dataset>-<id>`.
2. **Por dataset:** fila en `source-register.md` (id, revisión, licencia leída
   de la card, uso), descarga fijada por revisión + sha (patrón de
   `data/convert_banking77.py`), conversión, validación, `rejects.jsonl`,
   `manifest.json` en `artifacts/episodes-external/<dataset>/<split>/`.
3. **Test publicado fuera del train:** cada dataset con split `test` lo
   conserva en su carpeta `test/` marcada `eval_only: true` en el manifest, y
   `data/leakage.py` corre entre su train y (batería dev, batería sellada,
   typed-decisions test) — resultado al manifest.
4. **Cap por dataset:** ningún dataset aporta más del 25 % de la mezcla
   externa (escríbelo en el manifest de mezcla; `#T-loop-trainer` lo aplica).
   Un millón de filas de AG News no son un millón de lecciones.
5. **Gate** `artifacts/gates/T-ingest-public/gate.json`: por dataset n train /
   n test, rechazos, licencia, resultado de fuga; total consumible.
6. Tests: el puente produce episodios válidos para una fila sintética de cada
   tipo (categórica, binaria, ordinal, con pasaje); barajado reproducible;
   un dataset sin licencia declarada es rechazado por el fence, no convertido.

## Done when

- ≥ 8 fuentes convertidas y validadas, con manifest, sha, licencia y fuga
  medida; ≥ 2 de ellas con filas en español.
- El total consumible está publicado en el gate (cifra medida, no estimada).
- Ningún corte `eval-only` puede entrar en un manifest de mezcla (test).

## Qué NO hacer

- No inventar descripciones de etiquetas «para ayudar»: si el dataset no las
  trae, `text` es el nombre y punto. Documenta qué datasets las tienen.
- No convertir sin leer la card de licencia. Si duda, `eval-only` y se anota.

## Estado 2026-09-27 — a medias (el agente se cortó por límite de sesión)

`data/episode_bridge.py` (1 744 líneas: specs de fuentes, fence de licencia, puente
universal → `episode-v1`, cap de mezcla, fuga) está escrito e importa limpio, **sin tests, sin
ejecutar y sin ninguna fuente convertida**. Quien retome: leer el módulo entero, escribir
`data/test_episode_bridge.py` según el paso 6, convertir las fuentes y firmar el gate. Nada de
lo que hay en el módulo se ha validado contra datos reales.
