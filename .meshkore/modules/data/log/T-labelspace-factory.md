---
superseded_by: episodic-data / T-episode-gen
id: T-labelspace-factory
title: Fábrica de espacios de etiquetas con el Qwen local
status: superseded
priority: high
owner: unassigned
category: data
initiative: full-space-training
created: 2026-09-24
updated: 2026-09-25
---

# Fábrica de espacios de etiquetas con el Qwen local

> **Retirada el 2026-09-25 (plan de recuperación §8).** Generar 2.000 espacios
> de etiquetas con las plantillas deterministas de `data/labelgen.py:297` produce
> diversidad de nombres, no diversidad de razonamiento, y el marcador de línea
> (`Field report:` / `Cross-check against:`) regala un atajo. El trabajo pasa a
> `#T-episode-gen`, que genera **episodios completos y verificables** con
> contrafactuales. El job `labelspace-factory` queda muerto a propósito; las 294
> entradas ya publicadas en `artifacts/labelspace-qwen/spaces.jsonl` se conservan
> como evidencia.

Un objetivo sobre el espacio entero (`#T-fullspace-objective`) sobre un corpus
donde **9 taxonomías cubren el 83,1 % de las filas** aprende nueve espacios muy
bien. La diversidad de espacios deja de ser la hipótesis lateral que era en
`#T-labelspace-div` y pasa a ser materia prima del objetivo: hacen falta muchos
espacios distintos, cada uno usado **entero** como denominador.

## Qué hay ya en disco (no rehacer)

- `artifacts/episodic-div/episodic-div-*.jsonl` — **20 000 mini-taxonomías**
  (~52 filas cada una, 1 M de filas), generadas por `data/episodic.py`. Existen
  y nunca se entrenaron bajo el objetivo nuevo; son el primer material.
- `artifacts/data-qwen/*.aug.jsonl` — ~16 MB de paráfrasis ya generadas con el
  Qwen local (`data/qwen_convert.py`): banking77 13 083 filas, civil-comments
  4 906, boolq 2 141. Aumentan el **texto**, no el espacio de etiquetas.
- `artifacts/synth/v1/` — 6 shards del corpus sintético con adjudicación.

## Qué falta

**1. Levantar el Qwen.** `ollama` está instalado (`/opt/homebrew/bin/ollama`)
con `qwen3.6` y `embeddinggemma` en `~/.ollama/models`, pero **el servidor no
escucha en :11434** (comprobado 2026-09-24; el túnel `meshkore-ollama` de
cloudflared sigue vivo apuntando a nada). Arrancarlo **como job del daemon**,
nunca dentro de un turno.

**2. Generar espacios, no paráfrasis.** `data/qwen_convert.py` pide al modelo
una reescritura por fila. Lo que hace falta es lo contrario: pedirle **una
taxonomía** — un dominio, N etiquetas hermanas con sus definiciones, y las
confusiones plausibles entre ellas — y después poblarla. El diseño barato es
generar el espacio una vez y las filas con plantilla, no una llamada por fila.

**3. Negativos duros dentro del espacio.** Con el espacio entero como
denominador, el distractor ya no se muestrea: **está**. Lo que hay que medir es
la vecindad (`data/hardneg._embed`, 3-gramas de carácter) para poder reportar
qué parte del acierto viene de descartar lo obvio y qué parte de separar
hermanos.

**4. Fence de licencias y firewall.** Todo espacio generado entra por
`data/registry.py` con su card y pasa `data/firewall.py`. Sintético del Qwen
cuenta contra `MAX_SYNTHETIC_FRACTION = 0.50` y no puede desplazar el
`MIN_HUMAN_FRACTION = 0.20`.

## Done when

- Job del daemon que mantiene Ollama vivo y un generador que produce
  **espacios de etiquetas** (dominio + N etiquetas + definiciones +
  confusiones), no paráfrasis, con caché por hash y reanudable.
- ≥ 2 000 espacios nuevos registrados con card, licencia y firewall en verde,
  y su inventario de diversidad medido al lado del corpus actual (9 taxonomías
  / 83,1 %).
- La vecindad de cada espacio medida y publicada, de forma que un gate pueda
  separar "acertó descartando lo obvio" de "separó hermanos".
- Una mezcla entrenable que combine los espacios nuevos con `episodic-div` y el
  corpus humano sin romper las cuotas de `data/mix.py`.
