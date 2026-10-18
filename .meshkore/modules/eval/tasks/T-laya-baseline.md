---
id: T-laya-baseline
title: Laya sobre nuestra batería — la cota superior externa, sin entrenar nada
status: done
priority: high
owner: developer
category: eval
initiative: laya-teardown
created: 2026-09-27
updated: 2026-09-27
---

# Laya sobre nuestra batería — la cota superior externa, sin entrenar nada

Todo lo que hemos medido hasta ahora se compara contra el azar o contra nosotros
mismos. Laya es un checkpoint público que responde exactamente nuestro formato de
pregunta; pasarlo por nuestra batería convierte «no aprende» en un número con
referencia.

Alcance: `pip install laya` en un entorno aparte, `convaiinnovations/laya` y
`laya-multilingual`, **sin ajustar**. No se importa código de Laya al repo.

Lo que hay que medir, en este orden:

1. **Los pares contrafactuales del reanálisis del operador** (verde/rojo; cuál
   cuesta menos vs cuál dura más). Es la prueba barata: si Laya también elige
   igual en los dos, el fallo es del planteamiento del problema, no nuestro.
2. **La batería privada de `#honest-eval`**, misma suite, mismo sellado, una sola
   apertura, con IC95 %.
3. **BANKING77 a cardinalidad completa** en NUESTRO arnés (`eval/fullspace.py`),
   para saber si su 0,425 reproduce fuera de su harness o no.

Advertencia que va en el gate: sus checkpoints se entrenaron con mezclas que
incluyen tareas públicas; spam y phishing (0,993) están declarados «in training».
Cualquier dataset que esté en su mezcla NO es transferencia limpia y así hay que
publicarlo.

## Done when

- Hay un gate en `artifacts/gates/T-laya-baseline/` con las tres medidas, su n,
  su K, su IC95 % y la versión exacta del paquete y del checkpoint.
- Está declarado, por dataset, si estaba o no en la mezcla publicada de Laya.
- La comparación contra nuestro checkpoint corre sobre las MISMAS filas, y eso
  está comprobado por hash, no afirmado.
- Está escrito el veredicto en una línea: dónde nos gana, dónde empata y dónde
  falla igual que nosotros.

## Resolution

**done · 2026-09-27 · developer (claude-fable-5-1)** — sólo inferencia en CPU; ningún peso,
temperatura ni umbral tocado. Todas las cifras salen de `eval.metrics_suite.report()` o de
`eval.fullspace.tally()`; el gate las copia.

**Entorno y checkpoint.** `.venv-laya` (python 3.12, `laya==0.3.20`, torch 2.14.0,
transformers 5.17.0), fuera de `.venv-train` y sin importar código del clon. Checkpoints
descargados por commit pinado (los `PINNED_REVISIONS` de `laya/revisions.py` del clon; el
paquete de PyPI no admite `revision`, así que se descarga con `snapshot_download` y se carga el
directorio): `convaiinnovations/laya@55cf4c4e` (ModernBERT-large, `head_max_len` 192,
`max_len` 512, sha256 pesos `891102d3…`) para las filas `en`;
`convaiinnovations/laya-multilingual@e4e9ddf2` (mmBERT-base, 256/1024, sha256 `9d628fd9…`)
para las `es`. Formato forzado, uno, fijado antes de medir: estado tal cual, pregunta como
`instructions`, cada candidato como criterio `choice` con id opaco de etiqueta y texto de
descripción (`c1: <texto>`).

**Runner:** `eval/laya_baseline.py` (`battery` y `banking77` son los jobs; `gate` rehace el
gate sin modelo). Mismas filas que el preflight, comprobado: `same_rows()` sobre Laya + las
cuatro columnas de `#T-preflight-refs`, `rows_sha256`
`8e8ccea5a9916eeb31d63aa8f79c247a02d5e3cbbce23b21c75dacc18535c3fb` = el que firmó su gate.

**1. Pares contrafactuales (140 grupos).** Éxito conjunto **50/140 = 0,357**,
IC95 % [0,283 · 0,439], azar conjunto 0,083. Seguimiento del hecho decisivo: 74/140 grupos
mueven la elección (0,529). Al lado: nli-nograd 48/140 = 0,343 (empate, intervalos solapados),
pointer 2/140 y 1/140 (Laya gana), Qwen 123/140 = 0,879 (Laya pierde). Por idioma: en 30/70
= 0,429 [0,319 · 0,545]; es 20/70 = 0,286 [0,193 · 0,401].

**2. Batería completa (400 filas, forzada).** **238/400 = 0,595**, IC95 % [0,546 · 0,642],
azar 0,3375, macro por familia 0,595. Por familia: extracción 0,813, clasificación 0,788,
inferencia/negación 0,688 (las tres despejan su azar); comparación de atributos 0,325 y
prioridad 0,363 **no despejan el azar** — las dos mismas familias en las que nli-nograd
tampoco lo despeja (0,413 y 0,375). Por idioma: en 0,660 [0,592 · 0,722] (checkpoint
inglés), es 0,530 [0,461 · 0,598] (multilingüe). Contra nuestras columnas: empata con
nli-nograd (0,6225; conteo emparejado: 186 aciertan los dos, 52 sólo Laya, 63 sólo nli),
gana a los dos pointer (0,330 / 0,3275) y pierde contra Qwen (0,965). **Control de
permutación: FALLA** — con las opciones al revés cambian de id **128/400 elecciones
(0,32)** y el mayor desplazamiento de peso es 0,889 (tolerancia 1e-5). Laya lee la
posición: una parte de su 0,595 descansa en el orden. (Su propio BENCHMARKS.md declara
0,15–0,23 a 20 opciones; aquí, a K = 2–8, es peor.)

**3. BANKING77 a K = 77 (`eval.fullspace`, 3 080 filas del test oficial, checkpoint inglés,
presupuesto por defecto).** **0,379** (IC95 % [0,362 · 0,396], azar 0,013, n 3 080). Su 0,425 publicado **no reproduce** en nuestro arnés (queda fuera del intervalo); nuestro pointer head mide 0,0123 en el mismo corte. Artefacto `artifacts/gates/T-laya-baseline/fullspace.json`.

**Mezcla publicada, por dataset** (gate, `mix_declared_per_dataset`): batería-dev y BANKING77
fuera de la mezcla (BANKING77 sin marca en la tabla de temas de `BENCHMARKS.md` y
`in_training=False` en `research/scripts/bench_apps.py`); spam, phishing, RAG relevance y
support triage «in training» (`BENCHMARKS.md` §Themes); AG News y BoolQ «in training mix»
(`README.md` §English tasks). Sólo los dos primeros se miden aquí; los demás quedan
declarados para `#T-ingest-public`.

**Veredicto (una línea, del gate):** Laya sin ajustar: forzada 0,595 [0,546 · 0,642], contrafactual conjunto 0,357 [0,283 · 0,439]; **gana** a los dos pointer, **empata** con nli-nograd, **pierde** con Qwen, y **falla igual que nosotros** en comparación de atributos y prioridad; BANKING77 K=77 0,379, no reproduce su 0,425.

**Lo que enseña.** El checkpoint público de la misma clase de producto, sin ajustar, queda
donde nuestro cross-encoder sin ajustar (0,595 vs 0,6225) y falla en las MISMAS dos familias
(comparación de atributos y prioridad): el problema ahí es de planteamiento/datos, no de
arquitectura. Nos gana con holgura sólo donde el control pointer no aprende. Y su 0,595 lleva
un 32 % de dependencia del orden, que nuestro nli-nograd no tiene (delta 0,0).

**Seam en `eval/fullspace.py`:** `scored()` acepta un `engine` con `.entries()`; `run()` acepta
`engine`, `manifest`, `gate_path`, `parity=False`. La métrica (`tally`) no cambia; el pointer
sigue por su camino. `eval/splits.py`: el escáner de índice-módulo salta cualquier `.venv*`
(`.venv-laya` disparaba un rojo ajeno).

**Comandos exactos:**
```
python3.12 -m venv .venv-laya && .venv-laya/bin/pip install laya
PYTHONPATH=. .venv-laya/bin/python -m eval.laya_baseline battery --device cpu     # 218 s + 400 al revés
PYTHONPATH=. .venv-laya/bin/python -m eval.laya_baseline banking77 --device cpu --no-sweep
PYTHONPATH=. .venv-train/bin/python -m eval.laya_baseline gate
PYTHONPATH=. .venv-train/bin/python -m eval.gate_rules check artifacts/gates/T-laya-baseline   # C1-C7 pass
```

**Tests:** `eval/test_laya_baseline.py` — 27 tests con predictores falsos, sin descargar
pesos ni importar `laya`/torch (comprobado en intérprete limpio). `eval/test_laya_baseline.py` + `test_splits.py` + `test_fullspace.py`: 68 passed, 2 subtests passed in 4.53s. `ruff check`
limpio.

**NO hecho (dicho como tal):** el barrido de cardinalidad de `fullspace` (`--no-sweep`, 5 000
pasadas más en CPU); el corte dev de BANKING77 con `nli-nograd` a K=77 que `#T-laya-archdiff`
predijo (D1) — es la medida de `#T-laya-objective`; el checkpoint `laya-typed-decisions`.

— T-laya-baseline · Laya sin ajustar medida sobre nuestra batería, los contrafactuales y BANKING77 K=77
