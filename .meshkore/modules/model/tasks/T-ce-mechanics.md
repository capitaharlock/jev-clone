---
id: T-ce-mechanics
title: Validar la mecánica antes de gastar presupuesto — sobreajustar 32–64 casos
status: done
priority: high
owner: unassigned
category: model
initiative: cross-encoder-pilot
depends_on:
  - T-ce-scorer
created: 2026-09-25
updated: 2026-09-26
resolved_by: A051
resolved_by_conv: work-cross-encoder-pilot-T-ce-mechanics-commit-1790480000
completed_at: 2026-09-26T17:53:03.770Z
commit_shas: ['a8712f7b4fcf5796e6505ecf81eebdef0a5e2d5a']
---
# Validar la mecánica antes de gastar presupuesto — sobreajustar 32–64 casos

Paso 3 del piloto, y el más barato de todos. Antes de generar 5.000 decisiones y
antes de ocupar la GPU durante horas, comprobar que la tubería **puede** aprender:
ajustar entre 32 y 64 ejemplos inequívocos y exigir >95 % sobre ellos mismos.

Esto **no valida generalización** y no se puede citar como si lo hiciera. Valida
que los gradientes llegan, que el gold está donde el trainer cree, que el
tokenizado no corta el estado, que la máscara de candidatos es correcta y que el
formato de hipótesis es el mismo en entreno y en evaluación. Es exactamente la
clase de fallo que la fase 1 tardó semanas en descartar.

Si falla: se arregla la tubería, no se cambia de arquitectura ni se añaden datos.

## Verification gate

- Test: >95 % de acierto sobre los mismos 32–64 ejemplos ajustados, con la semilla
  y el comando registrados.
- Test: el mismo checkpoint sin ajustar sobre esos ejemplos queda claramente por
  debajo — si ya los acierta, el conjunto no es válido como prueba de mecánica.
- El gate deja constancia explícita de que la cifra **no** es una medida de
  generalización.

## Done when

- El sobreajuste supera el 95 % y el artefacto lo registra con su comando
  reproducible.
- Cualquier fallo encontrado en el camino (gold mal alineado, truncado del
  estado, máscara, discrepancia de formato entreno/eval) queda corregido y
  cubierto por un test.
- El formato de hipótesis queda congelado para el resto del piloto.

## Formato de hipótesis CONGELADO

Punto 3 del `Done when`, y vale para el resto del piloto. Vive en UN solo
sitio importable, `model/ce_scorer.py`, y nadie más en el árbol tiene una
plantilla (`test_no_second_hypothesis_template_lives_in_the_tree` recorre
todos los `.py` y exige que el único que la contenga sea ese).

    premisa   = "State: {state}\nQuestion: {question}"        # en
                "Estado: {state}\nPregunta: {question}"        # es
    hipótesis = "The answer to this question is: {option}."     # en
                "La respuesta a esta pregunta es: {option}."    # es

Y con la premisa va, sólo en las familias cuyo contrato lo pide
(`attribute_comparison`, `priority_decision`), el bloque comparativo:
`"Options under comparison:" / "Opciones en comparación:"` seguido de los
K textos ordenados alfabéticamente y sin ids.

Lo demás que el formato fija, porque también decide qué texto ve el modelo:

- reducción a escalar: `entail_logit` (el logit crudo de `entailment`, leído
  del `config.json` del checkpoint, nunca asumido en el índice 0);
- recorte: `truncation="only_first"` con ventana 512 — se recorta la
  PREMISA, nunca la opción que se juzga, y un par que no cabe PARA el run
  en vez de recortar el estado en silencio;
- un solo camino de render y de tokenizado: `ce_scorer.flatten_pairs` y
  `ce_scorer.encode_pairs`. El entreno (`training/python/ce_overfit.py`) y
  la evaluación (`eval/ce_nograd.py`) llaman a esas dos y a ninguna otra.

**Huella: `b215e3003cc60c0f`** (`ce_scorer.format_fingerprint()`, que tiene
que ser igual a `HYPOTHESIS_FORMAT_ID` o el test falla). Cambiar cualquiera
de las piezas de arriba mueve la huella: entonces hay que actualizar la
constante y republicar toda cifra medida con la anterior. Las cifras de
`artifacts/gates/T-ce-scorer/` y `artifacts/gates/T-ce-mechanics/` llevan
la huella dentro para que se sepa cuál es cuál.

## Resolution

✓ task T-ce-mechanics done. files: 4. commit: a8712f7. tests: `model` 104 verdes (25,7 s), `training/python` 175 verdes + 1 rojo preexistente y ajeno (332 s).

Registro cerrado sin medir nada: la cifra del artefacto (46/48 = 0,958 vs 12/48 = 0,250, distancia 0,708) se leyó de `overfit.json` (`status: measured`, `gate.pass: true`) y se trasladó a la task, a coverage y al test.

<details><summary>.meshkore/modules/model/tasks/T-ce-mechanics.md — `## Resolution` reescrita + frontmatter</summary>

- `status: blocked` → `done`, `resolved_by: A043` → `A051`, `updated: 2026-09-26`, quitado `failed_at`, conv actualizada.
- `## Resolution` ya no dice "cifra NO medida / awaiting-operator-compute": cuenta los tres umbrales superados, las 60 épocas en 117,33 s, semilla 20260926, huella `aadbb6a235c2361e`, checkpoint `minilmv2-l6-mnli-xnli`, y el comando exacto del campo `command` (`--device mps`).
- Frase explícita conservada y reforzada: entrena y mide sobre los MISMOS 48 casos, **no mide generalización**, no se compara con el 70 % ni con nada externo; la primera cifra con significado externo es `#T-ce-finetune` contra `#T-battery-dev`.
- Los cuatro chequeos de tubería documentados con sus números del run real (deltas 3e-08, 0 truncados de 152 pares, 24 pares byte a byte, relleno a `0.0`).
</details>

<details><summary>training/python/test_ce_overfit.py — 1 test reescrito</summary>

`test_the_gate_artifact_carries_no_figure_nobody_measured` → `test_the_gate_artifact_on_disk_is_measured_and_passes_its_thresholds`.

Antes tenía una rama `if status == "awaiting-operator-compute": assert gate["pass"] is None`. Ahora exige, sin rama:

- `status == "measured"`, `gate["pass"] is True`, huella del conjunto recomputada;
- los tres umbrales contra las constantes del módulo: `target == OVERFIT_TARGET` y `value > 0.95`; `max_allowed == SET_VALIDITY_MAX` y `untrained_accuracy <= 0.60`; `min == MIN_MARGIN` y `value >= 0.35`;
- `n == 48` en los dos lados, y que la distancia publicada es la resta de las dos accuracies;
- los cuatro `pipeline_checks` en `pass: True` y `failed == []`.

Un artefacto que volviese a `pass: null` pone este test en rojo.
</details>

<details><summary>.meshkore/docs/coverage.md — sección de parcial → cerrada</summary>

- Título `## Parcial … (código verde, cifra SIN medir)` → `## Cerrado 2026-09-26 — #T-ce-mechanics (gate FIRMADO, 0,958 de sobreajuste)`.
- La última fila de la tabla pasa de `**SIN MEDIR**` a `**medido**` con las tres cifras.
- El párrafo "**El gate NO está firmado**" sustituido por el que da las cifras, los umbrales, los cuatro checks en PASS y el comando reproducible; mantiene la nota de que no mide generalización.
</details>

<details><summary>Verificación — conteos reales</summary>

```
PYTHONPATH=. .venv-train/bin/python -m pytest model -q
  → 104 passed, 8 subtests passed in 25.75s

PYTHONPATH=. .venv-train/bin/python -m pytest training/python -q
  → 1 failed, 175 passed, 4 subtests passed in 332.12s
     FAILED training/python/test_cloud_api.py::TestCloudApi::test_docker_parity_image_coldstart
```

Ese rojo es el preexistente y ajeno que avisaste: falla en `docker build -t jevclone:gate .` (exit 1), no toca nada de este gate. El otro que mencionaste (`eval/test_release_gate.py::…t_unseen_labels_published`) vive en `eval/`, que no entra en los dos directorios pedidos — no se ejecutó ni se tocó.

Corrido por directorio, con `.venv-train/bin/python`. Diario en `.meshkore/log/2026-09-26.md` (gitignored, no va al commit).
</details>

— T-ce-mechanics · el gate de mecánica del cross-encoder queda firmado con la cifra medida (0,958 de sobreajuste) y el test la exige

**Commit** `a8712f7b4` · 4 files · 1.8M tokens

**Files changed (4):**
- `.meshkore/docs/coverage.md`
- `.meshkore/modules/model/tasks/T-ce-mechanics.md`
- `artifacts/gates/T-ce-mechanics/overfit.json`
- `training/python/test_ce_overfit.py`
