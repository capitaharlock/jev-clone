---
id: T-ingest-laya
title: typed-decisions de Laya a episode-v1 — el corte donde Jev publica 0,727
status: done
priority: high
owner: developer
category: data
initiative: data-flywheel
depends_on:
  - T-episode-contract
created: 2026-09-27
updated: 2026-09-27
---

# typed-decisions de Laya a episode-v1 — el corte donde Jev publica 0,727

## Contexto (léelo, no lo asumas)

Laya (`TMP/laya/`, Apache-2.0) se ajusta con `LocalLLaMA/typed-decisions`
(Hugging Face, público). Estructura medida el 2026-09-27 con la API de HF:

- config `all`: **train 1 200 estados, test 400**; también por flujo:
  `agent_trace_observability`, `customer_service`, `invoice_processing`,
  `security_incidents` (300/100 cada uno). Sólo inglés.
- Columnas: `id`, `workflow`, `split`, `state` (JSON en texto), `questions`
  (JSON: `{nombre: {type, instructions, criteria}}`), `gold` (JSON por
  pregunta: `label`, `confidence`, `probabilities` / `probability_true` /
  `score`), `factors`, `label_agreement`.
- Tres tipos de pregunta: `choice` (criteria = dict etiqueta → descripción),
  `score` (criteria = lista ordinal), `noul` (binaria, criteria true/false).
- ~5 preguntas por estado → ≈ 6 000 decisiones de train y ≈ 2 000 de test.

**Por qué importa:** en el test de 400 estados Jev 1.13.0 publica **0,727** y
Laya ajustado **0,766** (`TMP/laya/BENCHMARKS.md`). Es el único corte público
donde podemos ponernos al lado de Jev fila a fila. El test es **eval-only para
siempre**; el train es volumen para el trainer.

## Qué hacer, paso a paso

1. **Registrar la fuente** en `.meshkore/docs/source-register.md`: HF id,
   revisión (commit sha del dataset), licencia declarada en su card (léela;
   si no la declara, `eval-only` hasta que el operador decida), uso: train
   split → train; test split → `eval-only`.
2. **Adaptador** `data/convert_typed_decisions.py` (sigue el patrón de
   `data/convert_banking77.py` para descarga fijada por revisión y sha, y el
   de `data/episode_contract.py` para el formato). Una decisión = un episodio:
   - `state`: el JSON de `state` renderizado a texto legible (clave: valor por
     línea, listas con «; »). No inventes hechos; no resumas.
   - pregunta: `questions[q].instructions` tal cual.
   - `candidates`: para `choice`, un candidato por clave de `criteria` con
     `id` opaco (`c1..cK`, orden barajado con semilla) y `text` = la
     descripción (no la etiqueta: Laya avisa de que las etiquetas tipo
     `true/false` se siguen por el nombre). Para `score`, un candidato por
     valor ordinal, `text` = «valor — descripción». Para `noul`, dos
     candidatos con las dos descripciones.
   - `answer`: el `id` opaco del gold. `evidence`: cadena vacía permitida sólo
     si el contrato lo admite; si no, la línea del estado que `factors`
     señale, literal (el validador `EC.evidence_is_fragment` lo exige).
   - `family`: `external/typed-decisions/<workflow>/<tipo>`; `lang: en`;
     `origin: laya-typed-decisions`; `variant_group`: `td-<id>` (las ~5
     preguntas del mismo estado comparten grupo: **es el eje «misma
     situación, distinta pregunta»**, exactamente lo que el modelo falla).
   - Conserva `gold.probabilities` en un campo `teacher_soft` (Laya lo usa
     como CE suave): no se usa hasta `#T-loop-rl-jev`, pero no se tira.
3. **Validar** todos los episodios con el validador de `episode-v1`; los que
   no pasen van a `rejects.jsonl` con motivo, nunca se «arreglan» a mano.
4. **Escribir** `artifacts/episodes-external/typed-decisions/{train,test}/`
   con `episodes.jsonl`, `rejects.jsonl` y `manifest.json` (revisión, sha,
   n por flujo × tipo, semilla del barajado, comando).
5. **Gate** en `artifacts/gates/T-ingest-laya/gate.json`: n por split, flujo y
   tipo; rechazos y motivos; comprobación de que ningún `id` de test está en
   train; `eval_only: ["test"]`. Sin cifras de accuracy: aquí no se mide nada.
6. Tests (`data/test_convert_typed_decisions.py`): una fila de cada tipo
   convierte a lo esperado; el barajado es reproducible por semilla; el gold
   apunta al `id` correcto tras barajar; test y train no comparten ids.

## Done when

- Los dos splits existen como episodios válidos con manifest y sha, y el gate
  publica los conteos medidos.
- `source-register.md` tiene la fila con licencia y uso; el test está
  `eval-only` y hay test que impide que un manifest de mezcla lo incluya.
- `eval/` puede leer el test como corte externo (una función en
  `eval/cuts.py` o equivalente que lo devuelve por sha) para que
  `#T-loop-scoreboard` lo compare con Jev 0,727 y Laya 0,766.

## Qué NO hacer

- No copiar código de Laya al repo. Lee `TMP/laya/laya/common.py::build_sequence`
  para entender su formato si te ayuda; escribe el nuestro.
- No entrenar, no evaluar: esta task es sólo datos. CPU.

## Resolution (2026-09-27, developer)

**Fuente fijada.** `LocalLLaMA/typed-decisions`, config `all`, commit
`f7a2487edd7a043a5441a5e9ccc7fe5ddbd9ebe8`, licencia Apache-2.0 declarada en la
card. Parquet re-hasheado al bajar contra el LFS del Hub (train
`46a58d63…`, test `4f294f21…`); una revisión o un sha distintos hacen que
`fetch` se niegue a decodificar. Fila en `source-register.md`.

**Comandos (los del manifest):**

```
PYTHONPATH=. <venv-con-pyarrow>/bin/python -m data.convert_typed_decisions fetch
PYTHONPATH=. .venv-train/bin/python -m data.convert_typed_decisions convert
PYTHONPATH=. .venv-train/bin/python -m data.convert_typed_decisions gate
```

`fetch` necesita `pyarrow` (parquet), que el `.venv-train` no lleva porque
está bloqueado por hash (`requirements-train.txt`); se corrió desde un venv
efímero con `pyarrow huggingface_hub` y sólo escribe el raw jsonl
(`artifacts/data-raw/typed-decisions/`, gitignorado, con su manifest y sha).
`convert` y `gate` son stdlib y corren en `.venv-train`.

**Conteos medidos (`artifacts/gates/T-ingest-laya/gate.json`, verdict PASS,
10/10 comprobaciones):**

| split | casos | decisiones | episodios | rechazos | choice | score | noul | sha256 de `episodes.jsonl` |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| train | 1 200 | 6 000 | 6 000 | 0 | 1 800 | 2 400 | 1 800 | `d6dcafc4…de8e902` |
| test | 400 | 2 000 | 2 000 | 0 | 600 | 800 | 600 | `a294de92…f26843bc` |

Por flujo (train; test es exactamente un tercio en cada celda):
agent_trace_observability choice 600 / score 600 / noul 300;
customer_service 600 / 600 / 300; invoice_processing 300 / 600 / 600;
security_incidents 300 / 600 / 600. Ningún `case_id`, ningún estado y ningún
id de episodio compartido entre train y test; los 2 000 de test llevan
`eval_only: true`; `teacher_soft` presente en los 8 000 y con exactamente los
ids de sus candidatos. Sin cifra de accuracy: aquí no se mide nada.

**Decisiones del adaptador que la task no podía prever (todas en código,
ninguna a mano, todas registradas en el manifest bajo `rules`):**

- `family` es una enumeración cerrada en `episode-v1`, así que
  `external/typed-decisions/<flujo>/<tipo>` habría rechazado los 8 000. Va en
  `subfamily`; `family` mapea `choice`/`score` → `description_classification`
  (cada opción lleva su definición) y `noul` → `textual_inference_negation`.
- Las instrucciones `noul` son afirmaciones («This trace requires human
  review.») y el contrato exige pregunta: se publican como `<afirmación> Is
  this statement true?` (`question_render` por episodio). `choice` y `score`
  van verbatim (las 2 400 + 3 200 terminan en `?`).
- 800 decisiones `noul` (invoice `duplicate`, security `credential_compromise`)
  vienen sin `criteria`: reciben las dos descripciones de
  `NOUL_DEFAULT_CRITERIA`.
- El dataset no anota evidencia y el contrato la exige literal: se toma la
  línea del estado con más solape con los `factors` del caso (desempate por
  la descripción gold), marcada `evidence_origin: heuristic:factor-overlap`.
  Es una evidencia débil (en invoices gana `payment.terms`, en agent traces
  `constraints`/`task`) y queda dicho: sirve al contrato, no a un
  verificador. Un caso cuyos factores no tocan ninguna línea sería rechazo;
  no hubo ninguno.

**Barrera del corte test (tres capas, con test cada una):**
`data.convert_typed_decisions.assert_trainable` rechaza rutas
`…/typed-decisions/test`, manifests con `eval_only: true` u
`origin`+`split`, y los propios episodios; `data.mix.NEVER_TRAINABLE` y
`data.firewall.BENCHMARKS` (`typed-decisions-test`, revisión fijada) lo
conocen, así que `check_job_allowed("typed-decisions-test")` lanza.
`eval.cuts.external_cut("typed-decisions", "test")` devuelve el corte por
sha (manifest re-hasheado; sha esperado opcional; `reserved: true`) para
`#T-loop-scoreboard`.

**Tests.** `data/`: 510 passed, 6 skipped (33 nuevos en
`data/test_convert_typed_decisions.py`; `test_firewall` cuenta `BENCHMARKS`
dinámicamente y sigue en verde con la card nueva). `eval/`: 478 passed, 4 failed — los cuatro
ajenos: `test_release_gate…published` (rojo preexistente), dos de
`test_gate_rules::TestPilotCensusOnDisk` por `artifacts/gates/T-ce-finetune/`
que otro agente está escribiendo ahora mismo, y
`test_splits::test_repo_has_no_index_modulo_split` por `RecursionError` en
`.venv-laya/…/sympy/…/resolvent_lookup.py` (un venv ajeno que el escáner no
salta). `ruff check` limpio en los seis ficheros tocados.

**Git.** `episodes.jsonl` (14 MB + 4,5 MB) no se versiona
(`artifacts/episodes-external/.gitignore`): manifest + rejects + gate sí, y
el comando y el sha del manifest lo reconstruyen y verifican.
