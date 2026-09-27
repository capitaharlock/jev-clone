---
id: T-ingest-laya
title: typed-decisions de Laya a episode-v1 — el corte donde Jev publica 0,727
status: next
priority: high
owner: unassigned
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
