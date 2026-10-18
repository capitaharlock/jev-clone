---
id: T-battery-metrics
title: Las métricas mínimas, publicadas juntas — ranking y abstención por separado
status: done
priority: high
owner: developer
category: eval
initiative: honest-eval
created: 2026-09-25
updated: 2026-09-26
completed_at: 2026-09-26T13:58:06.881Z
resolved_by: A050
resolved_by_conv: work-honest-eval-T-battery-metrics-census-1790470000
commit_shas: ['f9f874c007df0a89b63e60fab256aac96b56db72']
---
# Las métricas mínimas, publicadas juntas — ranking y abstención por separado

Hoy se puede publicar un número que parece bueno porque el corte que lo produce
está elegido después. Esta task fija la suite que acompaña a **toda** medición
del piloto, y se construye antes de que exista la batería porque también se
aplica retroactivamente a los artefactos ya en disco.

Métricas obligatorias en cada veredicto:

- Accuracy **forzada** y accuracy **incluyendo abstención**, nunca una sola.
- Cobertura y precisión **entre las respondidas**.
- **Macro por familia** (la media global puede pasar con una familia a cero).
- **Azar por K**, calculado y escrito al lado de cada cifra.
- **Éxito conjunto en pares contrafactuales**: acertar el original y su variante;
  acertar uno de los dos no cuenta.
- **Invariancia a permutaciones**: el control de `#T-option-text` se incorpora
  aquí, y se distingue explícitamente del seguimiento del estado y la pregunta —
  son dos propiedades distintas y la prueba del 2026-09-24 las separa: las
  puntuaciones siguen al texto (<5 × 10⁻⁷ al permutar) y aun así el modelo falla
  el cambio decisivo.
- **NLL, Brier y calibración**, con temperatura/umbral calibrados en desarrollo y
  **verificados** en test.

Regla que hace honesta la lectura: **ranking y abstención se evalúan aparte**. Un
87,7 % de abstención con 0/1000 no es prudencia, es un ranking al azar tapado por
un umbral. Y los pesos softmax son relativos a los candidatos ofrecidos: no se
publican como probabilidad absoluta de verdad.

Incluye la corrección de narrativa que dejó pendiente `#T-fullspace-objective`: la
lectura automática de `verdict.reading` dice que el intervalo primario contiene el
azar cuando sus números lo excluyen por debajo (el que lo contiene es el de
elección forzada), y el resumen de `#T-bigk-optsets` afirma haber descartado la
cardinalidad como causa cuando un negativo con una semilla y un presupuesto sólo
descarta el beneficio observado de ese brazo. Se corrigen los textos y se añade la
regla de coherencia que impide volver a publicarlos: **un gate cuya lectura
contradiga sus propios números falla, no avisa.**

## Verification gate

- Test: un veredicto sintético al que le falte cualquiera de las métricas
  obligatorias, o el azar de su K, o el n de su corte, hace fallar el runner.
- Test: un artefacto con `reading` que afirme «contiene el azar» mientras su IC lo
  excluye hace fallar el runner.
- Test: el éxito conjunto en contrafactuales se calcula por `variant_group` y es
  0 cuando sólo se acierta una mitad.
- Los gates históricos que incumplen quedan **marcados, no borrados**.

## Done when

- La suite existe como una única función de reporte que todo gate del piloto
  invoca; no hay métricas calculadas a mano por run.
- Las tres lecturas erróneas identificadas están corregidas en disco y la regla
  que las prohíbe tiene test.
- Cada cifra publicada lleva su n, su K, su azar y su IC95 %.

## Resolution

✓ Censo del piloto en verde: `eval/` queda con el único rojo preexistente y `data/` verde. Commit `f9f874c`, sin push.

- #T-battery-metrics: la aserción hardcodeada `["T-battery-dev"]` + `T-battery-sealed` en `absent` ahora deriva el set de `PILOT_TASKS` × disco; conserva lista exacta, `publishes_a_figure: false` en todos y `offenders == []`.
- Añadido test a nivel censo que prueba que un gate del piloto con cifra a mano sigue fallando C7.
- `eval/`: 398 passed, 1 failed (el preexistente `test_the_numbers_are_the_ones_t_unseen_labels_published`, 0.292141 != 0.304943, intacto). `data/`: 439 passed, 6 skipped.
- ruff limpio, staged solo `eval/test_gate_rules.py`, snapshot §20 `20260926-135510434` previo a la edición. No se tocó entreno ni `test_release_gate.py`.

<details><summary>eval/test_gate_rules.py — qué cambió</summary>

- `test_the_only_pilot_gate_on_disk_publishes_no_figure_yet` → `test_the_pilot_gates_on_disk_publish_no_figure_yet` + helper `expected_on_disk()` (tareas de `G.PILOT_TASKS`, presencia por `is_dir` + `*.json`; hoy resuelve `["T-battery-dev", "T-battery-sealed"]`).
- Nuevo `test_a_pilot_gate_publishing_a_hand_computed_figure_fails_census`: árbol sintético con `measured.accuracy` a mano → `pass False`, `offenders == ["T-battery-dev"]`, fuera de `publishing_no_figure`.
- La detección de figura a mano ya existía a nivel `check_gate_dir` (`test_a_pilot_gate_computing_its_own_accuracy_fails_the_runner`); el nuevo la cubre a nivel `pilot_census`, que es lo que la lista derivada podría enmascarar.
</details>

<details><summary>Verificación verbatim</summary>

- `eval/test_gate_rules.py`: 52 passed, 3 subtests passed.
- `eval/`: `1 failed, 398 passed, 77 subtests passed` — el único fallo es `eval/test_release_gate.py::TestPublishedVerdict::test_the_numbers_are_the_ones_t_unseen_labels_published` (`AssertionError: 0.292141 != 0.304943`), preexistente de otro dueño, no tocado.
- `data/`: `439 passed, 6 skipped`.
- `ruff check eval/test_gate_rules.py`: All checks passed!
</details>

Nota: los integrity hints (8 refs rotas + 1 status) quedan sin tocar por orden expresa de "NADA más".

**Commit** `f9f874c00` · 1 file

**Files changed (1):**
- `eval/test_gate_rules.py`
