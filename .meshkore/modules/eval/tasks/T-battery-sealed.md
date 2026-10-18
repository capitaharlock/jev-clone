---
id: T-battery-sealed
title: Test sellado — 600 casos finales y el diagnóstico separado a K=20/77
status: done
priority: high
owner: unassigned
category: eval
initiative: honest-eval
depends_on:
  - T-battery-dev
created: 2026-09-25
updated: 2026-09-26
completed_at: 2026-09-26T13:53:20.240Z
resolved_by: A003
resolved_by_conv: roadmap-architect-uwgjq
---
# Test sellado — 600 casos finales y el diagnóstico separado a K=20/77

La otra mitad del paso 1, y la que da valor a cualquier afirmación pública. 600
casos **sellados** con las mismas cinco familias, ES y EN, y K=2/3/8; más un
**conjunto diagnóstico separado** a K=20/77 que no entra en la meta del 70 % pero
sí informa de cómo se degrada el modelo al crecer el denominador.

Reglas del sellado:

- Se abre **una sola vez**, por `#T-ce-confirm`. Su sha se marca como consumido y
  el runner rechaza una segunda apertura.
- Después de usarse pasa a **evidencia histórica** y esta iniciativa prepara otro
  para la siguiente decisión final. No se reutiliza para elegir el próximo brazo.
- Ningún `variant_group` se comparte con desarrollo.
- Es un corte **privado**: no lo ha producido el profesor ni se le ha mostrado
  durante la generación. No se declara ausencia de contaminación del
  preentrenamiento, que no es verificable.

BANKING77 a 77 vías se conserva —como transferencia difícil y como diagnóstico—
pero deja de ser la definición de éxito. Y `ModernBERT-base-zeroshot-v2.0` no
puede compararse ahí como «transferencia limpia»: su mezcla publicada incluye
BANKING77.

## Verification gate

- Test: el runner rechaza una segunda apertura del test sellado.
- Test: intersección de `variant_group` con desarrollo = 0; intersección de
  entidades y espacios declarada y por debajo del umbral escrito.
- Test: el diagnóstico K=20/77 se reporta en su propio bloque y no se promedia con
  la meta del 70 %.

## Done when

- Los 600 casos existen sellados, con sha registrado y estado «no consumido».
- El conjunto diagnóstico K=20/77 existe y está separado del corte de meta.
- La regla de apertura única está implementada, con test, no sólo escrita.
- El protocolo para preparar el **siguiente** test sellado está documentado antes
  de abrir este.

## Resolution

✓ #honest-eval #T-battery-sealed **done** · commit `c3e7a3f` en HEAD · 8 ficheros · gate **GO**, 14/14 checks medidos · 19/19 tests nuevos verdes (el `no-commit` del wake es falso otra vez; A049 sí commiteó).

**Ya existe el examen que nadie puede estudiar antes.** 600 casos sellados con protocolo de apertura única escrito, más 9 de diagnóstico a K=20/77 que quedan **fuera** de la meta del 70 % — para que el «se degrada al crecer el denominador» sea un dato y no una excusa. Los cortes no se tocan: 0 grupos compartidos con la batería de desarrollo, solape de entidades 16,3 % (umbral 35 %), solape de candidatos **0**.

Su único rojo real es honesto y es de otro: la aserción del censo en `eval/test_gate_rules.py` tenía la lista de gates del piloto **escrita a mano** (`["T-battery-dev"]`), así que el gate sellado la rompe por existir. El gate sellado en sí es C7-limpio (no publica cifra, 0 infractores).

🚀 #honest-eval #T-battery-metrics → `developer` (Muse, pid **80366**, vivo verificado con `ps`)
· Arreglar el censo para que la lista salga de las tasks del piloto y no de un literal, **conservando** que un gate que calcule su cifra a mano siga disparando C7. Sin GPU, sin entreno.
⏳ #T-preflight-refs sigue vivo (pid 77073, 11 min) — segundo carril de la misma iniciativa.

⚠ La cola del daemon apunta a `multi-cli-clients` y `deploy-pipeline-reconcile`, y **ninguna de las dos existe en disco** (0 initiatives, 0 tasks) → 0 despachables desde la cola. Sigo por #honest-eval, que es lo que tiene trabajo real; el orden lo decides tú, pero eso hay que crearlo o sacarlo de la cola.

<details><summary>#T-battery-sealed — qué se midió y qué quedó rojo</summary>

Gate `artifacts/gates/T-battery-sealed/gate.json` → `verdict: GO`, `consumed: false`, batería hasheada (`3168000…c9f7`) contra `episode-v1`:

- `n` 600/600 válidos, 0 inválidos, 0 ids duplicados
- `gold_rule_violations: 0` · `counterfactual_shape_violations: 0` · `distribution_violations: 0` · `namespace_violations: 0`
- `dev_shared_groups: 0` — disjunta de la batería de desarrollo por grupo, no por fila
- `entity_overlap: 0.1631` (umbral 0.35) sobre 1 429 entidades, 233 compartidas · `candidate_overlap: 0.0`
- diagnóstico aparte: `diag_valid` / `diag_gold` / `diag_ks` / `diag_namespace` en verde

Tests: `data/test_battery_sealed.py` → **19 passed**.

Rojos del árbol tras el aterrizaje:
- `eval/test_gate_rules.py::TestPilotCensusOnDisk::test_the_only_pilot_gate_on_disk_publishes_no_figure_yet` — nuevo, causado por esta entrega; lista exacta hardcodeada. **Despachado.**
- `eval/test_release_gate.py::TestPublishedVerdict::test_the_numbers_are_the_ones_t_unseen_labels_published` — preexistente (`0.292141 != 0.304943`), de otro dueño, intacto por orden.
- Los «env reds» que reportó A049 (`data/test_taxonomy.py`, `test_mix*.py`) no son reds: es que corrió `python -m pytest` y en este equipo pytest sólo está en `.venv-train/bin/pytest`. Con ese intérprete `data/` pasa.
</details>

<details><summary>Estado de #honest-eval — 6/9 hechas, 2 vivas, 2 esperando cómputo</summary>

- `done`: T-split-domain, T-unseen-labels, T-eval-cardinality, T-battery-dev, T-battery-metrics, **T-battery-sealed**
- vivas: T-preflight-refs (pid 77073), T-battery-metrics/censo (pid 80366)
- esperando: T-battery-calib (necesita `T-ce-finetune` — entreno, prohibido ahora) · T-release-gate (`blocked`, necesita `T-ce-confirm`)

La iniciativa **no** se puede cerrar: las dos últimas dependen de entrenar el cross-encoder, que es exactamente lo que tienes embargado.
</details>


— T-battery-sealed · el examen sellado de 600 casos + diagnóstico K=20/77, gate GO

18.3M tokens
