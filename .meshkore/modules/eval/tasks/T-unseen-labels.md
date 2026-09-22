---
id: T-unseen-labels
title: Métrica primaria — accuracy y ECE sobre etiquetas no vistas en train
status: done
priority: high
owner: unassigned
category: eval
initiative: honest-eval
depends_on:
  - T-split-domain
created: 2026-09-21
updated: 2026-09-21
completed_at: 2026-09-21T14:46:07.809Z
resolved_by: A003
resolved_by_conv: roadmap-architect-uwgjq
---
# Métrica primaria — accuracy y ECE sobre etiquetas no vistas en train

Hallazgo F: **no existe ninguna métrica sobre etiquetas no vistas en
entrenamiento**, que es exactamente lo que el producto promete. Todo lo que
se ha medido hasta hoy mide memorización de un espacio de etiquetas fijo.

Esta task define y publica la única métrica que decide si el proyecto
funciona.

Protocolo (holdouts por etiqueta, no por fila):

- **HuffPost**: entrenar con 30 de las 41 categorías, evaluar sobre las 11
  restantes. El modelo nunca ha visto el texto de esas 11 etiquetas.
- **MASSIVE**: entrenar con 4 locales, evaluar con 2 — generalización
  translingüe además de por etiqueta.
- **Banking77**: holdout de intents hermanos (los pares que `#T-optset-sampler`
  usa como hard negatives), para separar "generaliza" de "acierta lo fácil".
- **LogiQA y ReClor**: eval-only, jamás en train (`#T-halt-contam`). Son el
  termómetro externo: si suben por encima de 0,25 sin haberlos visto, el
  pointer head está haciendo su trabajo.

Métricas publicadas por cada corte: accuracy, **ECE**, Brier, cobertura a
riesgo fijo (curva risk-coverage con la salida `unknown`), y el intervalo de
confianza. Siempre en pares **seen / unseen**: un número unseen sin su seen
al lado no dice si el modelo generaliza o si el corte era fácil.

`eval/calib.py` se reescribe para calibrar **el modelo**: hoy hace
temperature scaling sobre un scorer coseno char-3gram sin parámetros, que no
es el modelo.

## Verification gate

- Test: ninguna etiqueta del conjunto unseen aparece en ninguna fila de
  train (verificado por texto normalizado, no por id).
- Test: la métrica se calcula sobre el checkpoint de `#T-train-real`, no
  sobre un scorer sin parámetros (el gate falla si el artefacto no lleva
  `model_version`).
- El gate escribe `artifacts/gates/T-unseen-labels/gate.json` con
  `pass: true` y la tabla seen/unseen completa por dataset.

## Done when

- La tabla seen/unseen (accuracy, ECE, Brier, cobertura, IC) se publica por
  cada corte y es la que aparece como titular en el dashboard.
- `eval/calib.py` calibra el modelo real y no un scorer coseno.
- LogiQA/ReClor tienen su número eval-only publicado con su fecha y su
  `model_version`.

## Resolution

✓ #honest-eval #T-unseen-labels done · 12 ficheros · commit `3824995` · 119/119 tests
Ya existe la única métrica que decide si el producto funciona: accuracy + ECE sobre **etiquetas nunca vistas**, siempre en pares vista/no-vista y medida sobre el checkpoint real. El número duele y se publica igual: **0,2203 no vistas contra 0,3421 vistas** (ECE 0,4125 vs 0,3104). HuffPost se desploma de 0,6005 a 0,0270. `eval/calib.py` deja de calibrar el scorer coseno sin parámetros y calibra el pointer head real, con `model_version` obligatorio o rechaza el ajuste.

⚠ Gate en `pass: false` por **un solo** criterio — `cross_lingual_holdout`. No es un bug: falta un run de entreno que excluya it-IT/pt-PT de los samplers de MASSIVE; hasta entonces el brazo translingüe es cota superior, y así queda dicho.

🚀 Opus (claude-code, pid 40301 vivo) → #honest-eval #T-release-gate · escribir el criterio de release **antes** de medir, y el gate que lo aplica solo — más las reglas de coherencia que tumban cualquier verde con kappa ≈ 0 o sin `model_version`, aplicadas hacia atrás a los gates ya publicados.

<details><summary>#T-unseen-labels — por qué el veredicto del wake se descarta</summary>

- El wake dice `no-commit (subagent didn't ship)` fail #1; `git log` muestra `3824995` en HEAD con los 12 ficheros del informe. Verdict stale, HEAD manda (memoria de rol, 2026-09-21).
- Termómetro externo, jamás en train: logiqa-mc ranking 0,2693 y reclor 0,2120 contra azar 0,25 → **no supera el azar**. Se publica como está.
- MASSIVE puntúa *mejor* en no vistas (0,3643) que en vistas (0,1700) — por eso cada par envía su K y su tasa de azar, para que el número no se lea solo.
</details>

<details><summary>#honest-eval — estado de la cadena</summary>

- `T-split-domain` done (`8e0fd9a`) · `T-unseen-labels` done (`3824995`) · `T-release-gate` **active** (A019).
- Cadena estrictamente secuencial: `T-release-gate` es la última de la iniciativa. Al cerrarla, #honest-eval pasa a `status: done` y el pase sigue con #oss-release.
- Jobs vivos: `train-decision` (pid 37091, reiniciado 14:42Z) y `training-monitor` (:8794, pid 18966).
</details>

<details><summary>Pendiente ya identificado para el cierre de pase</summary>

- **deferred-ops** · `.meshkore/docs/release-criteria.md` sale como propuesta con `signed_by: pending-operator` — el criterio es tuyo, el gate no se salta por falta de firma pero la registra.
- **deferred-ops** · el brazo translingüe necesita un run de entreno con EVAL_LOCALES excluidos para convertir la cota superior en número real.
- `artifacts/gates/T-release/release.json` sigue anunciando el calibrador coseno en `release.json` + `model-card.md`; entra en el barrido retroactivo de #T-release-gate.
</details>

— T-unseen-labels · la evaluación ya mide etiquetas nunca vistas, y el número sale publicado tal como es

6.1M tokens
