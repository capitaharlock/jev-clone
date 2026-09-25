---
id: T-release-gate
title: Criterio de release escrito antes de medir, y gate que lo aplica
status: blocked
blocked_reason: espera la firma del operador en .meshkore/docs/release-criteria.md v2 (criterio sobre la batería privada)
priority: high
owner: unassigned
category: eval
initiative: honest-eval
depends_on:
  - T-battery-sealed
  - T-ce-confirm
created: 2026-09-21
updated: 2026-09-25
---

# Criterio de release escrito antes de medir, y gate que lo aplica

La mitad mecánica de esta task está hecha y se conserva: el gate lee el criterio
de `.meshkore/docs/release-criteria.md` y emite GO/NO-GO sin juicio humano en el
medio, y las **reglas de coherencia** que hacen fallar cualquier gate del repo
—`pass: true` con `cohen_kappa <= 0.1`, una métrica sin `model_version`, un split
no sellado— siguen vigentes y se amplían en `#T-battery-metrics` con la regla de
lectura contradictoria.

**Lo que cambia el 2026-09-25 es el criterio, porque el sujeto de la medida
cambia.** La v1 pedía accuracy unseen mínima por corte y kappa contra el profesor
externo. La v1 no representaba el producto: `unseen_accuracy_all_min = 0,50` se
medía sobre cortes de etiquetas de datasets ajenos, y `teacher_cohen_kappa_min`
depende de una credencial que devuelve 401 y que el plan de recuperación prohíbe
tratar como bloqueante.

`release-criteria.md` **v2**, escrito y firmado antes de la medición que lo
evalúa, con umbrales sobre la **batería privada** de `#T-battery-sealed`:

- **Meta principal: ≥70 % macro entre familias** en las preguntas respondibles.
  Para afirmar «al menos 70 %» con respaldo estadístico, exigir además **límite
  inferior del IC95 % ≥70 %**.
- Mínimo por **familia** y por **idioma** — una media que pasa destruyendo una
  familia o el español no es un GO.
- **Éxito conjunto en pares contrafactuales** mínimo: es la propiedad cuyo fallo
  define el estado actual.
- **Invariancia a permutaciones** dentro de tolerancia numérica.
- **ECE máxima** y **cobertura mínima a riesgo fijo**, con ranking y abstención
  juzgados por separado.
- **Latencia p95 y memoria en hardware real** — y aquí el criterio reconoce que el
  cross-encoder relee el estado por opción: si la referencia de calidad no cumple
  latencia, el GO de calidad y el GO de producto son dos verdictos distintos, y el
  segundo depende de `#shared-state-distill`.
- Cada umbral con **una línea de justificación**: por qué ese número y no otro.

Lo que queda **fuera** del criterio v2: BANKING77 a 77 vías, que sigue publicándose
como diagnóstico de transferencia difícil pero no como condición de release.

El criterio es del operador: esta task lo redacta como propuesta y no se cierra
hasta que lo firma.

## Verification gate

- Test: un artefacto sintético con `pass: true` y `cohen_kappa: 0.0`, o con una
  lectura que contradiga su propio IC, hace fallar el runner de gates.
- Test: el gate recalcula GO/NO-GO desde `release-criteria.md` y no desde
  constantes en el código; cambiar un umbral en el doc cambia el veredicto.
- Test: un resultado que pasa de media pero incumple un mínimo por familia o
  idioma da NO-GO.
- El gate escribe `artifacts/gates/T-release-gate/gate.json` con el veredicto, el
  sha del criterio aplicado y el sha del test sellado consumido.

## Done when

- `release-criteria.md` v2 existe, está firmado por el operador y su sha queda
  registrado en cada veredicto; la v1 queda archivada, no borrada.
- El veredicto separa GO de calidad y GO de producto cuando la latencia lo exige.
- Ningún gate del repo puede publicar verde con kappa ≈ 0, sin `model_version`, o
  con una lectura que sus números contradigan.
- Los gates históricos que no cumplen están marcados, no borrados.
