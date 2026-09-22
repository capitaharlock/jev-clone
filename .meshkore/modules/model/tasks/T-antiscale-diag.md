---
id: T-antiscale-diag
title: Por qué la accuracy unseen cae al escalar — ablación, no opinión
status: active
priority: high
owner: unassigned
category: model
initiative: generalization-fix
depends_on: []
created: 2026-09-22
updated: 2026-09-22
---

# Por qué la accuracy unseen cae al escalar — ablación, no opinión

La curva de `artifacts/gates/T-mix-5m/gate.json` es anti-monótona en unseen
(0,239 → 0,050 de 250 k a 1 M en ettin-68m; 0,194 → 0,028 en modernbert-base)
mientras seen se mantiene ~0,58-0,61. Antes de tocar la arquitectura hay que
saber **qué** se degrada. Nadie propone un arreglo en esta task: se mide.

Ejes a aislar, cada uno con su experimento y su número:

1. **Memorización del espacio de etiquetas.** Correlacionar la caída unseen
   con `option_text_reuse` por dataset (ya está en el manifest). Predicción
   falsable: la caída se concentra en los cortes de vocabulario cerrado.
2. **Abstención desbocada.** `unseen_abstain_rate` pasa de 0,258 a 0,702 en el
   mismo tramo. Separar "falla" de "se calla": recalcular accuracy unseen
   forzando decisión (sin `unknown`) y ver si la caída persiste.
3. **Capacidad entrenable.** Backbone congelado (149 M) + cabeza 2,9 M. Repetir
   el tramo 250 k → 1 M con la cabeza al doble y al cuádruple de ancho. Si la
   pendiente no cambia, no es capacidad.
4. **Calibración vs. ranking.** `unseen_ranking` cae a 0,168 (azar 0,202): el
   orden de las opciones también se degrada, así que no es sólo temperatura.
5. **Régimen de entreno.** LR, schedule y 1 sola época sobre 1 M: comprobar si
   el tramo final está sobreentrenando el mapa de etiquetas (early-stopping
   sobre unseen como control).

## Verification gate

- Cada eje produce un número reproducible por seed + manifest, no una opinión.
- El informe nombra el eje dominante y **cuánto** de la caída explica (%).
- Test: re-ejecutar la ablación con la misma seed reproduce los números.
- El gate escribe `artifacts/gates/T-antiscale-diag/gate.json` con los 5 ejes,
  su medida y el eje señalado como causa dominante.

## Done when

- Los 5 ejes están medidos sobre los checkpoints ya existentes (sin reentrenar
  el corpus completo salvo el eje 3, que sí exige dos runs cortos).
- Hay una causa dominante nombrada con su cuota de la caída.
- `#T-unfreeze-backbone` y `#T-gen-objective` quedan priorizadas por ese
  hallazgo, no por la sospecha previa.
