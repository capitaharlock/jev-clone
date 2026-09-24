---
id: T-antiscale-diag
title: Por qué la accuracy unseen cae al escalar — ablación, no opinión
status: done
priority: high
owner: unassigned
category: model
initiative: generalization-fix
depends_on: []
created: 2026-09-22
updated: 2026-09-23
resolved_by: mb-developer
resolved_by_conv: mb-developer
commit_shas: ['0f160cac75210eedd12b202c9dac1822eb01d6a6', 'f66a6b4f313384b705d94afdcf8499a9e2cbc095']
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

## Hallazgo (medido, no argumentado)

`artifacts/gates/T-antiscale-diag/gate.json` · informe
`artifacts/gates/T-antiscale-diag/REPORT.md` · ablación
`python3 -m tools.diagnose.antiscale --seed 20260922`.

**Eje dominante: el 4 — calibración vs ranking — con el 75,4 % de la caída**
sobre la curva que la task cita (ettin-68m 88,1 %, modernbert-base 62,7 %).
Los ejes 2 y 4 parten la caída exactamente, porque son las mismas
probabilidades leídas con y sin `unknown` en la carrera; los ejes 1, 3 y 5
explican esa mitad y no reciben cuota propia.

**Robustez, publicada con el hallazgo:** repetida la misma descomposición
sobre el protocolo completo de `#T-unseen-labels` (mismos dos checkpoints,
n=5624, con el corte banking77 que el stage eval del trainer no lleva), el eje
4 sigue siendo el dominante en **3 de 4** celdas (protocolo × brazo), con una
cuota agrupada del **56,4 %** y un rango 4,7–88,1 %. La celda que se sale es
modernbert-base bajo `#T-unseen-labels`, donde la caída es casi toda
abstención (95,3 %). Las dos lecturas se publican; la prioridad no cambia con
ninguna de ellas, porque con el backbone congelado el logit de `unknown`
también es algo que la cabeza aprendió bajo este objetivo.

| eje | medida | número |
|---|---|---|
| 1 · memorización del espacio de etiquetas | `option_text_reuse` de los 12 datasets | 12/12 por encima del umbral 0,5 (rango 0,986–0,998): el **100 %** de la mezcla 1 M se contesta con un mapa texto→etiqueta. La correlación que pedía la task queda `measured: false`: el eje no varía |
| 2 · abstención desbocada | accuracy unseen forzando decisión (`accuracy_options_only`) | **24,6 %** de la caída. Forzar la decisión NO recupera la curva |
| 3 · capacidad entrenable | cabeza ×2 y ×4 (`--d-model 512/1024`) | **`measured: false`** — los dos runs cortos siguen entrenando en el job `antiscale-wide`; el gate se rellena solo al aterrizar |
| 4 · calibración vs ranking | `unseen_ranking` contra su propio azar (0,202) | **75,4 %** de la caída (56,4 % agrupando los dos protocolos). A 1 M ya no se distingue del azar en ninguno de los dos brazos, y en modernbert-base queda estrictamente por debajo (CI95 alto 0,182) |
| 5 · régimen de entreno | épocas, schedule y early-stopping sobre unseen | 1,0 épocas sobre el corpus: ninguna fila se ve dos veces, así que no es re-ajuste por repetición. El early-stopping es control de SELECCIÓN, no causa (recupera 41,2 %–100 % según la paciencia) |

Prioridad que sale del hallazgo: **`#T-gen-objective` primero,
`#T-unfreeze-backbone` después**. Se invierte si el eje 3 aplana la pendiente
250 k → 1 M con la cabeza ancha.

Queda abierto sólo el eje 3, a la espera del job `antiscale-wide`.

## Resolution



**Commit** `0f160cac7` (+1) · 15 files

**Files changed (15):**
- `.meshkore/docs/coverage.md`
- `.meshkore/modules/data/tasks/T-labelspace-div.md`
- `.meshkore/modules/model/tasks/T-antiscale-diag.md`
- `.meshkore/modules/model/tasks/T-gen-objective.md`
- `.meshkore/modules/model/tasks/T-unfreeze-backbone.md`
- `.meshkore/roadmap/initiatives/generalization-fix.md`
- `.meshkore/roadmap/initiatives/oss-release.md`
- `artifacts/gates/T-antiscale-diag/REPORT.md`
- `artifacts/gates/T-antiscale-diag/gate.json`
- `artifacts/gates/T-antiscale-diag/inputs/unseen-250k-ettin-68m-s20260922.json`
- `artifacts/gates/T-antiscale-diag/inputs/unseen-250k-modernbert-base-s20260922.json`
- `tools/diagnose/__init__.py`
- `tools/diagnose/antiscale.py`
- `tools/diagnose/test_antiscale.py`
- `training/python/train_decision.py`

## Eje 3 — regla pre-registrada (escrita antes de leer el d512)

Decision del operador 2026-09-22: **el d1024 se corta en las dos ramas**, asi
que el eje 3 se cierra con la medida a x2 y con el motivo del corte escrito.
Lo unico que varia es el veredicto, y esta fijado de antemano:

- Baseline de comparacion: `mix1m-curve-modernbert-base-s20260922`, tramo
  250 k -> 1 M, `fall = 0.166556` / `fall_ranking = 0.104388`.
- Brazo x2: `artifacts/runs/antiscale-wide-d512-modernbert-s20260922`.
- Si el d512 **aplana** la pendiente -> causa = capacidad. `measured: true`,
  el eje 3 pasa a ser el eje dominante y la prioridad se mueve a
  `#T-unfreeze-backbone`.
- Si el d512 **iguala** al baseline dentro del ruido -> el eje 3 se registra
  como *medido a x2, no a x4*, con el numero del d512 y el motivo del corte.
  **Prohibido extrapolar el x4**: el gate no puede publicar un numero que
  nadie ha medido.
- En ambas ramas la GPU liberada va al run de 1 M del brazo `prior`
  (job `prior-1m`), no a la curva de label-space.

## Done when (adenda eje 3)

- `gate.json` trae el eje 3 con `measured` explicito y, si aplica, el campo
  que dice que el x4 no se midio y por que.
- `#T-antiscale-diag` no se cierra con ningun numero de x4 inventado.


## Eje 3 — medido (2026-09-23), y cambia la prioridad

El run `antiscale-wide-d512-modernbert-s20260922` (cabeza ×2, d_model 512,
mismo corpus/seed/evals que la curva congelada) cerró. La pendiente 250 k → 1 M
**sí** depende de la capacidad entrenable:

| brazo | 250 k | 1 M | Δ |
|---|---|---|---|
| modernbert congelado d256 (baseline) | 0,1941 | 0,0276 | −0,1665 |
| modernbert congelado **d512** (eje 3) | 0,2387 | 0,1838 | **−0,0549** |
| ettin-68m congelado d256 (baseline) | 0,2394 | 0,0495 | −0,1899 |
| ettin-68m congelado + prior-penalty (#T-gen-objective) | 0,3215 | 0,2347 | −0,0868 |

Doblar la cabeza recorta el **67 %** de la caída; el objetivo prior recorta el
**54 %** y además sube todo el nivel. Ninguno la elimina: las dos palancas son
parciales y **acumulables**, y el eje 3 deja de ser "no es capacidad".

El brazo ×4 (d1024) queda cortado por decisión del operador: el eje se cierra
medido a ×2, nunca extrapolado a ×4.

**Consecuencia registrada:** se cumple la condición escrita en
`#T-unfreeze-backbone` ("vuelve a ser la primera si la cabeza ×2 o ×4 aplana la
pendiente"). Esa task pasa a `active`.
