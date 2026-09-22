---
id: T-gen-objective
title: Objetivo que premia comparar, no recordar
status: blocked
priority: high
owner: developer
category: model
initiative: generalization-fix
depends_on:
  - T-antiscale-diag
created: 2026-09-22
updated: 2026-09-22
---

# Objetivo que premia comparar, no recordar

Si la ablación de `#T-antiscale-diag` confirma memorización del espacio de
etiquetas, descongelar el backbone sólo le da más capacidad para memorizar
mejor. Esta task ataca el incentivo, no la capacidad.

Candidatos a evaluar, cada uno como un run comparable sobre el mismo 1 M:

1. **Dropout de etiquetas por episodio**: en cada batch, sustituir una fracción
   de las opciones por texto no visto, de forma que el mapa memorizado nunca
   sea suficiente para acertar.
2. **Entreno episódico few-shot**: muestrear conjuntos de opciones distintos
   para la misma pregunta, con el gold siempre presente pero los distractores
   remuestreados — ya existe el sampler de `#T-optset-sampler`.
3. **Pérdida contrastiva sobre el texto de la opción**, en vez de (o junto a)
   la cross-entropy listwise, para que la señal sea pregunta↔opción y no
   pregunta→índice del espacio conocido.
4. **Penalización de frecuencia de etiqueta**: descontar del logit el prior
   empírico de esa cadena en train, para que una etiqueta frecuente no gane
   por serlo.

No se adoptan los cuatro: se mide cuál mueve la pendiente unseen y se queda
el que la mueva, con el resto registrado como descartado y por qué.

## Verification gate

- Cada candidato corre con la misma seed, mix y eval que el baseline.
- Test: con opciones nuevas no vistas en train, el modelo adoptado supera el
  azar (0,165) en todos los cortes de `#T-unseen-labels`.
- Test: `unseen_ranking` supera su propio azar (0,202) — no basta con acertar,
  el orden tiene que ser informativo.
- El gate escribe `artifacts/gates/T-gen-objective/gate.json` con el candidato
  adoptado, los descartados y el motivo numérico de cada descarte.

## Done when

- Un candidato adoptado con ganancia unseen medida y reproducible.
- Los descartados quedan registrados con su número, no borrados.
- `#T-release-gate` vuelve a evaluarse sobre el checkpoint resultante.

## Prioridad (#T-antiscale-diag)

**Primera de las dos.** La ablación de `#T-antiscale-diag`
(`artifacts/gates/T-antiscale-diag/gate.json`) atribuye el **75,4 %** de la
caída 250 k → 1 M al eje 4, *ranking*, no a la temperatura: el número forzando
decisión cae con el crudo y a 1 M no se distingue del azar. El eje 1 mide,
además, que el **100 %** de la mezcla 1 M se contesta con un mapa
texto→etiqueta, así que cada fila nueva refuerza exactamente el incentivo que
esta task ataca. El orden no depende de cuál de los dos ejes de la partición
gane: bajo el protocolo completo de `#T-unseen-labels` la caída de
modernbert-base es casi toda abstención, y con el backbone congelado el logit
de `unknown` es igualmente algo que la cabeza aprendió bajo este objetivo. Se
invierte con `#T-unfreeze-backbone` si el eje 3 (cabeza ×2/×4, job
`antiscale-wide`) aplana la pendiente.

## Estado 2026-09-22 — código listo, medición en cola

El código de los cuatro candidatos está en `main` (`e14a411`): cuatro flags
componibles en `training/python/train_decision.py` con 24 tests verdes. Lo que
falta es exclusivamente la MEDICIÓN, y depende de un recurso ocupado, no de
una decisión: MPS está tomado por el job `antiscale-wide` (eje 3 de
`#T-antiscale-diag`, dos runs de 1 M, ~6,2 h cada uno al ritmo medido de
62 k filas / 1 390 s).

El job `gen-objective-sweep` ya está vivo y espera a que aparezca
`artifacts/runs/antiscale-wide-d1024-modernbert-s20260922/summary.json`;
entonces corre los cinco runs cortos de 62,5 k (base + los cuatro candidatos,
misma seed 20260922, mismo mix `--fence-clean`). Horizonte ≈ 14 h desde las
10:34Z.

Al cerrar el sweep queda por hacer, en este orden:

1. `artifacts/gates/T-gen-objective/gate.json` con el adoptado y el número de
   descarte de cada uno de los otros tres.
2. Run largo a 1 M del candidato adoptado, para que la ganancia se mida en el
   punto donde la pendiente es negativa hoy.
3. Re-evaluar `#T-release-gate` sobre el checkpoint resultante.

Esta task no se desbloquea a mano: se desbloquea cuando el sweep escribe sus
cinco `summary.json`.
