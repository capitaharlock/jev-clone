---
id: T-gen-objective
title: Objetivo que premia comparar, no recordar
status: next
priority: high
owner: unassigned
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
