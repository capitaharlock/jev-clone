---
id: T-listwise-format
title: Formato listwise — todas las opciones en una secuencia para poder compararlas entre sí
status: done
priority: medium
owner: unassigned
category: model
initiative: daily-learning-loop
depends_on:
  - T-ce-finetune
created: 2026-09-27
updated: 2026-09-30
---

# Formato listwise — todas las opciones en una secuencia para poder compararlas entre sí

**Por qué.** Peldaño 4 de la escalera de brazos. El cross-encoder puntúa cada
opción por separado, y «¿es ésta la más barata?» exige ver las demás. Laya
pone un marcador por opción dentro de la secuencia y lee su estado oculto
(D1 de `docs/laya-archdiff.md`). Laya sin ajustar también falla atributos y
prioridad (0,325 y 0,363), pero después de ajustarse llega a 0,766 en
typed-decisions. Idea adoptable bajo Apache-2.0 citando el commit `4066d5d`;
**no se copia código**.

**Qué hacer.**
1. Un scorer nuevo en `model/`: `[CLS] pregunta [SEP] [OPT] texto1 [OPT] texto2 … [SEP] estado`,
   logit = cabeza escalar sobre el estado oculto de cada `[OPT]`. El orden de
   las opciones se baraja en entreno y se promedia sobre 2 permutaciones en
   evaluación. El control de permutación de preflight tiene que pasar; Laya lo
   falla con un flip de 0,32.
2. El mismo trainer (`--scorer listwise`), la misma pérdida CE listwise y las
   mismas evaluaciones y gate.
3. Smoke de 5 000 contra el `current` del bucle, con predicción escrita antes.

**Done when.** Scorer con tests (invariancia a permutación dentro de la
tolerancia, sin recortar opciones hasta K=8), smoke medido y el peldaño
disponible en `training.python.loop`.

## Resolution

El scorer listwise, su trainer y siete tests de contrato quedaron implementados.
El mejor punto de investigación fue 0,7500 en `loop-0030`. La base activa
conservada es `loop-0024`, con 0,7325.

La auditoría binaria sobre las mismas 400 filas no justificó cambiar la
arquitectura: listwise obtuvo 0,7325, el torneo secuencial 0,7125 y Copeland
0,7075. En K=8 las cifras fueron 0,7000, 0,6500 y 0,6333. Se mantiene una
inferencia listwise de una pasada. El ranking pareado queda para la pérdida
auxiliar de `#T-aligned-ranking-v2`.
