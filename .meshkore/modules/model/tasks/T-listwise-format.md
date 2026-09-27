---
id: T-listwise-format
title: Formato listwise — todas las opciones en una secuencia para poder compararlas entre sí
status: next
priority: medium
owner: unassigned
category: model
initiative: daily-learning-loop
depends_on:
  - T-ce-finetune
created: 2026-09-27
updated: 2026-09-27
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
