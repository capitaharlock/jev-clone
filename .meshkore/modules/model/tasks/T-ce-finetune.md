---
id: T-ce-finetune
title: Aprender de verdad — ajustar con 5.000 decisiones verificadas, ampliar a 20.000 si mejora
status: next
priority: high
owner: unassigned
category: model
initiative: cross-encoder-pilot
depends_on:
  - T-ce-mechanics
  - T-preflight-refs
  - T-episode-splits
created: 2026-09-25
updated: 2026-09-25
---

# Aprender de verdad — ajustar con 5.000 decisiones verificadas, ampliar a 20.000 si mejora

Paso 4 del piloto. Sustituye a `#T-encoder-finetune`, cuyos cuatro brazos de
descongelado dependían de un «objetivo ganador» que no existe.

Ajustar el scorer de `#T-ce-scorer` con las decisiones verificadas de
`#episodic-data`: **5.000 primero**, ampliar a 20.000 sólo si desarrollo mejora.
Presupuestos de evaluación intermedia fijados antes de arrancar, para poder parar
por evidencia y no por cansancio.

Dos decisiones de método:

- **Ajustar el encoder** (completo o con adaptación eficiente), y comparar contra
  el mismo checkpoint **sin ajustar**. Entrenar sólo otra cabeza aleatoria sobre
  representaciones congeladas repite la limitación que queremos someter a prueba.
  El NO-GO de `#T-unfreeze-backbone` se midió con K≤8 bajo la pérdida de fase 1 y
  **no** equivale a probar esta receta.
- **Los checkpoints pointer actuales son el control**, no el rival a batir a toda
  costa: se mantienen medidos en la misma batería y con el mismo protocolo.

Criterio de continuación: mejorar respecto al mismo modelo sin ajustar en
desarrollo **sin destruir los pares contrafactuales ni un idioma**. Si empeora, se
revisan datos y objetivo y se vuelve al checkpoint base — no se escala.

## Verification gate

- Test: la mezcla consumida por el entreno es auditable contra el manifest de
  `#T-episode-splits` (familias, idiomas, grupos de variantes).
- Test: la evaluación intermedia corre en los presupuestos predeclarados y
  escribe accuracy forzada, con abstención, macro por familia y éxito conjunto en
  pares contrafactuales — las cuatro, no sólo la media.
- El gate compara siempre contra el mismo modelo sin ajustar con protocolo de
  información equivalente.

## Done when

- El ajuste a 5.000 está medido en desarrollo contra el checkpoint sin ajustar,
  con IC95 %.
- La ampliación a 20.000 se hace **sólo** si la primera mejoró, y su decisión
  queda escrita con la cifra que la justifica.
- Ninguna familia ni idioma cae por debajo de su nivel sin ajustar sin que quede
  registrado como limitación.
- Si el resultado es NO-GO, el gate lo dice con su causa candidata y el piloto no
  escala.
