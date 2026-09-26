---
id: T-battery-dev
title: Batería de desarrollo — 400 casos representativos, cinco familias, ES y EN
status: done
priority: high
owner: developer
category: eval
initiative: honest-eval
depends_on:
  - T-episode-contract
created: 2026-09-25
updated: 2026-09-26
---

# Batería de desarrollo — 400 casos representativos, cinco familias, ES y EN

Paso 1 del piloto. Hoy el proyecto no tiene ningún corte que represente las
decisiones del operador: BANKING77 mide 77 intenciones de banca, no preguntas
binarias sobre un estado ni comparaciones de producto. Sin este corte, «70 %» no
tiene sujeto.

400 casos de **desarrollo** —el corte contra el que se elige todo, y que por eso
se puede mirar tantas veces como haga falta— conformes a `#T-episode-contract`:

- Las **cinco familias** de decisión, con reparto declarado antes de construirlo.
- **ES y EN**, ambos con n suficiente para leer cada familia por separado.
- **K = 2 / 3 / 8**, el régimen del producto.
- Pares contrafactuales agrupados: original + variante decisiva + paráfrasis de
  control.

La **mezcla y los pesos se definen antes de evaluar nada**, y quedan versionados.
Un corte cuya composición se ajusta después de ver un resultado ya no mide.

Protocolo de información equivalente: cada modelo comparado recibe un **único
formato**, elegido sin consultar el test final. Si un modelo necesita hipótesis
explícitas y otro no, la diferencia de formato se documenta y se justifica; no se
elige el formato que más favorece a uno de ellos.

## Verification gate

- Test: los 400 casos pasan el validador de `#T-episode-contract`.
- Test: no hay `variant_group` compartido con la batería sellada de
  `#T-battery-sealed`.
- Test: el reparto real por familia × idioma × K coincide con la mezcla declarada
  (tolerancia escrita antes).
- Los desacuerdos de revisión están resueltos o el caso está fuera; no quedan
  casos ambiguos sin decidir.

## Done when

- Los 400 casos existen, versionados con su sha, revisados y con gold
  justificable caso a caso.
- La mezcla y los pesos están escritos y fechados **antes** de la primera
  medición.
- El formato de evaluación de cada modelo candidato está fijado y documentado.
