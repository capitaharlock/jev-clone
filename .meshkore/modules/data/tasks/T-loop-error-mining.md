---
id: T-loop-error-mining
title: Minar los errores de dev — la cuota de mañana la deciden los fallos de hoy
status: next
priority: medium
owner: unassigned
category: data
initiative: daily-learning-loop
depends_on:
  - T-loop-nightly
created: 2026-09-27
updated: 2026-09-27
---

> **2026-09-27:** peldaño 1 de la escalera de brazos (`docs/bucle-infinito.md` §4). Sube de backlog a next.

# Minar los errores de dev — la cuota de mañana la deciden los fallos de hoy

Plan de recuperación §7 paso 6: «ejemplos dirigidos a errores». Cuando el
bucle lleve una semana, los fallos en dev (por familia × idioma × K × tipo de
variante) mueven la cuota de `#T-episode-scale` para la noche siguiente:
más de lo que falla, sin dejar ninguna familia bajo un mínimo declarado.

- Regla escrita antes: cuota_f = base_f × (1 + α · error_f / error_medio),
  α y mínimos en el manifest; nunca más del 40 % para una sola familia.
- **Dev no se usa para generar**: se usan sus *categorías* de error, no sus
  filas. Test: ningún estado ni candidato de dev aparece en lo generado.
- Gate: dos ciclos reales donde la cuota cambió por errores, con el efecto
  en dev de la familia empujada (cifra, IC), positivo o no.
