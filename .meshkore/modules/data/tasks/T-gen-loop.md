---
id: T-gen-loop
title: Continuous decision-data generator loop
status: blocked
owner: rjj-mac-claude
category: data
initiative: data-training
depends_on:
  - T-halt-contam
  - T-gen-schemas
created: 2026-09-20
updated: 2026-09-21
created_by: live-anchor-loop
created_by_conv: general-09192230
---
# Continuous decision-data generator loop

**Bloqueada por la auditoría `.meshkore/docs/audit-2026-09-21.md`
(hallazgos C y D).** El loop tal y como está produce 258 700 filas desde 190
esqueletos y 5 plantillas frescas de Q-W-E-N en 11 h: volumen sin
información. Se para en `#T-halt-contam` y se rehace como generador de
esquemas de decisión completos en `#T-gen-schemas`.

No se cierra como `done` ni se borra: queda bloqueada como el registro de por
qué el generador cambió de contrato.
