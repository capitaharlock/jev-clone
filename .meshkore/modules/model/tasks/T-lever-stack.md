---
id: T-lever-stack
title: Apilar las dos palancas que sí aplanan la pendiente — cabeza ×2 + objetivo prior
status: active
priority: high
owner: unassigned
category: model
initiative: generalization-fix
depends_on:
  - T-antiscale-diag
  - T-gen-objective
  - T-unfreeze-backbone
created: 2026-09-23
updated: 2026-09-23
---

# Apilar las dos palancas que sí aplanan la pendiente

Tres hipótesis medidas contra la curva anti-monótona 250 k → 1 M, y solo
dos sobrevivieron:

| palanca | pendiente unseen 250 k → 1 M | fuente |
|---|---|---|
| nada (congelado, d256) | −0,1899 | `#T-mix-5m` |
| cabeza ×2 (d512) | −0,0549 | `#T-antiscale-diag`, brazo `antiscale-wide-d512` |
| objetivo prior-penalty | −0,0868 | `#T-gen-objective`, brazo `genobj-prior-1m` |
| backbone `last-n=2` @1e-5 | −0,0927 (y peor en el gate) | `#T-unfreeze-backbone` — NO-GO |

Las dos que funcionan se midieron **por separado** y nunca se han
apilado. Son mecanismos distintos —una da capacidad a la cabeza, el otro
quita la recompensa de memorizar la etiqueta frecuente— así que a priori
no compiten. Esta task mide la combinación con un único brazo controlado.

Brazo: `ettin-68m`, `--d-model 512`, `--prior-penalty 1.0`,
`--fence-clean`, 1 M filas, batch 64, seed y mix-seed 20260922, MPS — todo
idéntico al brazo prior de `#T-gen-objective` salvo `d_model`. Ese brazo
(unseen 0,2347 a 1 M, 0,2879 en el gate) es el control.

Coste: ~2 h de GPU por la ruta batched de `#T-metal-throughput`, no las
~11 h de antes. Eso permite, si el apilado da señal, un segundo brazo d1024
sin renegociar la GPU.

## Done when

- El run de 1 M termina con las 5 etapas (62 k…1 M) y su `summary.json` en
  `artifacts/runs/`, y `eval.unseen gate` corre sobre el checkpoint de 1 M.
- `artifacts/gates/T-lever-stack/gate.json` publica la pendiente
  250 k → 1 M del apilado junto a las tres ya medidas, y dice si el efecto
  es aditivo, parcial o nulo — con el número, no con adjetivos.
- La decisión queda escrita: si la pendiente apilada no mejora la mejor de
  las dos sueltas (−0,0549), el eje de capacidad/objetivo se declara
  agotado y el relevo pasa al espacio de etiquetas (`#T-labelspace-div`).
