---
id: T-labelspace-div
title: Diversidad de espacios de etiquetas — medirla y, si falta, generarla
status: next
priority: high
owner: unassigned
category: data
initiative: generalization-fix
depends_on:
  - T-antiscale-diag
created: 2026-09-22
updated: 2026-09-22
---

# Diversidad de espacios de etiquetas — medirla y, si falta, generarla

La hipótesis de arquitectura (`#T-antiscale-diag`, `#T-unfreeze-backbone`) tiene
una gemela por el lado de los datos: `decision-mix-clean-1m` está construido
sobre un número **pequeño** de taxonomías (dbpedia14 = 14 etiquetas,
civil-comments = 2, MASSIVE = 60, ...). Añadir filas dentro de esas taxonomías
no enseña a comparar pregunta↔opción: refuerza el mapa cerrado `texto → etiqueta`.
Es la explicación de la anti-monotonía que **no** requiere tocar el modelo, y hay
que descartarla o confirmarla con la misma disciplina.

Primero se mide sobre el corpus que ya existe: cuántos espacios de etiquetas
distintos hay, cuántas filas por espacio, y cuánto se solapan con los cortes
unseen de `#T-unseen-labels`. Si el inventario sale pobre, se **genera**: la
fábrica sintética (`#T-synth-factory`, `#T-gen-schemas`) ya sabe producir
esquemas de decisión completos — aquí se la apunta a producir *muchos espacios
pequeños* en vez de muchas filas de pocos espacios, y se re-mezcla a igual
volumen total para que la comparación contra la curva actual sea limpia.

No se entrena a ciegas: la mezcla nueva se evalúa con el gate de
`#T-unseen-labels` a los mismos tamaños (62 k / 250 k / 1 M) que la curva vieja,
para que el veredicto sea una curva contra otra curva.

## Done when

- Existe un inventario versionado (`artifacts/gates/T-labelspace-div/`) con nº de
  espacios de etiquetas, filas por espacio y solape con los cortes unseen.
- Hay una mezcla alternativa a **igual volumen** con ≥ 10× más espacios de
  etiquetas distintos que `decision-mix-clean-1m`.
- La curva unseen de esa mezcla está medida a 62 k / 250 k / 1 M con el gate de
  `#T-unseen-labels`, junto a la curva vieja en el mismo JSON.
- El veredicto queda escrito: la diversidad de etiquetas explica la
  anti-monotonía (GO al camino de datos) o no la explica (NO-GO, y el peso cae
  en `#T-unfreeze-backbone`).
