---
id: T-counterfactuals
title: Contrafactuales — cambiar el hecho decisivo cambia la respuesta
status: next
priority: high
owner: unassigned
category: data
initiative: episodic-data
depends_on:
  - T-episode-gen
created: 2026-09-25
updated: 2026-09-25
---

# Contrafactuales — cambiar el hecho decisivo cambia la respuesta

Esta es la clase de ejemplo que el corpus actual casi no contiene y que explica
el fallo medido: los dos checkpoints probados eligen «green» tanto si el estado
dice que el color favorito es verde como si dice **rojo**, y eligen el mismo
producto al preguntar cuál cuesta menos y cuál dura más (la probabilidad de A
pasa de 69,709 % a 69,747 % — no sigue el cambio de respuesta correcta).

Para cada episodio de `#T-episode-gen`, generar variantes que alteren **una sola
cosa** y queden agrupadas por `variant_group`:

- **Cambia el hecho decisivo** del estado → cambia el gold.
- **Cambia la pregunta** sobre el mismo estado y las mismas opciones → cambia el
  gold. Este es el eje que más falta.
- **Cambia una descripción de opción** → cambia el gold.
- **Paráfrasis equivalente** (del estado, de la pregunta o de una opción) → el
  gold **no** cambia. Sin este control, «sensibilidad» se confunde con
  inestabilidad.

Balancear lo que el modelo puede aprender como atajo: nombres de entidad,
posición del ganador, vocabulario y qué etiqueta gana. Incluir distractores
plausibles. Mezclar etiquetas de otros espacios como negativos **no vale por sí
solo**: filtrar duplicados literales no prueba exclusión semántica, y un negativo
que también responde bien a la pregunta envenena el gold.

El ejemplo «le gusta la fruta verde → su color favorito es verde» es una
inferencia plausible, no una consecuencia necesaria: se puede entrenar como
«elegir la alternativa más compatible», nunca etiquetar como certeza factual. Los
casos básicos usan hechos inequívocos («su color favorito es verde»).

## Verification gate

- Test: en cada `variant_group`, la variante de hecho y la variante de pregunta
  tienen gold distinto al original; la paráfrasis tiene el mismo.
- Test: la posición del candidato correcto está uniformemente distribuida por
  familia (χ² sobre el reparto, umbral escrito antes de medir).
- Test: ningún negativo del conjunto de candidatos satisface la regla de
  respuesta del episodio.

## Done when

- Todo episodio publicado pertenece a un `variant_group` con al menos un
  contrafactual y una paráfrasis de control.
- Los cuatro tipos de variante existen en las cinco familias, en ES y EN.
- El reparto de posición, nombres y etiqueta ganadora está medido y publicado, no
  asumido.
