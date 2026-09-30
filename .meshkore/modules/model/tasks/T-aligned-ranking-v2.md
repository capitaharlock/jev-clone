---
id: T-aligned-ranking-v2
title: Alinear el corpus numérico y añadir ranking pareado al scorer listwise
status: next
priority: high
owner: unassigned
category: model
initiative: daily-learning-loop
depends_on:
  - T-listwise-format
  - T-loop-trainer
  - T-dev-rotation
created: 2026-09-30
updated: 2026-09-30
---

# Alinear el corpus numérico y añadir ranking pareado al scorer listwise

## Por qué

El scorer listwise llega a 0,8086 en el holdout de una mezcla fresca, pero se
queda entre 0,72 y 0,75 en la batería objetivo. El corpus por regla es exacto y
no tiene fuga directa, pero genera demasiadas formas ordinales, umbrales,
empates y cadenas de tres criterios. Además, 83 de las 390 comparaciones del
piloto tienen unidades rotas. El cuello es la alineación semántica.

La auditoría binaria tampoco justifica sustituir la inferencia listwise. Su
señal se usará como pérdida auxiliar para separar el gold de negativos
difíciles dentro de la misma decisión.

## Trabajo

1. Retirar las 83 filas rotas del piloto y terminar su muestra humana.
2. Generar un corpus numérico que iguale familia, idioma, K, dominio, forma de
   pregunta, unidades y longitud de la batería sin copiar estados ni
   candidatos de desarrollo.
3. Parafrasear con Qwen local y aceptar una fila sólo cuando la regla original
   siga recomputando el mismo gold.
4. Entrenar con `CE_listwise + lambda * ranking_pairwise`. Declarar `lambda`,
   semillas, mezcla y presupuestos antes de medir.
5. Comparar contra `loop-0024` con bootstrap pareado en el dev rotado. La
   batería actual queda como regresión y no elige el checkpoint.

## Done when

- El corpus publica conteos y hashes por familia, idioma, K, dominio, forma de
  pregunta y unidad; ninguna fila de dev aparece en train.
- Cero unidades inválidas en la muestra humana y 5.000 de 5.000 gold
  recomputados sin discrepancia.
- La ablación mide listwise puro frente a la pérdida híbrida con el mismo
  checkpoint inicial y presupuesto.
- El candidato supera 0,80 en el dev rotado y pasa el gate pareado sin degradar
  ninguna familia ni idioma.
- El sellado sigue cerrado hasta el hito previsto por `#T-ce-confirm`.
