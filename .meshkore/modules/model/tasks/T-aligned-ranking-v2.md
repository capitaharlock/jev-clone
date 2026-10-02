---
id: T-aligned-ranking-v2
title: Alinear el corpus numérico y añadir ranking pareado al scorer listwise
status: active
priority: high
owner: unassigned
category: model
initiative: daily-learning-loop
depends_on:
  - T-listwise-format
  - T-loop-trainer
  - T-dev-rotation
created: 2026-09-30
updated: 2026-10-02
---

# Alinear el corpus numérico y añadir ranking pareado al scorer listwise

**Activado el 2026-10-02.** Los ciclos 35 a 60 de la receta anterior dieron
una media de 0,7320, mínimo 0,7075 y máximo aislado 0,7725. El holdout llegó a
0,87 sin transferencia estable a dev. La mezcla v3 sustituye esa receta:
12.000 filas numéricas en grupos completos, sólo `superlative` para
comparación y `chain2` para prioridad, balance 50/50 por familia e idioma y
K=2/3/8 en proporción 40/30/30 dentro de cada celda. Añade 3.000 filas frescas
de las tres familias fuertes. Cada candidato ve una pasada completa de las
12.000 filas de train antes del gate pareado.

El ciclo 62 es el control: mezcla v3 con CE listwise puro. Desde el ciclo 63
el bucle entrena con `--pairwise-weight 0.3`, el brazo central del paso 4: una
pérdida logística del mejor positivo contra cada negativo, promediada dentro
de cada decisión para que K=8 no pese más que K=2. El peso queda en el
manifiesto de cada checkpoint y `state.json` puede cambiarlo por brazo. El
bucle varía semilla y ronda de mezcla por ciclo, así que la comparación con el
control es por el gate pareado contra el activo, no una ablación de semilla fija.

## Por qué

El scorer listwise llega a 0,8086 en el holdout de una mezcla fresca, pero se
queda entre 0,72 y 0,75 en la batería objetivo. El corpus por regla es exacto y
no tiene fuga directa, pero genera demasiadas formas ordinales, umbrales,
empates y cadenas de tres criterios. Además, 83 de las 390 comparaciones del
piloto tienen unidades rotas. El cuello es la alineación semántica.

La auditoría binaria tampoco justifica sustituir la inferencia listwise. Su
señal se usará como pérdida auxiliar para separar el gold de negativos
difíciles dentro de la misma decisión.

La evaluación actual ordena el texto de las opciones antes de una sola pasada.
Por eso el control publicado con delta cero no prueba invariancia: distintas
permutaciones terminan en la misma secuencia canónica. El siguiente pase debe
medir orden real sin canonicalizar.

## Trabajo

1. Retirar las 83 filas rotas del piloto y terminar su muestra humana.
2. Generar un corpus numérico que iguale familia, idioma, K, dominio, forma de
   pregunta, unidades y longitud de la batería sin copiar estados ni
   candidatos de desarrollo. Separar train y holdout por paquete de
   plantillas, catálogo de entidades y semilla.
3. Parafrasear con Qwen local y aceptar una fila sólo cuando la regla original
   siga recomputando el mismo gold.
4. Entrenar con `CE_listwise + lambda * ranking_pairwise`, normalizado una vez
   por decisión. Medir `lambda` 0,1, 0,3 y 1,0 con el mismo inicio, semillas y
   presupuesto.
5. Para comparación y prioridad, probar un reranker simétrico sobre el top 3
   o top 4 listwise. Agregar con Bradley-Terry o Copeland y mantener una sola
   pasada en las tres familias fuertes.
6. Puntuar las 400 filas bajo orden canónico, inverso y cuatro órdenes con
   semilla, sin canonicalización. Publicar cambios de ganador, delta máximo de
   probabilidad, ciclos de preferencia, ECE, NLL y latencia por familia y K.
7. Comparar contra `loop-0024` con bootstrap pareado en el dev rotado. La
   batería actual queda como regresión y no elige el checkpoint.

## Done when

- El corpus publica conteos y hashes por familia, idioma, K, dominio, forma de
  pregunta y unidad; ninguna fila de dev aparece en train.
- Cero unidades inválidas en la muestra humana y 5.000 de 5.000 gold
  recomputados sin discrepancia.
- La ablación mide listwise puro frente a la pérdida híbrida con el mismo
  checkpoint inicial y presupuesto.
- El candidato supera 0,80 en el dev rotado, la media de comparación y
  prioridad alcanza 0,745 y la consistencia contrafactual supera 0,65.
- Ninguna de las dos familias hoy en 0,925 cae más de 0,02; los cambios por
  orden o cuadro no superan 0,01 y ECE no supera el 0,134 actual.
- El sellado sigue cerrado hasta el hito previsto por `#T-ce-confirm`.
