---
id: T-split-domain
title: Split por plantilla y dominio — nunca por índice de fila
status: active
priority: high
owner: unassigned
category: eval
initiative: honest-eval
depends_on: []
created: 2026-09-21
updated: 2026-09-21
---

# Split por plantilla y dominio — nunca por índice de fila

Hallazgo C: el split `i % 10` pone la **misma plantilla** en train y en test.
Con 258 700 filas salidas de 190 esqueletos, eso da acc 1,0000 y logloss
0,00074 — y el "forward test sobre 20 000 ejemplos no vistos" evalúa ids
nuevos, no contenido nuevo. La tendencia "1.000 → 1.000 → 1.000" del
dashboard no mide nada.

`data/firewall.py` (hash normalizado) y `data/leakage.py` (Jaccard de
3-gramas) son un detector de contaminación en dos capas bien diseñado. Se
reutilizan tal cual: lo que falta es aplicarlos al split, no reescribirlos.

Trabajo, en `eval/splits.py`:

1. **Clave de agrupación** por fila: esqueleto normalizado (cifras, nombres
   propios y montos sustituidos por placeholders) + dominio + idioma. El
   split se hace por grupo, nunca por fila: un grupo entero cae en train o
   en test, jamás en ambos.
2. Splits sellados y versionados: `artifacts/splits/<name>/{train,test}.ids`
   con su manifest, seed y sha256. Un split no se regenera silenciosamente.
3. **Auditoría retroactiva** de todos los datasets del corpus: para cada uno,
   nº de esqueletos distintos, tamaño del grupo más frecuente, y solape
   Jaccard train↔test. El informe va a `artifacts/gates/T-split-domain/`.
4. Los números publicados hasta hoy que se apoyaban en `i % 10` se marcan
   como inválidos en el dashboard y en los artefactos, con la fecha.

## Verification gate

- Test: sobre el fichero en cuarentena `synth-loop-20260921.jsonl`, el nuevo
  split produce **0 % de solape de esqueletos** entre train y test, y la
  accuracy del baseline cae de 1,0000 a un número realista. Ese contraste es
  la prueba de que el split funciona.
- Test: ningún split del repo se genera ya por módulo del índice (grep en CI).
- El gate escribe `artifacts/gates/T-split-domain/gate.json` con `pass: true`
  y, por dataset, esqueletos únicos + solape train↔test.

## Done when

- `eval/splits.py` agrupa por esqueleto+dominio+idioma y sella los splits.
- La auditoría retroactiva está publicada dataset por dataset.
- Los verdes históricos sostenidos por `i % 10` están marcados como
  inválidos, con fecha y motivo.
