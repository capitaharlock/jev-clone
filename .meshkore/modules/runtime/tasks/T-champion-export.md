---
id: T-champion-export
title: Exportar el campeón MLX a safetensors
status: active
priority: high
owner: architect-master
category: runtime
initiative: champion-runtime
depends_on: []
created: 2026-10-15
updated: 2026-10-15
---

# Exportar el campeón MLX a safetensors

Fusionar el adaptador LoRA r16 en el backbone, guardar la cabeza de puntero
por separado y escribir `config.json`, tokenizer y un manifiesto con el hash
del checkpoint de origen.

## Criterios

- Corre con la GPU solo entre ciclos del autoloop o en su fase de benchmark.
- El modelo fusionado reproduce en MLX la puntuación de la sonda v3 del
  campeón.
- Una promoción nueva genera una exportación nueva; las anteriores se
  conservan por hash.

## Estado del 16 de octubre

- `tools/export_champion.py` fusiona las 252 capas LoRA en CPU en menos de un
  minuto y escribe `artifacts/runtime/champion-cycle-0005/` con la misma
  estructura que el snapshot v1, más `export.json` con los hashes. El enlace
  `artifacts/runtime/latest` apunta a la última exportación.
- Antes y después de fusionar se puntúa un lote fijo de tokens aleatorios. La
  deriva máxima de los logits de puntero fue 0,034; el límite es 0,05.
- El autoloop lanza la exportación en segundo plano tras cada promoción. El
  proceso del autoloop en marcha lo recoge en su próximo reinicio.
- Falta comprobar la sonda v3 con el modelo fusionado.
