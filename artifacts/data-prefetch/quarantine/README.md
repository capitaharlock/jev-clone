# Cuarentena — synth-loop (#T-halt-contam, 2026-09-21)

`synth-loop-20260921.jsonl` (360 700 filas, ~210 MB) es el corpus del
generador `tools/data_gen_loop.py`, congelado por #T-halt-contam. NO se borra:
es la evidencia de la contaminación. NO vuelve a `artifacts/data-prefetch/` ni
a ningún `JOBS` de entrenamiento.

Por qué está aquí (auditoría 2026-09-21, hallazgo C):

- 190 esqueletos distintos (≈70 plantillas × 4 estados); el más frecuente
  aparece 10 776 veces. La variación es solo nombre/empresa/mes/importe.
- Split por `i % 10` → la misma plantilla cae en train y en test.
- Resultado: acc 1,0000, logloss 0,00074. El dashboard "1.000 → 1.000 → 1.000"
  medía fuga de datos, no capacidad.
- Crecía a ≈358 filas/min sin aportar nada desde la fila ~2 000.

Condición de salida: solo #T-gen-schemas puede sustituir este corpus, con
split por plantilla/dominio (nunca por fila) y firewall previo.
