---
id: T-optset-sampler
title: Sampler de opciones dinámicas y hard negatives sobre el corpus ya convertido
status: active
priority: high
owner: unassigned
category: data
initiative: decision-rebuild
depends_on:
  - T-halt-contam
created: 2026-09-21
updated: 2026-09-21
---

# Sampler de opciones dinámicas y hard negatives sobre el corpus ya convertido

El material bueno ya está en disco: los converters (`convert_banking77.py`,
`convert_massive.py`, `convert_huffpost.py`, `convert_logiqa_reclor.py`)
preservan estado, opciones y textos correctamente. Lo que falta es el
*dataloader* que alimenta a `#T-pointer-head` con el contrato correcto.

Hallazgo B, segunda mitad: **HuffPost convierte a 4 candidatos por fila, pero
el trainer entrena sobre las 41 categorías globales**, destruyendo el
contrato de opciones dinámicas. Ese es exactamente el fallo que este sampler
impide que vuelva a ocurrir.

Trabajo, en `data/optset.py`:

1. Por fila, muestrear **K opciones dinámicas** (K variable, no fijo) que
   incluyan siempre la correcta y K-1 distractores.
2. **Hard negatives** por similitud, reutilizando `data/hardneg.py` y
   `leakage.jaccard`: el distractor plausible (`card_arrival` vs
   `card_delivery_estimate` en banking77, categorías hermanas en HuffPost)
   pesa más que el aleatorio. Mezcla explícita fácil/difícil, medida.
3. Filas `unknown`: una fracción del batch donde la respuesta correcta **no
   está** entre las opciones ofrecidas, para entrenar la salida `unknown`
   de verdad y no como umbral.
4. Barajado del orden de las opciones en cada epoch (alimenta el test de
   invariancia de `#T-pointer-head`).
5. Cobertura inicial: banking77, massive, huffpost, boolq — lo ya convertido
   y limpio. Nada de synth-loop, nada de logiqa/reclor (eval-only).

El sampler NO reescribe los jsonl: lee el schema V1 tal cual y construye los
batches. `data/schema.py` se conserva intacto.

## Verification gate

- Test: ninguna muestra emitida tiene el conjunto de opciones igual al
  espacio global de etiquetas del dataset (el bug de HuffPost no puede
  reproducirse).
- Test: la opción correcta aparece en posición uniforme (chi-cuadrado sobre
  10 k muestras) — sin prior posicional como el que dio 0,435 en LogiQA.
- Test: la fracción de filas `unknown` y la mezcla hard/easy son parámetros
  verificados por conteo, no implícitos.
- El gate escribe `artifacts/gates/T-optset-sampler/gate.json` con
  `pass: true`, distribución de K, % hard negatives y % `unknown` por
  dataset.

## Done when

- `data/optset.py` emite batches con K variable, distractores duros y filas
  `unknown`, sobre banking77 + massive + huffpost + boolq.
- Los tres tests del gate pasan y el artifact publica la composición real.
- Está documentado que el trainer nunca vuelve a ver un espacio de etiquetas
  global.
