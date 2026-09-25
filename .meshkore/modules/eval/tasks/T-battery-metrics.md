---
id: T-battery-metrics
title: Las métricas mínimas, publicadas juntas — ranking y abstención por separado
status: active
priority: high
owner: unassigned
category: eval
initiative: honest-eval
created: 2026-09-25
updated: 2026-09-25
---

# Las métricas mínimas, publicadas juntas — ranking y abstención por separado

Hoy se puede publicar un número que parece bueno porque el corte que lo produce
está elegido después. Esta task fija la suite que acompaña a **toda** medición
del piloto, y se construye antes de que exista la batería porque también se
aplica retroactivamente a los artefactos ya en disco.

Métricas obligatorias en cada veredicto:

- Accuracy **forzada** y accuracy **incluyendo abstención**, nunca una sola.
- Cobertura y precisión **entre las respondidas**.
- **Macro por familia** (la media global puede pasar con una familia a cero).
- **Azar por K**, calculado y escrito al lado de cada cifra.
- **Éxito conjunto en pares contrafactuales**: acertar el original y su variante;
  acertar uno de los dos no cuenta.
- **Invariancia a permutaciones**: el control de `#T-option-text` se incorpora
  aquí, y se distingue explícitamente del seguimiento del estado y la pregunta —
  son dos propiedades distintas y la prueba del 2026-09-24 las separa: las
  puntuaciones siguen al texto (<5 × 10⁻⁷ al permutar) y aun así el modelo falla
  el cambio decisivo.
- **NLL, Brier y calibración**, con temperatura/umbral calibrados en desarrollo y
  **verificados** en test.

Regla que hace honesta la lectura: **ranking y abstención se evalúan aparte**. Un
87,7 % de abstención con 0/1000 no es prudencia, es un ranking al azar tapado por
un umbral. Y los pesos softmax son relativos a los candidatos ofrecidos: no se
publican como probabilidad absoluta de verdad.

Incluye la corrección de narrativa que dejó pendiente `#T-fullspace-objective`: la
lectura automática de `verdict.reading` dice que el intervalo primario contiene el
azar cuando sus números lo excluyen por debajo (el que lo contiene es el de
elección forzada), y el resumen de `#T-bigk-optsets` afirma haber descartado la
cardinalidad como causa cuando un negativo con una semilla y un presupuesto sólo
descarta el beneficio observado de ese brazo. Se corrigen los textos y se añade la
regla de coherencia que impide volver a publicarlos: **un gate cuya lectura
contradiga sus propios números falla, no avisa.**

## Verification gate

- Test: un veredicto sintético al que le falte cualquiera de las métricas
  obligatorias, o el azar de su K, o el n de su corte, hace fallar el runner.
- Test: un artefacto con `reading` que afirme «contiene el azar» mientras su IC lo
  excluye hace fallar el runner.
- Test: el éxito conjunto en contrafactuales se calcula por `variant_group` y es
  0 cuando sólo se acierta una mitad.
- Los gates históricos que incumplen quedan **marcados, no borrados**.

## Done when

- La suite existe como una única función de reporte que todo gate del piloto
  invoca; no hay métricas calculadas a mano por run.
- Las tres lecturas erróneas identificadas están corregidas en disco y la regla
  que las prohíbe tiene test.
- Cada cifra publicada lleva su n, su K, su azar y su IC95 %.
