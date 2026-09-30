---
id: T-backbone-ladder
title: La escalera de backbones — mismo trainer, encoder más grande, control sin ajustar medido en cada peldaño
status: active
priority: high
owner: unassigned
category: model
initiative: daily-learning-loop
depends_on:
  - T-ce-finetune
  - T-numeric-gen
created: 2026-09-27
updated: 2026-09-30
---

> **2026-09-30:** en espera de `#T-capacity-probe`. La premisa «MiniLM no las aprende ni en
> su propia distribución» salió de un smoke sub-entrenado (pérdida 1,07 → 1,02); la sonda
> parcial ya ajusta el 0,76 del fit. Se sube de peldaño sólo si la sonda da capacidad < 0,70
> o si listwise tampoco mueve unseen.

> **2026-09-28:** Jev resuelve atributos y prioridad al 100 % en nuestro dev; MiniLM no
> las aprende ni en su propia distribución. Esta task es **el salto esperado**, no un
> plan B: pasa a P0 en cuanto el bucle lleve 3 ciclos seguidos sin mover esas familias
> (`docs/bucle-infinito.md` §0.1). Si un peldaño no cabe en tiempo o memoria, se pide
> cómputo al operador con la cifra medida.

# La escalera de backbones — mismo trainer, encoder más grande, control sin ajustar medido en cada peldaño

**Por qué.** Peldaño 3 de la escalera de brazos (`docs/bucle-infinito.md` §4).
MiniLMv2-L6 (107 M) aprende clasificación y contrafactuales, pero no
comparación numérica. Un cross-encoder mayor es la forma más directa de
comprobar si ese techo es de capacidad.

**Activada el 2026-09-27 por la medida de `#T-numeric-gen`, no por calendario.**
Ese brazo dio 5 000 decisiones de exactamente la familia que falla y
`attribute_comparison` **no se movió**: 0,5625 → 0,5625 en el holdout del piloto
(Δ pareado 0,000 [−0,172, 0,172]) y 0,3706 → 0,4025 en el holdout entero, 564
filas, IC95 % [−0,014, 0,080], sin despegarse de 0. En la misma corrida
`description_classification` fue 0,558 → 1,000 y `extraction_paraphrase` 0,739 →
1,000, así que el trainer y los datos funcionan. **El volumen no es el cuello;
el encoder sí lo es**, y esta task es la que lo prueba o lo desmiente
(`artifacts/gates/T-numeric-gen/gate.json`, sección `value_proof.bottleneck`).

**Peldaños** (cada uno con fila en `source-register.md`, revisión fijada y sha
verificado por `model/weights.py`):
1. `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` (hoy)
2. `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (≈ 280 M, ES/EN)
3. `joeddav/xlm-roberta-large-xnli` o un NLI de ModernBERT-large (≈ 400 M).
   Elige por licencia y multilingüe, y escribe por qué.

**Qué hacer.**
1. `ce_finetune.py` ya lee `--weights`: registra cada peldaño en
   `model/weights.py` y comprueba que `format_fingerprint()` no cambia. Si el
   modelo no tiene la clase entailment en el índice 0, el scorer lo lee de la
   config (`entail_index`); pon un test.
2. Mide el **control sin ajustar** de cada peldaño en dev
   (`ce_finetune eval --checkpoint none --weights <id>`) y publícalo antes de
   entrenar nada.
3. Smoke de 5 000 decisiones por peldaño con la misma mezcla que el último
   ciclo promovido. Anota la memoria en MPS y los segundos por 1 000
   decisiones: esto actualiza la tabla de tiempos del manual §2.
4. Gate por peldaño contra su control y contra el `current` del bucle, con
   la regla de promoción del manual §3.

## Done when

1. Los peldaños 2 y 3 están registrados en `model/weights.py` con revisión fija y
   `sha256` verificado, y su fila está en `.meshkore/docs/source-register.md`.
2. El **control sin ajustar** de cada peldaño está medido en dev y publicado
   *antes* de entrenarlo (`ce_finetune eval --checkpoint none --weights <id>`),
   con `format_fingerprint()` idéntico al del peldaño 1.
3. Cada peldaño tiene un smoke de 5 000 decisiones con la misma mezcla que el
   último ciclo promovido, y su gate escrito contra su propio control y contra el
   `current` del bucle (regla de promoción del manual §3).
4. `attribute_comparison` en el holdout del piloto (n=64) tiene su Δ pareado con
   IC95 % para cada peldaño — la cifra que decide si el techo era de capacidad.
   La predicción se escribe antes de entrenar, como en `#T-numeric-gen`.
5. Segundos por 1 000 decisiones y memoria MPS anotados por peldaño, y la tabla
   de tiempos de `docs/bucle-infinito.md` §2 actualizada con ellos.
