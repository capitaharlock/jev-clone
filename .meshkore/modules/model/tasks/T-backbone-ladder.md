---
id: T-backbone-ladder
title: La escalera de backbones — mismo trainer, encoder más grande, control sin ajustar medido en cada peldaño
status: next
priority: high
owner: unassigned
category: model
initiative: daily-learning-loop
depends_on:
  - T-ce-finetune
created: 2026-09-27
updated: 2026-09-27
---

# La escalera de backbones — mismo trainer, encoder más grande, control sin ajustar medido en cada peldaño

**Por qué.** Peldaño 3 de la escalera de brazos (`docs/bucle-infinito.md` §4).
MiniLMv2-L6 (107 M) aprende clasificación y contrafactuales, pero no
comparación numérica. Un cross-encoder mayor es la forma más directa de
comprobar si ese techo es de capacidad.

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

**Done when.** Al menos dos peldaños por encima del actual medidos (control y
smoke), con tiempos y memoria, y registrados en `training.python.loop` como
peldaños seleccionables.
