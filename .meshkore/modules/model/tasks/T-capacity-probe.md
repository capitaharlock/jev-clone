---
id: T-capacity-probe
title: ¿Capacidad o entreno? Sobreajustar las familias numéricas antes de subir de backbone
status: next
priority: high
owner: unassigned
category: model
initiative: daily-learning-loop
depends_on:
  - T-numeric-gen
created: 2026-09-30
updated: 2026-09-30
---

# ¿Capacidad o entreno? — la prueba que falta antes de culpar al backbone

**Por qué.** `#T-numeric-gen` cerró «el cuello es el backbone» con un smoke de
**625 pasos, 1 época, lote 8, LR constante sin warmup**, cuya pérdida de entreno bajó
sólo de **1,07 a 1,02** (`artifacts/checkpoints/ce/numeric-5k/curve.jsonl`), casi el
azar de la mezcla K=2/3/8. El modelo **no llegó a ajustar lo que vio**: eso no separa
«no puede» de «no se le dejó entrenar». Además se usaron 5 000 de los 50 000 episodios
por regla ya generados. La conclusión de §0.1 punto 2 de `docs/bucle-infinito.md` es,
por tanto, **prematura**.

**Qué se mide.** `training/python/ce_capacity.py` (nuevo): 512 episodios por regla de
atributos + prioridad (fit) y 1 024 de grupos distintos (unseen), MiniLMv2-L6 sin ajustar,
20 épocas. Umbrales escritos antes en `PREDICTION`: fit > 0,95 → hay capacidad y el
smoke estaba sub-entrenado; fit < 0,70 → límite de representación en formato pairwise.

**Parcial (2026-09-30, interrumpido en la época 8 de 20 por el operador: en esta
máquina no se entrena).** `artifacts/gates/T-capacity-probe/partial.json`:

| época | pérdida | fit | unseen |
|---|---|---|---|
| 0 | — | 0,342 | 0,315 |
| 4 | 1,20 | 0,551 | 0,390 |
| 8 | 0,86 | 0,756 | 0,422 |

Lectura provisional, no veredicto: el fit sube sin estancarse (ya pasa el 0,70 de
«límite»), así que el MiniLM **sí** ajusta estas familias cuando se le entrena; unseen
sube +0,11 pero despacio. Esto apunta a **generalización/formato**, no a incapacidad pura.

## Qué hacer en la máquina de entreno (en este orden)

1. **Terminar la sonda:** `PYTHONPATH=. .venv-train/bin/python -m training.python.ce_capacity run --device mps`
   (~20 min). Publicar `fit.json` con su veredicto.
2. **Entreno de verdad, mismo backbone:** los 50 000 por regla + piloto verificado,
   ≥ 2 épocas, `--schedule warmup-linear` (nuevo en `ce_finetune.py`; `constant` sigue
   siendo el defecto y reproduce lo publicado), `--holdout 0.1`. Medir dev y holdout por
   familia contra el control sin ajustar. ~70 min por época en MPS.
3. **Decidir con la cifra:**
   - Si unseen/holdout numérico sube de forma clara (≥ 0,65) → el volumen y el entreno
     sí eran el cuello; el bucle sigue con MiniLM y escala volumen.
   - Si fit llega a ~1,0 y unseen se queda plano → el formato pairwise es el límite:
     `#T-listwise-format` **antes** que `#T-backbone-ladder` (es más barato y ataca la
     causa: cada opción se puntúa sin ver a las otras).
   - Si fit no pasa de 0,70 → capacidad: `#T-backbone-ladder`.

## Done when

- `artifacts/gates/T-capacity-probe/fit.json` publicado con las 20 épocas y veredicto.
- Entreno largo (paso 2) medido contra el control, con manifest y curva.
- §0.1 de `docs/bucle-infinito.md` actualizado con la rama que tocó (paso 3).
