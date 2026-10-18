---
id: T-numeric-gen
title: Generador por regla de comparación de atributos y prioridad — volumen ilimitado para lo que no aprendemos
status: done
priority: high
owner: unassigned
category: data
initiative: daily-learning-loop
depends_on:
  - T-episode-gen
created: 2026-09-27
updated: 2026-09-27
completed_at: 2026-09-27
---

# Generador por regla de comparación de atributos y prioridad — volumen ilimitado para lo que no aprendemos

**Por qué.** Es el primer brazo dirigido de la escalera
(`docs/bucle-infinito.md` §4). La comparación de atributos no se aprende ni en
el holdout del piloto: se queda en 0,574 antes y después de 1 000 decisiones
(`#T-ce-finetune`). En esas dos familias el gold sale de una regla
(`rule_trace`: `comparison_gold`, `priority_gold`), así que no hace falta Qwen
para tener volumen. Con prosa local se generan miles por minuto.

**Qué hacer.**
1. `data/episode_gen.py run` ya construye estas familias con prosa `local` y
   profesor `stub`. Añade `--families attribute_comparison,priority_decision`,
   que filtra el plan y mantiene la proporción ES/EN y K=2/3/8. Añade también
   `--variety N`, que amplía las plantillas de estado y de pregunta:
   - **atributos**: al menos 12 atributos (precio, duración, peso, distancia,
     batería, capacidad, valoración, fecha…), unidades y formatos numéricos
     mezclados («1.200 €», «1,2 mil euros», «dos horas y cuarto»), superlativos
     y comparativos («el más barato», «el que menos tarda», «el segundo más
     caro»), y empates que exigen un desempate explícito;
   - **prioridad**: dos y tres criterios encadenados, el orden de los criterios
     como único cambio contrafactual, umbrales («por debajo de 50 €»).
2. Contrafactuales en cada grupo: cambiar sólo un número, sólo el criterio o
   sólo el orden de prioridad cambia el gold. Una paráfrasis no lo cambia.
3. Posición del gold uniforme (χ² escrito antes) y entidades de un vocabulario
   distinto al de la batería dev/sellada (`data/leakage.py` contra ambas).
4. Salida: `artifacts/episodes-rule/batch-XXXX/` con manifest (semilla =
   número de lote). Los `episodes.jsonl` no se versionan; el manifest los
   reconstruye.
5. **Prueba de valor antes de integrarlo en el bucle:** entrena el smoke con
   5 000 de estas más el piloto verificado. Usa el mismo comando que
   `smoke.json` con `--budget 5000`, predicción escrita antes. Mide dev por
   familia y el holdout del piloto por familia. Si atributos no se mueve ni en
   el holdout, el cuello es el backbone: se anota y se pasa a
   `#T-backbone-ladder` sin esperar al bucle.

**Done when.** Generador con tests (gold por regla, contrafactuales,
posición uniforme, sin fuga), un lote de 50 000 publicado, y la prueba de
valor medida y escrita.

## Resolution

**done · 2026-09-27 · developer-copy (`rq-f80e6e`) + A001** — generador entregado
y **prueba de valor NO-GO**. Gate: `artifacts/gates/T-numeric-gen/gate.json`.

### 1. El generador funciona y es prácticamente gratis

`data/rule_variety.py` (75 tests, sin GPU). Lote 1 = **50 000 episodios en 6,7 s**
(7 441 ep/s, CPU), 0 inválidos, 0 duplicados. 15 atributos, 49 combinaciones de
formato numérico, 7 formas de pregunta (superlativo, ordinal, umbral, empate con
desempate, 2 y 3 criterios encadenados, umbral+cadena), K 2/3/8, ES/EN.
Contrafactuales: 12 500 grupos, 12 500/12 500 en las tres variantes. Posición del
gold uniforme (χ² 0,00 / 4,57 / 10,91 contra críticos 6,64 / 9,21 / 18,48 a α=0,01).
Sin fuga: 0 coincidencias de vocabulario y 0 de paráfrasis contra dev **y** sellado.

### 2. La prueba de valor dice que el volumen no es el cuello

Smoke de 5 000 decisiones sobre la mezcla (5 000 por regla + 1 840 del piloto
verificado), mismo control sin ajustar, predicción escrita **antes** de mezclar
(`value_proof_prediction`).

| corte | familia objetivo | control → ajustado | Δ pareado IC95 % | ¿mueve? |
|---|---|---|---|---|
| holdout del piloto (la regla pre-registrada) | attribute_comparison, n=64 | 0,5625 → **0,5625** | 0,000 [−0,172, 0,172] | **no** |
| holdout entero (8× de potencia, informativo) | attribute_comparison, n=564 | 0,3706 → 0,4025 | +0,032 [−0,014, 0,080] | **no** |
| dev por familia | attribute_comparison, n=80 | 0,4125 → 0,3375 | −0,075 | **cae** |

Lo que sí se movió, y mucho: `description_classification` 0,558 → **1,000**,
`extraction_paraphrase` 0,739 → **1,000** en el holdout; dev forzada 0,6225 →
0,6750 (+0,0525 [0,0075, 0,0975]) y contrafactual 0,343 → 0,493 (+0,150
[0,079, 0,221]). El modelo aprende lo que es clasificación y no aprende a
comparar dos números dentro de una secuencia.

### 3. La conclusión, escrita como la pedía la predicción

**Cinco veces el volumen exacto de esa familia no es lo que falta: el cuello es
el backbone.** Pasa a `#T-backbone-ladder` (`docs/bucle-infinito.md` §4, r = 3)
sin esperar al bucle. El generador queda publicado y utilizable: es volumen
barato para las familias que sí se mueven, no la palanca de las numéricas.

Modelo de la prueba: MiniLMv2-L6 cross-encoder, **~107 M parámetros, todos
ajustables** (`artifacts/checkpoints/ce/numeric-5k`).

### 4. La predicción acertó en atributos y FALLÓ en prioridad

La predicción (`value_proof_prediction`, escrita antes de mezclar) decía dos
cosas. La primera se cumplió: atributos no se mueve. La segunda **no**:

> «`priority_decision` se mueve en dev y en el holdout: ya se movió en
> distribución en el piloto (0,62 → 0,78)».

Medido: en el holdout del piloto `priority_decision` se queda en **0,6118 →
0,6118** (Δ 0,000 [−0,106, 0,106], n=85) y en dev CAE de 0,3750 a 0,3250. En el
holdout entero sube +0,041 [0,0017, 0,0826] sobre 605 filas — apenas despega
de 0. O sea: la subida de prioridad del piloto (0,62 → 0,78) **no se reproduce**
con cinco veces el volumen y un corpus más duro. Las DOS familias numéricas se
comportan igual, y eso refuerza el diagnóstico de backbone en vez de debilitarlo:
no es una rareza de una familia.

### 5. Una restricción del backbone salió antes que la conclusión

El scorer tiene una ventana de 512 tokens y `model.ce_scorer.length_report` se
niega a recortar el estado en silencio. Con K = 8 y tres atributos por opción,
**419 de 17 328 pares pasaban de 512** (máximo 570, medido con el tokenizador de
minilmv2-l6-mnli-xnli). De ahí `rule_variety.MAX_STATE_ATTRS = {2: 4, 3: 3,
8: 2}`: las formas que necesitan tres atributos (umbral y tres criterios
encadenados) viven en K = 3, no en K = 8. Con la cota, máximo 482 tokens y 0
recortados en el run real (`holdout.json#length_report`). No es una decisión de
diseño del generador: es el techo del backbone apareciendo en la longitud antes
de aparecer en el acierto.

### 6. Comandos (los que dejaron cada cifra)

```bash
# lote de 50 000 — job `datagen`
env PYTHONPATH=. .venv-train/bin/python -u -m data.rule_variety publish \
    --batch 1 --n 50000 --variety 16
env PYTHONPATH=. .venv-train/bin/python -u -m data.rule_variety gate \
    --dir artifacts/episodes-rule/batch-0001 --required 50000

# predicción, ANTES de todo lo demás
env PYTHONPATH=. .venv-train/bin/python -m eval.numeric_value predict

# prueba de valor — job `trainer`, cadena completa
env PYTHONPATH=. .venv-train/bin/python -m eval.numeric_value mix \
    --rule artifacts/episodes-rule/batch-0001 \
    --pilot artifacts/episodes-qwen/pilot-2k/verified.jsonl \
    --n-rule 5000 --out artifacts/episodes-rule/mix-smoke
env PYTHONPATH=. TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1 \
  .venv-train/bin/python -m training.python.ce_finetune eval --checkpoint none \
    --battery data/battery_dev.jsonl --device mps \
    --out artifacts/gates/T-numeric-gen/eval-none.json
env PYTHONPATH=. TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1 \
  .venv-train/bin/python -u -m training.python.ce_finetune train \
    --episodes artifacts/episodes-rule/mix-smoke \
    --weights minilmv2-l6-mnli-xnli --init none \
    --out artifacts/checkpoints/ce/numeric-5k --device mps --seed 20260926 \
    --epochs 4 --lr-encoder 2e-5 --lr-head 1e-4 --decisions-per-batch 8 \
    --eval-every 1250 --budget 5000 --holdout 0.2 --split-seed 20260927
env PYTHONPATH=. ... ce_finetune gate --run artifacts/checkpoints/ce/numeric-5k \
    --control artifacts/gates/T-numeric-gen/eval-none.json \
    --out artifacts/gates/T-numeric-gen/smoke.json
env PYTHONPATH=. ... -m eval.numeric_value measure \
    --run artifacts/checkpoints/ce/numeric-5k \
    --mix artifacts/episodes-rule/mix-smoke
env PYTHONPATH=. .venv-train/bin/python -m eval.numeric_value gate \
    --run artifacts/checkpoints/ce/numeric-5k
```

Los dos `gate` salen con exit 1 en NO-GO, así que van con `;` y no con `&&` en
el command del job: por eso el job `trainer` aparece como `failed` con todo
medido. El entreno tardó 1 806 s + 579 s de evaluación (5 000 decisiones, 625
pasos, MPS).

### 7. Conteo verbatim de tests

```
$ PYTHONPATH=. .venv-train/bin/python -m pytest data/test_rule_variety.py -q
56 passed in 5.59s
$ PYTHONPATH=. .venv-train/bin/python -m pytest eval/test_numeric_value.py -q
19 passed in 0.55s
$ PYTHONPATH=. .venv-train/bin/python -m pytest data/test_episode_gen.py -q
31 passed in 0.07s
$ PYTHONPATH=. .venv-train/bin/python -m pytest data/ -q
566 passed, 6 skipped in 22.74s
$ PYTHONPATH=. .venv-train/bin/python -m pytest eval/ -q
1 failed, 514 passed, 78 subtests passed in 44.34s
$ PYTHONPATH=. python3 -m eval.gate_rules scan
errors 0
$ ruff check data/rule_variety.py data/test_rule_variety.py \
      data/episode_gen.py eval/numeric_value.py eval/test_numeric_value.py
All checks passed!
```

El único rojo es `eval/test_release_gate.py::TestPublishedVerdict::
test_the_numbers_are_the_ones_t_unseen_labels_published` (0,292141 != 0,304943),
preexistente y ajeno a esta task (`#T-unseen-labels`).
`eval/test_data_eval.py::test_ood_abstention_separates`, el otro rojo conocido,
pasó en esta tanda.

### 8. Nota de operación: el profesor se quedó en la GPU

A mitad del entreno el paso cayó de 11 s a 625 s. Causa: `ollama` seguía
reteniendo **29 GB** de `qwen3.8:27b-mlx` en GPU con `#T-qwen38-ref` ya
terminado (exit 0), y la máquina llegó a **28,8 GB de swap**. Se descargó con
`ollama stop qwen3.8:27b-mlx` (reversible, sin tocar `ollama serve`) y el ritmo
volvió. **Cualquier entreno MPS de este repo debe comprobar `ollama ps` antes de
empezar**, no sólo que el job de Qwen haya terminado: el job termina y el modelo
se queda cargado hasta que expira su `keep_alive`.
