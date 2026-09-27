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
