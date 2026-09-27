---
id: T-numeric-gen
title: Generador por regla de comparación de atributos y prioridad — volumen ilimitado para lo que no aprendemos
status: next
priority: high
owner: unassigned
category: data
initiative: daily-learning-loop
depends_on:
  - T-episode-gen
created: 2026-09-27
updated: 2026-09-27
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
