---
id: T-gen-schemas
title: Q-W-E-N genera esquemas de decisión completos, no plantillas de email
status: next
priority: high
owner: unassigned
category: data
initiative: data-training
depends_on:
  - T-halt-contam
created: 2026-09-21
updated: 2026-09-21
---

# Q-W-E-N genera esquemas de decisión completos, no plantillas de email

Hallazgo D: en ~11 h de loop, Q-W-E-N aportó **5 plantillas frescas en
total**. `fetch_qwen_templates()` pide 5 plantillas cada 24 lotes y descarta
todo lo que no case con los 4 `STATES` hardcodeados o contenga `{}`. El
resultado sabe generar **una** tarea: triaje de email en español con 4
etiquetas fijas. Cero variación de dominio, idioma, cardinalidad de opciones,
`unknown`, hard negatives o formato de estado.

El error de raíz no es el hook: es **pedir plantillas en vez de casos de
decisión**. Aunque el hook funcionara al 100 %, seguiría produciendo el mismo
problema 258 700 veces.

Trabajo, reescribiendo `tools/data_gen_loop.py`:

1. La unidad generada pasa a ser un **esquema de decisión completo**:
   dominio, formato de estado, pregunta, K opciones (K variable) con
   distractores plausibles, respuesta correcta, y una fracción de casos donde
   la correcta es `unknown` porque el estado no la contiene.
2. **Presupuesto por dominio**, no volumen bruto: una cuota por dominio ×
   idioma × cardinalidad, y el loop para cuando la cuota está llena. Nada de
   "358 filas/min indefinidamente".
3. **Dedup por similitud** contra todo lo ya generado, reutilizando
   `leakage.jaccard`: un esquema cuyo 3-grama Jaccard supera el umbral contra
   uno existente se descarta y se cuenta como rechazo.
4. Diversidad medida y publicada: esqueletos únicos, dominios, idiomas,
   distribución de K, % `unknown`, % hard negatives. Si los esqueletos únicos
   no crecen, el loop se para solo — es la señal que faltó durante 11 h.
5. El loop corre como **job del daemon**, con su cuota, no como nohup.

Lo que se genera entra en los splits de `#T-split-domain` **por esquema**,
nunca por fila.

## Verification gate

- Test: 1 000 esquemas generados dan ≥ 200 esqueletos normalizados distintos
  (contra los 190 en 258 700 filas de hoy).
- Test: ≥ 5 dominios y ≥ 2 idiomas representados; ninguna cuota supera su
  presupuesto.
- Test: el dedup rechaza un esquema duplicado inyectado a propósito.
- El gate escribe `artifacts/gates/T-gen-schemas/gate.json` con `pass: true`,
  diversidad por eje y tasa de rechazo del dedup.

## Done when

- El generador emite esquemas de decisión completos con K variable y
  `unknown`, no plantillas de email.
- El presupuesto por dominio está definido y el loop se detiene al llenarlo.
- La curva de esqueletos únicos es visible en el dashboard y el loop se
  autodetiene si se aplana.
