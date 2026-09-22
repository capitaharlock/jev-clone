---
id: T-gen-schemas
title: Q-W-E-N genera esquemas de decisión completos, no plantillas de email
status: done
priority: high
owner: unassigned
category: data
initiative: data-training
depends_on:
  - T-halt-contam
created: 2026-09-21
updated: 2026-09-21
completed_at: 2026-09-21T15:24:40.321Z
resolved_by: A003
resolved_by_conv: roadmap-architect-uwgjq
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

## Resolution

✓ #data-training #T-gen-schemas done · 14 ficheros · commit `39e6322` · 27 tests verdes
El generador ya no fabrica plantillas: emite esquemas de decisión completos con K 3–8, distractores y `unknown`. **1 152 esquemas desde 1 152 esqueletos únicos** — el corpus viejo tenía 190 esqueletos en 258 700 filas. Gate `pass: true`.

🚀 #data-training #T-prog-gold → A022 (pid 63475 vivo)
· gold programático desde el grafo Wikidata — el grafo decide la respuesta, los teachers solo redactan. Es la raíz que desbloquea #T-mix-1m → #T-mix-5m → #T-data-eval.

⏳ #T-corpus-rebalance sigue vivo (A021, 14 min) · civil-comments 74,6 % → ≤15 %.

<details><summary>Estado de las 14 tareas de #data-training</summary>

- **done (5)**: `T-bigsrc`, `T-massive-huff`, `T-synth-factory`, `T-teacher-intent`, `T-tester-engine`, `T-train-monitor`, y ahora `T-gen-schemas`
- **vivas (2)**: `T-corpus-rebalance` (A021), `T-prog-gold` (A022)
- **encadenadas detrás de prog-gold**: `T-mix-1m` → `T-mix-5m` → `T-data-eval`; `T-mix-10m` en backlog
- **`T-gen-loop` se queda `blocked` a propósito**: sus dos dependencias (`T-halt-contam`, `T-gen-schemas`) ya están done, pero su cuerpo dice literalmente que no se cierra ni se borra — es el registro de por qué el generador cambió de contrato. No lo toco.
</details>

<details><summary>Trabajo de fondo</summary>

- Job `train-decision` corriendo (pid 37091, 42 min de uptime) · `training-monitor` en :8794 (pid 18966).
</details>

— T-gen-schemas · el generador produce esquemas de decisión completos en vez de plantillas (commit 39e6322)

8.9M tokens
