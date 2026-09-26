---
id: T-preflight-refs
title: Referencias antes de entrenar — Qwen local, NLI sin ajustar y el checkpoint actual
status: done
priority: high
owner: unassigned
category: eval
initiative: honest-eval
depends_on:
  - T-battery-dev
  - T-battery-metrics
  - T-ce-scorer
created: 2026-09-25
updated: 2026-09-27
resolved_by: A003
resolved_by_conv: roadmap-architect-uwgjq
---
# Referencias antes de entrenar — Qwen local, NLI sin ajustar y el checkpoint actual

Paso 2 del piloto, y una puerta: **si ni Qwen responde bien a la batería, el
problema está en la tarea, los datos o el formato, y entrenar sería gastar la GPU
en un enunciado roto.** Medir antes de entrenar es lo que la fase 1 no hizo.

Tres referencias sobre la batería de **desarrollo**, con el protocolo de
información equivalente de `#T-battery-dev` y la suite completa de
`#T-battery-metrics`:

1. **Qwen local** — la referencia de capacidad. Elección estructurada entre IDs
   válidos, no texto libre. Es el techo práctico que tenemos en casa.
2. **El scorer NLI sin ajustar** de `#T-ce-scorer`, por entailment con hipótesis
   explícitas. Si funciona razonablemente, se conserva como punto de partida del
   ajuste. No se asume que un checkpoint NLI resuelva aritmética o preferencias
   complejas sin adaptación.
3. **Los checkpoints pointer actuales** (leverstack 1 M y fullspace 62.528) como
   **control**. Aquí es donde el fallo medido debe reproducirse en la batería
   nueva: eligen «green» diga el estado verde o rojo, y el mismo producto para
   «cuál cuesta menos» y «cuál dura más».

El profesor externo (TypeSafe) no bloquea nada: `#teacher-distill` está en
`backlog` y su papel de referencia lo cubre Qwen. Cuando haya acceso válido y
protocolo comparable, se añade como cuarta columna.

## Verification gate

- Test: las tres referencias se miden con el mismo corte, las mismas filas y el
  mismo protocolo; el runner falla si los n no coinciden.
- Test: el control pointer reproduce en la batería el fallo ya documentado en
  `.meshkore/docs/evidence/probe-mechanism-2026-09-24.json` — si no lo reproduce,
  la batería no es representativa y hay que revisarla.
- Ninguna referencia consulta el test sellado.

## Done when

- La tabla de referencias está publicada con accuracy forzada, con abstención,
  macro por familia, azar por K, IC95 % y n.
- La puerta está resuelta y escrita: o Qwen responde y el piloto sigue, o se
  revisa tarea/datos/formato antes de entrenar.
- Queda decidido y justificado con cifras cuál de los checkpoints preentrenados
  es el punto de partida del ajuste.

## Estado 2026-09-26 — runner entregado, tabla sin medir

El runner está completo y ejecutable en `eval/preflight_refs.py`, con sus 37
tests verdes en 0,7 s y **sin cargar un solo peso** (cada columna recibe su
predictor por inyección; `NoModelTest` lo comprueba en un intérprete limpio).
Medido de verdad, sin GPU: las tres columnas comparten las 400 filas por
identidad y el runner levanta `ProtocolMismatch` si no; ninguna referencia abre
el corte sellado —que ya existe en disco y sigue sin abrirse—; la firma del
fallo se lee de `probe-mechanism-2026-09-24.json` (2/2 pares invariantes en los
dos checkpoints) y el comprobador de reproducción separa un predictor que sólo
lee el texto de la opción de uno que lee el estado.

Lo que falta es **cómputo del operador**, no código: la tabla de referencias, la
puerta de Qwen, la reproducción del control sobre la batería y la elección del
checkpoint de partida están en el gate con `pass: null` y
`reason: awaiting-operator-compute`. Las reglas con las que se decidirán
(`REPRO_RULE`, `QWEN_GATE_RULE`, `STARTING_POINT_RULE`) están escritas **antes**
de que exista una sola medición, que es el único momento en que escribirlas
significa algo. Lo firma el job parado `preflight-refs`
(`.venv-train/bin/python -m eval.preflight_refs refs`).

## Estado 2026-09-27 — las cuatro casillas medidas, veredicto GO

Cómputo autorizado y gastado. Las cuatro columnas están medidas sobre las
MISMAS 400 filas del corte de desarrollo (`rows_sha256`
`8e8ccea5…8535c3fb`, idéntico en las cuatro) y el gate está firmado con
cifras reales: `verdict: PASS`, `resolved: true`, ninguna casilla en
`null`. El corte sellado sigue sin abrirse.

**La puerta de Qwen ABRE.** Elección forzada 0,965 (386/400), IC95 %
[0,942 · 0,979] contra un azar de 0,3375 a K medio 4,1; macro por familia
0,965; y las cinco familias despejan su propio azar por separado —la peor,
`description_classification`, en 0,9375 con IC [0,862 · 0,973]. La
condición que se escribió para que un cero de familia no se esconda tras
una media global no tuvo que dispararse. **El enunciado, los datos y el
formato no están rotos: entrenar sobre esta batería no es gastar la GPU en
una tarea imposible.**

**El control pointer REPRODUCE el fallo, en los dos checkpoints.**
`fullspace-62528` e `leverstack-1m` se quedan en 0,330 y 0,3275 de
elección forzada, con el azar en 0,3375 — es decir, en el azar, con el
intervalo cubriéndolo. La invariancia documentada el 2026-09-24 sale
también aquí: 0,964 y 0,929 de los 140 grupos contrafactuales reciben el
MISMO slot en las dos mitades, y el acierto conjunto es 2/140 y 1/140
(0,0143 y 0,0071) contra un azar conjunto de 0,0826, con los dos
intervalos enteros por debajo. La batería es representativa del fallo con
el que se la comparó, así que no hay que revisarla.

**El punto de partida del ajuste es `nli-nograd`** (minilmv2-l6-mnli-xnli
`@0a71e92a`), el único candidato elegible: 0,6225 de elección forzada, IC
[0,574 · 0,669] sobre un azar de 0,3375, y acierto conjunto contrafactual
0,343 (48/140) muy por encima del conjunto de 0,0826 — sigue el hecho
decisivo en vez de fijarse en el texto de la opción. Los dos checkpoints
pointer quedan fuera por la regla escrita antes de medir: quien reproduce
la invariancia documentada es el control, no un punto de partida. El
checkpoint contaminado con BANKING77 es `modernbert-zeroshot-v2`, que NO
es el elegido: el que sale no arrastra ese caveat.

**Lo que acota la cifra de Qwen.** Su control de permutación
`#T-option-text` FALLA: en la muestra de 24 filas fijada por sha antes de
medir, 3 elecciones cambian al ofrecer los candidatos en orden inverso
(flip_rate 0,125). El chooser muestrea a temperatura 0,7, así que ese 12,5
% acota junto sensibilidad al orden y ruido de muestreo sin separarlos —es
la cota superior de estabilidad de orden, no su descomposición. La cifra
se publica AL LADO del veredicto (`the_qwen_gate.
order_stability_beside_the_verdict`) y en la tabla, con
`enters_the_rule: false`: `QWEN_GATE_RULE` se escribió antes de que
existiera una medición y añadirle una condición después de ver este fallo
sería elegir la regla por el resultado.

Lo firma el job `evalgate` (`.venv-train/bin/python -m
eval.preflight_refs refs`); la tabla y el gate se rehacen sin modelo con
`eval.preflight_refs table`, porque cada columna medida queda en
`artifacts/gates/T-preflight-refs/columns/` sellada con las filas sobre
las que se midió.

## Resolution

**GO — el piloto sigue.** Las tres referencias están medidas sobre las
mismas 400 filas y el gate (`artifacts/gates/T-preflight-refs/gate.json`)
está firmado `PASS` con las diez casillas resueltas. Qwen responde la
batería (0,965 contra 0,3375, las cinco familias despejando), el control
pointer reproduce en ella el fallo de mecanismo documentado (azar en
forzada, 2/140 y 1/140 en conjunto contrafactual contra 0,0826) y el
punto de partida del ajuste queda decidido con cifras: `nli-nograd`.
`#T-ce-finetune` puede arrancar desde ahí.

Queda dicho y no tapado: el control de orden de la columna Qwen falla al
12,5 % en 24 filas, y esa cota viaja con el 0,965 a donde se cite.
