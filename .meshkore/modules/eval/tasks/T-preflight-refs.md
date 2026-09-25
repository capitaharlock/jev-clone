---
id: T-preflight-refs
title: Referencias antes de entrenar — Qwen local, NLI sin ajustar y el checkpoint actual
status: next
priority: high
owner: unassigned
category: eval
initiative: honest-eval
depends_on:
  - T-battery-dev
  - T-battery-metrics
  - T-ce-scorer
created: 2026-09-25
updated: 2026-09-25
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
