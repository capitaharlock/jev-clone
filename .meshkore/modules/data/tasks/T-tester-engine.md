---
id: T-tester-engine
title: Tester engine terminal con juez LLM y minería de training sets
category: data
initiative: data-training
status: done
created: 2026-09-20
completed_at: 2026-09-20T17:38:35.250Z
resolved_by: A004
resolved_by_conv: general-09192230
---
# T-tester-engine — tester engine en terminal

CLI interactivo que prueba el modelo v0, muestra request + latencia, pide
veredicto a un juez LLM (Qwen local vía ollama, Gemini Flash opcional) y
mina las sesiones aceptadas como nuevos training sets con variedad
estructurada.

## Done when

- `tools/tester_engine.py --once --task massive --text "..."` imprime
  request JSON, top-5 del modelo, latencia ms y veredicto del juez.
- Sesión interactiva (REPL) guarda `artifacts/tester/sessions/*.jsonl`.
- `--emit-training` escribe filas en schema canónico a
  `artifacts/tester/mined/*.jsonl` con matriz de cobertura por slice.
- Juez Qwen funciona con ollama local; sin ollama cae a heurística sin romper.

## Resolution

No, no falla todo: 5 de 6 van ok, solo falla `call mom please` en #data-training / #T-tester-engine. Ese fallo es un hueco de cobertura, no un modelo roto, y el arreglo es entrenamiento: añadir el slice de llamadas y reentrenar.
Vivos ahora: Qwen (ollama) + conversor Qwen; el entrenador baseline está parado (último run hoy) y los minados ya caen en `artifacts/tester/mined/`. Tus 2 testers viejos (PIDs 41267, 33409) siguen colgados con código antiguo: mátalos.

<details><summary>El fallo — qué dice realmente</summary>

- `pred=calendar_set p=0.142` no es "confunde llamadas con calendario": es el modelo diciendo "no sé" (confianza bajísima).
- La cabeza `massive` del v0 no tiene clase de llamadas (`phone_call`/`call_contact`); ante algo fuera de cobertura, reparte probabilidad y cae donde puede.
- El juez Qwen hace bien su trabajo al marcar `wrong`: para eso existe el forward-testing, cazar estos slices.
- Los otros 5 casos (alarm, música, tiempo, luces, reminder) están ok con p>0.8: el modelo base funciona en su cobertura.
</details>

<details><summary>El arreglo (training) — plan</summary>

- El caso `call mom` + sus variantes (llama a X, phone mum, haz una llamada) entran como slice nuevo: 200-500 ejemplos vía fábrica sintética + minado del tester con `--emit-training`.
- Se reentrena la cabeza `massive` (v6) con ese slice mezclado; el `p=0.142` debería pasar a `phone_call p>0.8`.
- Los `ok` que ya minaste (`artifacts/tester/mined/*.jsonl`) sirven como regresión para no romper lo que hoy funciona.
- Si quieres, lo lanzo yo ahora: dime y arranco slice + entrenamiento v6 en fondo.
</details>

<details><summary>Limpieza — procesos colgados</summary>

- `41267` corre el tester viejo (el que se quedaba mudo esperando al juez); `33409` está suspendido (`T`, tu `^Z`).
- En cada terminal: `Ctrl-C`, y el suspendido además `kill %1` o `kill 33409 41267`.
- Re-ejecuta con el código nuevo de dos fases (predicción al instante, veredicto después).
</details>

28M tokens
