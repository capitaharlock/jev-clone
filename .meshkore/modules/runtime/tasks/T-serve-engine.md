---
id: T-serve-engine
title: Conectar el servidor al motor neuronal, cargar el encoder afinado y arreglar la clave de caché V1
status: next
priority: high
owner: unassigned
category: runtime
initiative: oss-release
created: 2026-09-24
updated: 2026-09-24
---

# Conectar el servidor al motor, cargar el encoder afinado, arreglar la caché

Tres hallazgos de la revisión externa
(`.meshkore/docs/revision-externa-2026-09-24.md`, J/K/L), verificados leyendo el
código. Ninguno explica el resultado a 77 vías —salen del pipeline Python— pero
los tres bloquean **servir** el modelo de fase 2 cuando exista, y el segundo es
específicamente el régimen que fase 2 va a producir.

## 1 · El servidor no usa el motor (J)

Existe implementación Candle real del encoder y del pointer head
(`crates/jev-model/src/{engine,modernbert,pointer}.rs`). Pero `jevclone serve`
verifica opcionalmente un artefacto y llama a `jev_server::serve` **sin
entregarle un `Engine`** (`crates/jev-cli/src/main.rs:37-60`); el servidor crea
`jev_runtime::Runtime::new()` (`crates/jev-server/src/lib.rs:263-278`) y
`/v1/choice` recibe `weights`, `bias` e `input` por la petición y se los pasa al
scorer lineal (`v1.rs:28-37, 141-154`; `jev-runtime/src/lib.rs:153-168`).

Las opciones de esa API sólo llevan `id`, sin texto semántico, y `unknown` se
decide por umbral de probabilidad: es un contrato **distinto** del pointer
entrenado, que puntúa el texto de cada opción y aprende su propia abstención.
Hay que decidir y escribir el contrato de la API sobre el modelo real —opciones
con texto, `unknown` aprendido— y conectar el `Engine`.

## 2 · El cargador Rust ignoraría un encoder afinado (K)

El trainer guarda `backbone.safetensors` con su nombre y su hash cuando el
encoder es entrenable (`training/python/train_decision.py:973-978`), y Python lo
carga (`:1022-1035`). En Rust, `BackboneRef` sólo deserializa `id`,
`hidden_size` y `revision` (`crates/jev-model/src/engine.rs:40-44`) y
`Engine::load` carga **siempre**
`artifacts/weights/<id>/pytorch_model.bin` (`:149-170`): nunca mira
`backbone.weights_file` ni su hash.

Un checkpoint de fase 2 cargaría con la cabeza nueva y el encoder original —
silenciosamente, sin fallar. O se carga y se verifica el backbone del
checkpoint, o se **rechaza explícitamente** ese régimen con un error claro. La
paridad de encoders congelados no cubre este caso.

## 3 · La clave de caché de la API V1 colisiona (L)

`choose()` construye `state_hash` y `tokenizer_hash` como cadenas vacías
(`crates/jev-server/src/v1.rs:144-147`). Esa clave compuesta es la que usa
`Runtime::encode()` para devolver el vector guardado
(`jev-runtime/src/lib.rs:153-162`): dos peticiones con la misma `model_version` y
entradas distintas comparten clave, y la segunda puede puntuar la entrada de la
primera. Afecta también a `choose_batch()`.

Caso mínimo, deducido del código: dos opciones, `weights=[1,-1]`, `bias=[0,0]`,
`input=[1]` y después `input=[-1]`, misma versión. Lo correcto pasa de
≈`[0,881, 0,119]` a ≈`[0,119, 0,881]`; con esta clave la segunda petición
reutiliza la primera. La clave tiene que derivarse del contenido real de la
petición (input, pesos, bias, tokenizador).

## Done when

- `jevclone serve` levanta el servidor con un `Engine` real y `/v1/choice`
  responde sobre el modelo, con el contrato de opciones-con-texto y `unknown`
  aprendido escrito en la spec de la API.
- El cargador Rust lee `backbone.weights_file` y su hash y carga esos pesos, o
  rechaza el checkpoint con un error explícito; hay test para ambos caminos.
- La clave de caché V1 se deriva del contenido de la petición, con un test de
  dos peticiones sucesivas de input distinto (resultados distintos) y una
  repetición idéntica (acierto de caché).
- `#T-candle-infer` queda actualizada: su texto «no hay Candle» está
  desactualizado respecto al código presente.
