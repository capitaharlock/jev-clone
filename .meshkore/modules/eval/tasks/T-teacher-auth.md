---
id: T-teacher-auth
title: Desbloquear el profesor — endpoint identificado, key rechazada
blocked_reason: necesita del operador una key válida de Jev (proveedor exacto + nombre del modelo); la aportada devuelve 401 en los seis endpoints probados
status: blocked
priority: medium
owner: unassigned
category: eval
initiative: teacher-distill
created: 2026-09-24
updated: 2026-09-27
---

> **Reactivada el 2026-09-27** como primera task de `#teacher-distill` (P2). Es la única del roadmap que **no puede hacer un agente solo**: hace falta que el operador aporte la key válida y diga el proveedor. Cuando llegue, el resto son dos variables de entorno y un gate.
>
> **Cómo se comprueba una key en 30 segundos** (sin gastar saldo):
> `curl -s -H "Authorization: Bearer $KEY" https://api.typesafe.ai/v1/models` → 200 con el catálogo = válida; 401 = no válida; 403 «Must supply an API key» = cabecera mal puesta. La key vive en `.meshkore/credentials/` (gitignored) y se lee por `eval/teacher.py::api_key()`; nunca se pega en una task, un gate ni un log.

# Desbloquear el profesor — endpoint identificado, key rechazada

Medido el 2026-09-24. El proveedor que indicó el operador es **TypeSafe** y su
API responde en `https://api.typesafe.ai`:

| prueba | resultado |
|---|---|
| `GET /v1/models` sin credencial | `403 · "Must supply an API key!"` |
| `GET /v1/models` con `x-api-key: <key>` | `403 · "Must supply an API key!"` → cabecera no leída |
| `GET /v1/models` con `Authorization: Bearer <key>` | **`401 · "Cannot authenticate... check your API key"`** |

Eso es decisivo y acota el problema: **la URL y el esquema son correctos**
(Bearer), lo que el servidor rechaza es la credencial. La key aportada
(`apikey_<32hex>_<8hex>`) está caducada, es de otro entorno o no tiene ese
scope. No es un problema de cableado nuestro y no se arregla probando más
endpoints.

Bloquea `#T-teacher-probe` (la distancia fila a fila) y `#T-teacher-kappa` (uno
de los criterios fallados de `#T-release-gate` por evidencia ausente). Mientras
tanto la referencia externa la da `#T-jev-parity`, que no gasta saldo.

## Qué hacer

1. Pedir al operador una key válida de `api.typesafe.ai` y el **nombre exacto
   del modelo** a usar como profesor (el catálogo se lista en `/v1/models` en
   cuanto autentique).
2. Cablear `eval/teacher.py` contra ese endpoint con la key **fuera de git**
   (fichero en `.meshkore/credentials/`, nunca en un JSON de gate ni en el
   repo), caché en disco por hash de `(modelo, prompt, opciones)`, backoff y
   contador de tokens/coste.
3. Presupuesto declarado antes de la primera llamada masiva: una estimación en
   € del corte completo, y un tope que el cliente respeta y publica.

## Done when

- Una llamada real a `api.typesafe.ai` autentica y el catálogo de modelos queda
  registrado, con el modelo profesor elegido y escrito.
- La credencial vive fuera del repo y ningún artefacto la contiene; hay test
  que lo verifica.
- El cliente cachea: re-ejecutar el mismo experimento cuesta 0 llamadas, y todo
  gate que lo use publica llamadas y coste.
- `#T-teacher-probe` deja de estar bloqueada por credencial.
