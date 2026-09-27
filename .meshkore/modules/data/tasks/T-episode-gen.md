---
id: T-episode-gen
title: Qwen genera episodios completos por familia, no plantillas rellenadas
status: done
priority: high
owner: unassigned
category: data
initiative: episodic-data
depends_on:
  - T-episode-contract
created: 2026-09-25
updated: 2026-09-27
outcome: partial
---

> **Estado 2026-09-27 12:40.** El piloto murió a 1 394/2 100 (último escrito 07:37, ollama caído). Se añadió `--resume` a `data.episode_gen run` (mismo `--n` y `--seed`: retoma en el ítem que toca, en modo append; distinto plan → error) con tests, se relanzó ollama y el job `datagen` con `--resume`, y el `command` del job canónico ya lo lleva. Copia de seguridad de los 1 394 en el scratchpad de la sesión. **Quien recoja esta task:** vigilar `artifacts/episodes-qwen/pilot-2k/run-resume.log`; al terminar, el propio job firma el gate (`volume_ge_2000`); si vuelve a morir, relanzar el job tal cual (retoma).

# Qwen genera episodios completos por familia, no plantillas rellenadas

Sustituye a `#T-labelspace-factory`. El generador deja de producir taxonomías
para rellenar una plantilla (`Field report: …` / `Cross-check against: …`) y pasa
a producir **episodios** conformes a `#T-episode-contract`, en las cinco
familias:

1. **Extracción y paráfrasis** — hechos explícitos, reformulados sin copiar la
   frase del candidato.
2. **Comparación de atributos** — precio, duración, calidad definida por regla,
   restricciones y desempates.
3. **Clasificación por descripciones** — categorías nuevas cuya definición viaja
   con la opción.
4. **Inferencia y negación** — evidencia favorable, contraria y datos
   insuficientes.
5. **Decisiones con prioridades** — «prioriza duración; si empatan, menor
   precio». Nunca «mejor» sin criterio.

Dos reglas de método que no son negociables:

- **Los atributos viajan en el estado.** No se extrae el catálogo del mundo de
  Qwen: para comparar dos productos, sus atributos están en `state` o se
  recuperan de una fuente. El estudiante aprende a aplicar un criterio a esos
  hechos, no a memorizar catálogos.
- **Gold por regla en lo numérico.** Qwen redacta la prosa; los hechos
  estructurados y la respuesta correcta de una comparación numérica los fija una
  regla determinista. Al profesor se le pide una **elección estructurada entre
  IDs válidos** y su evidencia, y se guarda la traza. Los porcentajes de
  confianza que escriba un LLM no se usan como probabilidades.

Qwen local es la referencia de capacidad y la fuente; no es el modelo servido.

## Verification gate

- Test: el 100 % de los episodios publicados pasa el validador de
  `#T-episode-contract`; los rechazados quedan con motivo, no se descartan en
  silencio.
- Test: en la familia de comparación, alterar un atributo del estado cambia el
  gold que calcula la regla — el generador no puede publicar un gold que la regla
  contradiga.
- Test: ningún episodio contiene el marcador de plantilla de `data/labelgen.py`
  ni una pregunta igual a un ID de dataset.


## Cómputo del piloto — medido, no estimado (2026-09-27)

El piloto de 2 000 episodios no cabía en el generador que había: pedía
**una petición a Ollama por episodio**, y con `qwen3.6:27b-mlx` una
petición cuesta ~19 s de sobrecarga fija contra ~3 s de cómputo real
(26 tokens generados en 3,4 s dentro de 21 s de reloj). Concurrencia 4
sólo da ×1,5 (0,062 vs 0,041 llamadas/s medidas sobre 8 llamadas). Es
decir: **~18 h de GPU** sin tocar nada.

La palanca que sí mueve la aguja es meter muchos episodios en la misma
petición. Medido el 2026-09-27 sobre el modelo real:

| Camino | s/item | Piloto de 2 000 |
|---|---:|---:|
| 1 petición por episodio (secuencial) | 21 + 21 | ~23 h |
| 1 petición por episodio, 4 en vuelo | — (×1,5) | ~18 h |
| Lote de 12, prosa | 6,0 | — |
| Lote de 12, profesor | 7,2 | — |
| **Lote de 16, prosa + profesor, 2 en vuelo** | **~13** | **~6 h** |

Lo que el lote NO cambia es qué se acepta: `_teacher_verdict` es la
misma regla para el camino de uno y el de muchos (id válido + evidencia
literalmente presente en los hechos), un item que falte en la respuesta
sale como rechazo con motivo y no como hueco desplazado, y la prosa de
Qwen sigue descartándose entera si pierde un token crítico de la regla.
`test_batch_12_publishes_exactly_what_batch_1_publishes` lo fija.

## Done when

- Las cinco familias producen episodios en ES y EN, con evidencia, y el reparto
  por familia e idioma está declarado antes de generar.
- El gold numérico sale de una regla determinista, reproducible por semilla.
- Existe un piloto publicado (≥2.000 episodios) con su manifest, semilla y
  versión de generador, consumible por un entreno sin conversión manual.

## Estado 2026-09-27 — piloto EN VUELO, volumen sin firmar

`status` sigue `active` a propósito. Tres de las cuatro casillas del gate
están medidas en verde desde `fdfa2a7`; `volume_ge_2000` **no se firma**
porque el piloto todavía no existe: está generándose en el job canónico
`datagen` (`artifacts/episodes-qwen/pilot-2k`, semilla 20260926, 2 100
episodios = los 2 000 exigidos más un 5 % de holgura). Ritmo medido
**11,0 s/episodio** → ETA ~6,5 h. No hay cifra que firmar hasta que el
job termine y `_volume_check` lea su manifest.

Lo que este turno sí cerró son los dos motivos por los que el piloto no
podía existir:

1. **Coste.** Una petición por episodio contra `qwen3.6:27b-mlx` cuesta
   ~19 s de sobrecarga fija frente a ~3 s de cómputo, y la concurrencia
   sólo daba ×1,5 → 18 h. Agrupando 16 episodios por petición: ~13 s de
   arranque y 11,0 s/episodio medidos → ~6,5 h. Commit `834d2a0`.
2. **Corrección.** El primer tramo publicó 54 de 100: los 46 rechazos
   eran todos `evidence is not a fragment of the state`, con la prosa de
   Qwen llevándose por delante el tramo que el contrato exige literal —
   el camino `--prose qwen` no se había ejercitado nunca (el smoke de 25
   era `--prose local`). Y la familia de extracción caía al 100 % a
   prosa local por una comparación sensible a mayúsculas. Commit
   `a5e029b`; medido después: 20 publicados, 0 rechazos, 18 con prosa de
   Qwen.

Cuando el job termine, `datagen` firma el gate él mismo
(`data.episode_gen gate --dir … --job-id datagen`). Si el piloto se
quedara por debajo de 2 000, `volume_ge_2000` sale **`false` medido**, no
`null`: hay cifra, y dice que no llega.

## Resolution (2026-09-27)

**done.** Gate `artifacts/gates/T-episode-gen/gen.json`, `volume_ge_2000`: **measured, pass** — 2 092 publicados de 2 100 planificados (8 rechazos, todos `teacher failed: invalid choice/evidence`), `manifest_sha` `72f25d32…`, semilla 20260926, prosa Qwen en 1 618 y fallback local en 474 (`prose_origins`). Tiempo: primer tramo 01:3x → murió a 1 394 (07:37, ollama caído); retomado con `--resume` a las 12:43 y terminado a las 15:34 (`elapsed_s` 1 788,7 cubre sólo el tramo final del manifest). Comando: el `command` del job `datagen` en `.meshkore/public/jobs.yaml`.

Lo que no tapa: 118 estados arrastran el marcador `KEEP:` del prompt (fuga de prompt, 113 en extracción); `#T-episode-verify` los pone en cuarentena. El modelo local cambió después a `qwen3.8:27b-mlx` (default de `QWEN_MODEL`); este piloto se generó con `qwen3.6:27b-mlx`, y cada traza lleva ahora el campo `model`.
