---
id: T-episode-gen
title: Qwen genera episodios completos por familia, no plantillas rellenadas
status: active
priority: high
owner: unassigned
category: data
initiative: episodic-data
depends_on:
  - T-episode-contract
created: 2026-09-25
updated: 2026-09-27
---

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
