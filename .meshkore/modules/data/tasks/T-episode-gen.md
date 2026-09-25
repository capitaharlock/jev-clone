---
id: T-episode-gen
title: Qwen genera episodios completos por familia, no plantillas rellenadas
status: next
priority: high
owner: unassigned
category: data
initiative: episodic-data
depends_on:
  - T-episode-contract
created: 2026-09-25
updated: 2026-09-25
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

## Done when

- Las cinco familias producen episodios en ES y EN, con evidencia, y el reparto
  por familia e idioma está declarado antes de generar.
- El gold numérico sale de una regla determinista, reproducible por semilla.
- Existe un piloto publicado (≥2.000 episodios) con su manifest, semilla y
  versión de generador, consumible por un entreno sin conversión manual.
