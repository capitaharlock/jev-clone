---
id: episodic-data
title: Episodios verificados y contrafactuales — datos que enseñan a decidir
status: active
owner: architect-master
modules:
  - data
created: 2026-09-25
updated: 2026-09-25
---

# Episodios verificados y contrafactuales — datos que enseñan a decidir

Segunda mitad del plan de recuperación (`.meshkore/docs/plan-recuperacion-2026-09-24.md`
§§4, 6). Sustituye la línea de «más filas / más taxonomías» de
`#data-training` y `#full-space-training`.

**El diagnóstico de datos.** Un millón de filas no son un millón de lecciones.
`data/episodic.py` genera 20.000 taxonomías de pseudopalabras con estructuras
`campo = valor`: buena prueba de recuperación literal, no enseña comparación,
negación, preferencias ni inferencia, y sus preguntas no llevan prosa natural.
`data/labelgen.py:297` toma las taxonomías de Qwen y construye las filas con
plantillas deterministas (`Field report: …` / `Cross-check against: …`):
diversidad de nombres de categoría no es diversidad de razonamiento, y el
marcador de línea regala un atajo. El inventario de `#T-labelspace-div` lo
cuantifica: 9 taxonomías cubren el 83,1 % del corpus.

**Lo que falta es una clase de ejemplo que hoy casi no existe:** muchos casos
donde **el mismo estado y las mismas opciones cambian de respuesta al cambiar la
pregunta**, y viceversa. Eso es exactamente lo que los checkpoints actuales
fallan.

**Lo que no hay que hacer:** extraer todo el conocimiento del mundo de Qwen. Los
atributos del catálogo viajan **en el estado** o se recuperan de una fuente; el
estudiante aprende a aplicar un criterio a esos hechos. Y no se paraleliza la
plantilla actual para llegar antes a 2.000 espacios: ese camino está retirado.

## Principios no negociables

- **Episodio completo, no fila:** `state`, pregunta natural, candidatos con ID
  opaco + texto, respuesta válida, evidencia, familia, idioma, origen,
  semilla/versión del generador y **grupo de variantes**.
- **Contrafactuales por familia:** cambiar sólo el hecho decisivo, sólo la
  pregunta, o sólo una descripción de opción. La respuesta cambia cuando
  corresponde — y **no** cambia ante una paráfrasis equivalente.
- **Gold por regla** en comparaciones numéricas (Qwen redacta, no altera los
  hechos); en decisiones semánticas, gold + evidencia, verificador **separado** y
  revisión manual estratificada. Un segundo pase del mismo modelo no es verdad
  independiente: los desacuerdos del primer piloto se descartan.
- **Cinco familias iniciales:** extracción/paráfrasis, comparación de atributos,
  clasificación por descripciones, inferencia y negación, decisiones con
  prioridades explícitas («prioriza duración; si empatan, menor precio»).
- **Nada de confianzas inventadas:** los porcentajes que escribe un LLM no son
  probabilidades calibradas. CE sobre gold validado; KL con temperatura sólo si
  aparecen distribuciones del profesor reproducibles y útiles.
- **Splits por familia, entidad, espacio y grupo de variantes** — nunca repartir
  las variantes de un mismo caso entre cortes.

## Done when

- El contrato de episodio está versionado con su validador, y todo episodio
  publicado lo cumple o es rechazado con motivo.
- Las cinco familias producen episodios en ES y EN con evidencia, y cada uno
  tiene al menos un contrafactual agrupado con él.
- El gold de comparación numérica sale de una regla, no de la opinión de un
  modelo; el semántico pasa verificador separado + muestra revisada a mano.
- El embudo está medido por separado: generado → aceptado → publicado →
  **consumido por un entreno** (la cifra que hoy nadie tiene).
- Los splits no reparten variantes del mismo grupo, y existe un corte privado
  que el profesor no ha producido ni ha visto durante la generación.
