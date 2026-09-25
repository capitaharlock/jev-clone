---
id: T-episode-contract
title: Contrato de episodio — el formato que hace auditable cada decisión
status: active
priority: high
owner: unassigned
category: data
initiative: episodic-data
created: 2026-09-25
updated: 2026-09-25
---

# Contrato de episodio — el formato que hace auditable cada decisión

Hoy una fila de entreno no permite responder «¿por qué esta respuesta es la
correcta?» ni «¿qué otra variante de este caso existe?». Sin eso no se puede
construir un contrafactual, ni un split que no filtre, ni una batería honesta.
Esta task escribe el contrato **antes** de generar nada, porque todo lo demás
—`#T-episode-gen`, `#T-battery-dev`, `#T-battery-sealed`— lo consume.

Campos obligatorios por episodio: `state` (hechos y atributos pertinentes,
incluidas las alternativas cuando la pregunta compara), `question` (prosa
natural, criterio explícito), `candidates[]` (**ID opaco** + texto/descripción),
`answer` (o conjunto de respuestas aceptables, o preferencia explícita cuando el
caso es ambiguo — no se impone una verdad arbitraria), `evidence` (el fragmento
del estado que la justifica), `family`, `lang`, `origin`, `generator_seed`,
`generator_version` y `variant_group`.

El `variant_group` es el campo que hace posible todo lo demás: agrupa un caso con
sus contrafactuales para que ninguna variante del mismo caso caiga en dos cortes
distintos.

Dos reglas del contrato que hay que escribir explícitamente:

1. **Contrato de contexto comparativo.** Si la pregunta compara opciones y la
   información decisiva vive dentro de cada opción, puntuar una sin acceso a las
   demás pierde el significado de «mejor». El contrato declara, por familia, si
   el estado lleva los datos de todos los candidatos o si el scorer recibe un
   contexto de candidatos idéntico en cada pasada.
2. **`canonical_question()` deja de valer como pregunta.** El corpus convertido
   usa IDs de dataset como pregunta; un episodio con eso en `question` es
   inválido, no degradado.

## Verification gate

- Test: un validador rechaza con motivo un episodio sin `evidence`, sin
  `variant_group`, con IDs de candidato no opacos, o con un ID de dataset por
  pregunta.
- Test: dos episodios del mismo `variant_group` nunca se asignan a cortes
  distintos por el repartidor de splits.
- El esquema queda versionado (`schema_version`) y su sha se registra en cada
  manifest de generación.

## Done when

- El esquema del episodio existe, está versionado y tiene validador ejecutable.
- El contrato de contexto comparativo está escrito por familia, no implícito.
- Los cinco nombres de familia están fijados como enumeración, no como texto
  libre.
- Un lote de ejemplo hecho a mano (≥20 episodios, ES y EN, con contrafactuales)
  pasa el validador y sirve de fixture para el resto de la iniciativa.
