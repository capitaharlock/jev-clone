---
id: T-evidence-cadence
title: Auditar resultados cada 48 horas e investigar novedades cada semana
status: active
priority: high
owner: architect-master
category: eval
initiative: daily-learning-loop
depends_on:
  - T-loop-scoreboard
created: 2026-10-09
updated: 2026-10-09
---

# Auditoría e investigación recurrentes

El calendario canónico está en
`.meshkore/docs/research-cadence.md`.

## Próximas ejecuciones

- Salud operativa: diaria a las 18:00, `Europe/Madrid`.
- Auditoría interna: cada 48 horas desde 2026-10-09 a las 18:00.
- Investigación externa: cada viernes a las 19:00, mirando 14 días atrás.
- Revisión de publicación: primer viernes de cada mes.

## Criterios

- La auditoría compara checkpoints y cortes fijados; no abre el sellado.
- El holdout de reglas se informa, pero no decide el rumbo.
- JEv y cualquier referencia se etiquetan como `measured`, `cited` o
  `missing`.
- Cada hallazgo externo conserva URL, fecha, revisión, licencia y decisión.
- Un dataset no entra en entrenamiento por aparecer en una búsqueda.
- Publicar exige procedencia, licencia, hashes, card y reproducción.

## Done when

- Hay cuatro auditorías consecutivas con veredicto y evidencia.
- Hay cuatro investigaciones semanales con fuentes y decisiones.
- Al menos una revisión mensual decide qué publicar o explica por qué no.
- El calendario se convierte en jobs solo cuando cada informe pueda producirse
  sin abrir el corte sellado ni ingerir fuentes sin revisión.
