---
id: T-jev-soft-targets
title: Jev sobre nuestro train — probabilidades cacheadas como objetivo suave y recompensa
status: backlog
priority: medium
owner: unassigned
category: data
initiative: data-flywheel
depends_on:
  - T-teacher-probe
  - T-episode-splits
created: 2026-09-27
updated: 2026-09-27
---

# Jev sobre nuestro train — probabilidades cacheadas como objetivo suave y recompensa

## Contexto

El operador tiene cuenta en Jev. Cuando `#T-teacher-auth` tenga una key válida
y `#T-teacher-probe` haya demostrado que nuestro protocolo reproduce la cifra
publicada de Jev (≈ 0,73 en typed-decisions test), Jev puede hacer por
nosotros lo que Laya hace con su «profesor»: dar **distribuciones** sobre los
candidatos de cada episodio de train. Dos usos, los dos en `#daily-learning-loop`:

- **CE suave / KL con temperatura** sobre las probabilidades de Jev (Laya:
  «soft CE weight 1.0 encima de la recompensa»).
- **Recompensa** para un objetivo estilo RLCD (`#T-loop-rl-jev`): regla de
  puntuación propia (log + esférica; RPS para ordinales) contra la
  distribución del profesor.

El principio del plan de recuperación sigue: las confianzas de un LLM no son
probabilidades calibradas por sí solas. Se guardan crudas y se decide su
peso midiendo en dev, no por fe.

## Qué hacer, paso a paso

1. Muestra estratificada de train (por familia × idioma × K), tamaño escrito
   antes de llamar, con **coste estimado en €** a partir del precio por 1k
   que `eval/teacher.py` ya gestiona. Tope duro declarado; el cliente para al
   llegar.
2. Llamadas por `eval/teacher.py::TeacherClient` (caché por hash: repetir
   cuesta 0). Guarda por episodio `teacher_soft: {id: p}`, `teacher_top`,
   `teacher_model`, `teacher_fingerprint`, en
   `artifacts/teacher/jev/<split>/<sha>.jsonl`, nunca dentro de
   `episodes.jsonl` (el dato original no se toca).
3. Acuerdo Jev ↔ gold por familia e idioma, con n e IC95 %: si en una familia
   Jev acierta menos que nuestra regla de gold, esa familia **no** usa
   objetivo suave (escríbelo en el gate).
4. Gate `artifacts/gates/T-jev-soft-targets/gate.json`: n cubierto, llamadas,
   coste real, acuerdo por familia, familias excluidas y por qué.

## Done when

- Existe la caché con cobertura y coste publicados, reproducible sin gastar.
- El acuerdo por familia está medido y decide dónde se usa el objetivo suave.
- Ninguna credencial ni respuesta cruda del profesor está en git.
