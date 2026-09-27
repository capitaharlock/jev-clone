---
id: teacher-distill
title: Profesor externo — cuánto nos falta, medido, y datos propios para cerrarlo
status: next
owner: architect-master
modules:
  - eval
  - data
created: 2026-09-23
updated: 2026-09-27
---

> **Reactivada el 2026-09-27** con otro papel: Jev deja de ser «evidencia del
> release gate» y pasa a ser **la vara de medir y el profesor** del objetivo
> único (ver `docs/guia-un-solo-objetivo.md`). Prioridad P2. Sigue bloqueada
> por la credencial: `#T-teacher-auth` necesita del operador una key válida y
> el proveedor exacto. En cuanto exista: `#T-teacher-probe` publica la distancia
> a Jev sobre nuestra batería y sobre typed-decisions test, y
> `#T-jev-soft-targets` (`#data-flywheel`) cachea sus probabilidades como
> objetivos suaves y recompensa. `#T-teacher-kappa` queda en backlog: era del
> release gate, que ya no es objetivo.

> **A `backlog` el 2026-09-25.** El profesor externo (TypeSafe) sigue
> devolviendo 401 con la key aportada y el plan de recuperación (§8) es
> explícito: *no bloquear la recuperación por esa credencial*. El papel de
> «referencia de capacidad y generador verificable» lo asume **Qwen local**
> dentro de `#episodic-data` y `#honest-eval`. Esta iniciativa se reactiva
> cuando exista acceso válido y un protocolo comparable, y entonces sirve
> para lo único que no puede hacer Qwen: situar la distancia contra el
> sistema publicado.

# Profesor externo — cuánto nos falta, medido, y datos propios para cerrarlo

El operador ha aportado (2026-09-23) una **API key con saldo** contra un modelo
profesor. Eso desbloquea dos cosas que el proyecto lleva desde el principio sin
poder hacer, y que no son la misma:

1. **Una referencia externa medida.** Hoy la pregunta "¿estamos lejos del
   original?" sólo se puede contestar contra el umbral que escribimos nosotros
   (`unseen_accuracy_all_min = 0,50`), no contra otro sistema. Con la key se
   puntúa **el mismo corte unseen, las mismas filas, el mismo protocolo** con el
   profesor, y la distancia deja de ser una opinión.
2. **La evidencia ausente del release gate.** `teacher_cohen_kappa_min = 0,60` es
   uno de los 9 criterios fallados de `#T-release-gate`, y falla por la regla de
   §2 —evidencia ausente = criterio fallado—, no por desacuerdo medido: *no
   existe ningún artefacto que mida el acuerdo del checkpoint contra el
   profesor*. Con la key, existe.

Y abre una tercera, condicionada: si `#T-labelspace-div` da GO al eje de datos,
la fábrica sintética deja de estar limitada a lo que sabe generar de forma
programática y puede pedirle al profesor **espacios de etiquetas** que ninguna
plantilla nuestra produce.

Esta iniciativa es distinta de `#generalization-fix`: aquella busca *por qué* la
curva baja, tocando modelo y mezcla; ésta trae una referencia y un profesor de
fuera. Se cruzan en un punto: el veredicto de `#T-labelspace-div` decide si
`#T-teacher-labelspaces` se ejecuta o se queda en backlog.

Regla de gasto, desde la primera llamada: el saldo es finito y prestado. Toda
llamada va cacheada en disco por hash del prompt, todo gate publica el coste que
consumió, y ningún experimento re-puntúa lo ya puntuado.

## Done when

- El cliente del profesor está cableado, cacheado y con presupuesto declarado, y
  ningún experimento puede gastar saldo sin registrar cuánto gastó.
- Existe un número publicado de **el profesor sobre nuestro corte unseen**, fila
  a fila comparable con el nuestro: la distancia al estado del arte deja de ser
  una estimación.
- `artifacts/gates/T-teacher-kappa/gate.json` publica Cohen's kappa del
  checkpoint candidato contra el profesor, y `#T-release-gate` deja de fallar
  `teacher_cohen_kappa_min` por evidencia ausente.
- El camino de datos con profesor está decidido con el veredicto de
  `#T-labelspace-div` delante: GO y se generan espacios de etiquetas, NO-GO y se
  archiva sin gastar saldo.

## Estado 2026-09-24 — endpoint identificado, credencial rechazada

El proveedor es **TypeSafe**: `https://api.typesafe.ai` responde, el esquema es
`Authorization: Bearer`, y con esa cabecera la key aportada devuelve **401
"Cannot authenticate"** (con `x-api-key` ni se lee: 403 "Must supply an API
key"). URL y esquema correctos, credencial no válida. Detalle y siguientes
pasos en `#T-teacher-auth`, que pasa a ser la primera task de la cadena y
bloquea a `#T-teacher-probe` y `#T-teacher-kappa`.

Mientras no haya key válida, la referencia externa la da `#T-jev-parity`
(`#full-space-training`), que compara contra la cifra publicada del profesor
sin gastar saldo.
