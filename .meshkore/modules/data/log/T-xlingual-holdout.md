---
superseded_by: honest-eval / T-battery-dev
id: T-xlingual-holdout
title: El corte cross-lingual falla por construcción — retener it-IT/pt-PT en el entreno
status: superseded
priority: high
owner: unassigned
category: data
initiative: generalization-fix
depends_on:
  - T-labelspace-div
created: 2026-09-23
updated: 2026-09-25
---

# El corte cross-lingual falla por construcción

> **Absorbida el 2026-09-25.** El corte cross-lingual deja de arreglarse
> retocando el holdout del corpus actual: ES y EN son **ejes de primera clase de
> la batería privada** (`#T-battery-dev`, `#T-battery-sealed`), con n publicado
> por cruce familia × idioma × K, y el criterio de release prohíbe una media que
> pase destruyendo un idioma.

`eval.unseen gate` falla `cross_lingual_holdout` en **todos** los checkpoints
medidos hasta hoy — los cuatro brazos del anti-escalado, los cinco del sweep de
objetivo, los dos de backbone descongelado. No es una propiedad del modelo: el
criterio pide que it-IT y pt-PT de MASSIVE no se hayan visto en entrenamiento, y
ningún run de este repo los ha excluido nunca. El gate está midiendo memorización
translingüe y llamándola holdout.

Mientras esto no se corrija, ese criterio es **ruido constante**: no puede pasar,
así que no informa, y arrastra el veredicto de `#T-unseen-labels` a FAIL con
independencia de lo que haga la palanca que se esté midiendo. Eso contamina la
lectura de cualquier brazo nuevo.

El arreglo es del lado de la mezcla, no del gate: el criterio está bien escrito
(`.meshkore/docs/release-criteria.md` §3.1) y no se toca. Lo que se toca es
`data/mix.py`, para que la receta pueda declarar locales excluidos y el manifest
lo registre, de forma que el gate pueda verificar la exclusión contra el corpus
en vez de confiar en ella.

## Done when

- `data/mix.py` acepta una exclusión de locales por receta y la escribe en el
  `manifest.json` de la mezcla (locales excluidos + nº de filas descartadas).
- Existe una mezcla a igual volumen sin it-IT/pt-PT, y un run entrenado sobre
  ella con el mejor brazo vigente del momento.
- `eval.unseen gate` sobre ese checkpoint verifica la exclusión contra el
  manifest y publica `cross_lingual_holdout` con un veredicto **real** —pase o
  falle— en vez de fallar por construcción.
- Queda escrito en el gate si el modelo transfiere a un idioma no visto o no: es
  la primera medición honesta de esa propiedad en el proyecto.
