---
id: T-ce-confirm
title: Confirmación — segunda semilla y una única apertura del test sellado
status: next
priority: high
owner: unassigned
category: model
initiative: cross-encoder-pilot
depends_on:
  - T-loop-nightly
  - T-battery-sealed
created: 2026-09-25
updated: 2026-09-27
---

> **2026-09-27:** la dispara un hito H3/H4 del bucle (`docs/bucle-infinito.md` §5), no una task previa. El candidato es el `current` del bucle en ese momento; la segunda semilla repite su último ciclo con otra semilla. Después de abrir el sellado, prepara el siguiente con `data/battery_sealed.py` y el protocolo de `.meshkore/docs/sealed-rotation-protocol.md`. **No hay push** hasta que el operador lo diga.

# Confirmación — segunda semilla y una única apertura del test sellado

Paso 5 del piloto, y el único que puede producir una afirmación pública.

1. Elegir el candidato **con desarrollo**. El test final no participa en la
   elección.
2. Repetir ese candidato con **otra semilla**: una mejora que no sobrevive al
   cambio de semilla no es una mejora.
3. Abrir el test sellado **una sola vez** y publicar **todos** los cortes: por
   familia, por idioma, por K. Después de abrirlo, ese test pasa a evidencia
   histórica y `#honest-eval` prepara otro para la siguiente decisión final.

La meta del operador es **≥70 % macro entre familias en las preguntas
respondibles** de la batería representativa. Para afirmar «al menos 70 %» con
respaldo estadístico hace falta además que el **límite inferior del IC95 % sea
≥70 %** — la media sola no basta. Y una familia o un idioma flojos siguen siendo
una limitación aunque la media pase: se publican con su n, y los cruces críticos
se amplían antes de cualquier release.

El tamaño de la batería no permite estimaciones precisas en cada cruce familia ×
idioma × K. Publicar n siempre; no convertir un n pequeño en una conclusión.

## Verification gate

- Test: el runner rechaza una segunda apertura del mismo test sellado (el sha
  queda marcado como consumido).
- Test: el veredicto no puede escribirse sin accuracy forzada, accuracy con
  abstención, macro por familia, azar por K, IC95 % y n por cruce.
- La elección del candidato queda trazada a métricas de desarrollo, con fecha
  anterior a la apertura del test.

## Done when

- El candidato elegido está confirmado con una segunda semilla y la diferencia
  entre semillas está publicada.
- El test sellado se abrió una vez, todos sus cortes están publicados y su sha
  está marcado como consumido.
- El veredicto contra la meta del 70 % está escrito con su IC95 %, y las familias
  o idiomas por debajo figuran como limitación explícita.
