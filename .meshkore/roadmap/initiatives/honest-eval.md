---
id: honest-eval
title: Evaluación honesta — batería privada de decisiones y criterio de release
status: active
owner: architect-master
modules:
  - eval
created: 2026-09-21
updated: 2026-09-25
---

# Evaluación honesta — batería privada de decisiones y criterio de release

Esta iniciativa nació con los hallazgos C, F y G de
`.meshkore/docs/audit-2026-09-21.md` y **cumplió esa mitad**: hay split por
plantilla y dominio (`#T-split-domain`), métrica primaria sobre etiquetas no
vistas (`#T-unseen-labels`), cardinalidad completa como métrica primaria
(`#T-eval-cardinality`) y reglas de coherencia que impiden un verde con
`cohen_kappa ≈ 0`. Fue esta evaluación la que expuso el fallo, y por eso el
proyecto sabe hoy dónde está.

**Lo que se le añade el 2026-09-25** (plan de recuperación §§7, 8): la
evaluación mide bien un benchmark que no es el producto. BANKING77 a 77 vías se
conserva como transferencia difícil y como diagnóstico, pero sus 77 intenciones
no representan las preguntas binarias y las comparaciones de producto del
operador. **El 70 % tiene que medirse sobre esos usos concretos**, y hoy no
existe ningún corte que los represente.

## Lo que se construye

Una **batería privada** con contrato explícito: 400 casos de desarrollo y 600
finales **sellados**, las cinco familias de decisión, ES y EN, K=2/3/8, más un
conjunto diagnóstico separado a K=20/77. Mezcla y pesos definidos **antes** de
evaluar. Reglas que la hacen honesta:

- El test final se abre **una sola vez**; después pasa a evidencia histórica y se
  prepara otro para la próxima decisión final. No se reutiliza para elegir el
  siguiente brazo.
- **Protocolos de información equivalentes** entre modelos comparados: un único
  formato por modelo, elegido sin consultar el test final.
- Ranking y abstención se evalúan **por separado**, para que uno no tape el
  fallo del otro. Calibrar la abstención no arregla un ranking que sigue al azar:
  el 0,9 % forzado de `#T-fullspace-objective` tiene IC95 % [0,474 %, 1,702 %]
  contra un azar de 1,299 % — no acredita mejora.
- Los pesos softmax son **relativos a los candidatos ofrecidos**; no se publican
  como probabilidad absoluta de verdad.

## Deuda de narrativa que esta iniciativa paga

Las conclusiones automáticas escritas en gates ya publicados no pueden alimentar
el plan como hechos causales: `verdict.reading` de fullspace afirma que el
intervalo primario contiene el azar cuando sus números lo excluyen por debajo (el
que lo contiene es el de elección forzada), y el resumen de bigK dice haber
descartado la cardinalidad como causa cuando un negativo con una semilla y un
presupuesto sólo descarta el beneficio observado de ese brazo. Se corrigen.

## Done when

- La batería existe, está versionada, revisada, y su mezcla y pesos se fijaron
  antes de la primera medición.
- El test final está sellado con regla de apertura única y su sha registrado.
- Las métricas mínimas se publican juntas: accuracy forzada, accuracy con
  abstención, cobertura y precisión entre respondidas, macro por familia, azar
  por K, éxito conjunto en pares contrafactuales, invariancia a permutaciones,
  NLL/Brier y calibración — con n por cruce familia × idioma × K.
- Ningún gate del repo publica una lectura que sus propios números contradigan.
- El criterio de release está escrito, fechado y firmado **antes** de la medición
  que lo evalúa, y exige ≥70 % macro entre familias en preguntas respondibles con
  límite inferior del IC95 % ≥70 % para afirmarlo con respaldo estadístico.
