---
id: T-eval-cardinality
title: Ordenar el testing — la cardinalidad completa pasa a métrica primaria
status: active
priority: high
owner: unassigned
category: eval
initiative: honest-eval
created: 2026-09-24
updated: 2026-09-24
---

# Ordenar el testing — la cardinalidad completa pasa a métrica primaria

Primer movimiento de la fase 2 (`.meshkore/docs/fase-2-espacio-completo.md`):
**el eval se mueve antes que el entreno**. En fase 1 el eval compartía régimen
con el objetivo —`eval.unseen` medía con K≤8, que es como se entrenaba—, que es
lo coherente mientras el objetivo es K≤8 y también la razón por la que la
distancia real sólo aparece al medir a cardinalidad completa. Cambiar el
objetivo sin mover primero la métrica dejaría el mismo punto ciego abierto.

Hoy hay **tres cortes distintos que se contradicen** y ningún sitio donde se
lean juntos: el stage eval del trainer (n=3 008), el gate de `eval.unseen`
(n=5 624, con banking77), y `eval.fullspace` (n=3 080, K=77). Un agente que
llega nuevo no sabe cuál manda.

## Qué se ordena

**1. Jerarquía explícita.** Una métrica primaria —accuracy a cardinalidad
completa sobre etiquetas no vistas—, y todo lo demás marcado como
**diagnóstico** en su propio artefacto. K≤8 no desaparece: deja de poder
encabezar un reporte.

**2. `beats_chance` obligatorio (regla R2) — y arreglado.** Ningún gate de este
repo puede emitir una accuracy sin su azar, su K y su IC 95 % al lado. Se añade
la comprobación a `eval/gate_rules.py` y **falla el gate** si faltan, igual que
falla hoy por evidencia ausente.

Además hay un defecto de semántica que se corrige aquí: `eval/fullspace.py:159`
decide `beats_chance` con `flo`, el límite inferior del **ranking forzado**
(`accuracy_options_only`, sin `unknown`). Un modelo que abstuviera en todas las
filas tendría `accuracy = 0` y podría publicar `beats_chance: true`. Hoy
`abstain_rate = 0` y el veredicto del artefacto no cambia, pero el campo miente
en cuanto haya abstención. Se separan `beats_chance` (con `lo`, la accuracy
real) y `ranking_beats_chance` (con `flo`), y el test lo fija.

**2-bis. El barrido de K, como diagnóstico exploratorio.** Sobre 1 000 filas
K=8 y K=40 dan límite inferior por encima del azar; K=5 (0,215, IC
[0,191, 0,242] contra 0,200) y K=20 no. La frase «la ventaja aguanta hasta K=40»
no sobrevive a los intervalos: el barrido se publica con conteos e IC y sin
narrativa de progresión.

**3. Un solo scoreboard.** Un comando, un checkpoint, una tabla: primaria,
diagnósticos, paridad externa (`#T-jev-parity`), profesor cuando exista
(`#T-teacher-kappa`), y coste. Es lo que se pega en el chat y lo que lee el
operador, en lugar de tres JSON que hay que cruzar a mano.

Por corte, el scoreboard lleva: hash del checkpoint, hash de datos y split, K,
n, aciertos, azar, IC 95 %, tasa de abstención, protocolo de ejemplos (R8),
seen/unseen, coste y régimen de entreno. **Ningún número aislado encabeza un
reporte.** Y con la tabla van tres diagnósticos que distinguen colapso de error
semántico, hoy inexistentes: frecuencia de predicción por etiqueta, matriz de
confusión y una muestra de errores.

**3-bis. Desarrollo y test, separados (regla R7).** `eval/fullspace.py:106` carga
el split `test`, y contra esas mismas 3 080 filas el plan quería elegir K, log-Q,
encoder, descripciones, mezcla y seeds: eso convierte el test en conjunto de
desarrollo por selección adaptativa aunque nunca entre en el gradiente. Se
congela un corte de desarrollo para elegir brazos, se reserva el corte final, y
se abre un registro de consultas al corte final —fecha, checkpoint, motivo— que
empieza documentando las que ya se hicieron.

**4. Los cortes contradictorios, reconciliados.** Documentar qué mide cada uno,
por qué difieren en n, y cuál es el canónico para cada pregunta — o unificarlos
si la diferencia no defiende nada.

**5. Régimen declarado en cada checkpoint.** El artefacto de un run dice con
qué cardinalidad se entrenó. Comparar un run K≤8 con uno de espacio completo
sin que el lector lo sepa invalida la comparación (regla R4).

**6. Higiene de suite.** La suite pytest del repo pasa en verde; se le añaden
los casos de este cambio (gate sin azar → falla; K=|espacio| → soportado) y se
mide cuánto tarda, para que nadie la salte por lenta.

## Done when

- `eval/gate_rules.py` rechaza cualquier artefacto con accuracy sin `chance`,
  `cardinality` e IC 95 %, y hay test que lo demuestra.
- `beats_chance` se calcula con la accuracy real y `ranking_beats_chance` con el
  ranking forzado, con un test que falla si vuelven a confundirse (caso:
  abstención total → `beats_chance: false`).
- Un comando produce el scoreboard completo de un checkpoint (primaria +
  diagnósticos + paridad + coste + frecuencia de predicciones + confusión) y su
  salida es la que se publica en chat.
- Existe un corte de desarrollo congelado, distinto del test final, y un
  registro de consultas al test final que incluye las ya realizadas.
- `eval.unseen` y el stage eval del trainer quedan etiquetados como
  diagnóstico, con su n y su K en el encabezado del artefacto.
- Documentado en `.meshkore/docs/` qué corte contesta qué pregunta y cuál
  manda en un desacuerdo.
- Todo checkpoint nuevo publica el régimen de cardinalidad con el que se
  entrenó.
