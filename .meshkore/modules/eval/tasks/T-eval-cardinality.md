---
status: done
id: T-eval-cardinality
title: Ordenar el testing — la cardinalidad completa pasa a métrica primaria
status: done
priority: high
owner: unassigned
category: eval
initiative: honest-eval
created: 2026-09-24
updated: 2026-09-24
completed_at: 2026-09-24T11:58:04.653Z
resolved_by: A003
resolved_by_conv: roadmap-architect-uwgjq
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

## Resolution

✓ #honest-eval #T-eval-cardinality **done** · commit `320e572` en HEAD · 19 ficheros · +74 tests (el `no-commit` del wake es falso, otra vez).

Y trae el dato que ordena la fase 2: **solo 26 de las 77 etiquetas se predicen alguna vez**. La cifra a cardinalidad completa es 0,0090 con azar 0,0130 — el modelo no está fallando por poco, está colapsado sobre un tercio del espacio. Ya no hay tres cortes contradictorios (3008/5624/3080): un scoreboard, corte de desarrollo congelado y test reservado con registro de consultas.

🚀 Etapa 2 lanzada · `developer-copy` (opus, pid **52356**, vivo verificado con `ps`) → #full-space-training #T-option-text
· Mide si la distancia con el profesor (0,924) la explica **la entrada**, no el objetivo: tres brazos a K=77 sobre el checkpoint que ya existe — identificador crudo vs. descripción legible vs. descripción + ejemplos. No entrena nada.

Un carril, y es lo correcto: #T-bigk-optsets cuelga de esta task en disco, y hay un job de MPS vivo (`labeldiv-curve-d512`).

<details><summary>Por qué no aplico el veredicto `blocked` — y por qué `partial` sí era honesto</summary>

- El wake dice `no-commit`; `git log` dice `320e572` en HEAD con los 19 ficheros del informe. Sexta vez esta sesión que el veredicto llega antes que el commit.
- A034 marcó `outcome: partial` porque la suite global queda roja: 857 pasan / 7 fallan / 5 skip. Midió el baseline antes de tocar nada: **783/7/5, los mismos 7 nombres**. Los fallos son de `#T-release-gate` (criteria_sha + números del gate), `#T-repo-clean` (índice con `target/`), antiscale, gen_objective y OOD de data_eval — ninguno en código de esta task, y arreglarlos pisa ficheros de otras dos.
- Los 7 Done-when están entregados y verificados con artefacto en disco: `artifacts/gates/T-eval-cardinality/scoreboard.json` + `test-queries.json`, manifests de ambos cortes, `beats_chance` arreglado y obligatorio en `eval/gate_rules.py`.
- Marcarla `blocked` por fallos ajenos sería congelar la fase 2 entera sobre deuda de otra iniciativa.
</details>

<details><summary>Cola real de #full-space-training — cadena estricta, no paralelo</summary>

```
T-eval-cardinality  done ──> T-option-text  (LANZADA, A036)
                                  └──> T-bigk-optsets
                                          ├──> T-fullspace-objective
                                          └──> T-encoder-finetune (+ fullspace-objective)
                                  └──> T-jev-parity
T-labelspace-factory  next, sin deps — único candidato a segundo carril
```
No lo abro: consume el Qwen local y competiría por MPS con `labeldiv-curve-d512`. Medido en este cluster, el segundo carril de cómputo compra +5 % de throughput a cambio de 4,6× de latencia por job.

Sigue en pie la discrepancia que te señalé: tu etapa 2 pedía #T-bigk-optsets ∥ #T-fullspace-objective; en disco #T-fullspace-objective depende de #T-bigk-optsets, y #T-bigk-optsets depende de #T-option-text. Ejecuto el orden del disco; si quieres el tuyo, el que reordena el roadmap es el Architect, no yo.
</details>

— #T-eval-cardinality · cardinalidad completa es ya la métrica primaria, un solo scoreboard y test reservado
— #T-option-text · lanzada a `developer-copy` (A036): tres brazos a K=77 para separar entrada de objetivo

1.8M tokens
