---
id: T-gen-objective
title: Objetivo que premia comparar, no recordar
status: done
priority: high
owner: developer
category: model
initiative: generalization-fix
depends_on:
  - T-antiscale-diag
created: 2026-09-22
updated: 2026-09-22
resolved_by: A031
resolved_by_conv: mb-developer
commit_shas: ['3b144f1c9705c066814a34a0611058c64037d365', '14d6f6d439e727136b70d956b3b72a85f1d35916', '3aff8c77d7e63b97afcb381d243d15b4a98a3de2', 'e14a411cfb9bc7c3027e8ca7b152941b1fcbca2a']
completed_at: 2026-09-22T22:06:24.625Z
---
# Objetivo que premia comparar, no recordar

Si la ablación de `#T-antiscale-diag` confirma memorización del espacio de
etiquetas, descongelar el backbone sólo le da más capacidad para memorizar
mejor. Esta task ataca el incentivo, no la capacidad.

Candidatos a evaluar, cada uno como un run comparable sobre el mismo 1 M:

1. **Dropout de etiquetas por episodio**: en cada batch, sustituir una fracción
   de las opciones por texto no visto, de forma que el mapa memorizado nunca
   sea suficiente para acertar.
2. **Entreno episódico few-shot**: muestrear conjuntos de opciones distintos
   para la misma pregunta, con el gold siempre presente pero los distractores
   remuestreados — ya existe el sampler de `#T-optset-sampler`.
3. **Pérdida contrastiva sobre el texto de la opción**, en vez de (o junto a)
   la cross-entropy listwise, para que la señal sea pregunta↔opción y no
   pregunta→índice del espacio conocido.
4. **Penalización de frecuencia de etiqueta**: descontar del logit el prior
   empírico de esa cadena en train, para que una etiqueta frecuente no gane
   por serlo.

No se adoptan los cuatro: se mide cuál mueve la pendiente unseen y se queda
el que la mueva, con el resto registrado como descartado y por qué.

## Verification gate

- Cada candidato corre con la misma seed, mix y eval que el baseline.
- Test: con opciones nuevas no vistas en train, el modelo adoptado supera el
  azar (0,165) en todos los cortes de `#T-unseen-labels`.
- Test: `unseen_ranking` supera su propio azar (0,202) — no basta con acertar,
  el orden tiene que ser informativo.
- El gate escribe `artifacts/gates/T-gen-objective/gate.json` con el candidato
  adoptado, los descartados y el motivo numérico de cada descarte.

## Done when

- Un candidato adoptado con ganancia unseen medida y reproducible.
- Los descartados quedan registrados con su número, no borrados.
- `#T-release-gate` vuelve a evaluarse sobre el checkpoint resultante.

## Prioridad (#T-antiscale-diag)

**Primera de las dos.** La ablación de `#T-antiscale-diag`
(`artifacts/gates/T-antiscale-diag/gate.json`) atribuye el **75,4 %** de la
caída 250 k → 1 M al eje 4, *ranking*, no a la temperatura: el número forzando
decisión cae con el crudo y a 1 M no se distingue del azar. El eje 1 mide,
además, que el **100 %** de la mezcla 1 M se contesta con un mapa
texto→etiqueta, así que cada fila nueva refuerza exactamente el incentivo que
esta task ataca. El orden no depende de cuál de los dos ejes de la partición
gane: bajo el protocolo completo de `#T-unseen-labels` la caída de
modernbert-base es casi toda abstención, y con el backbone congelado el logit
de `unknown` es igualmente algo que la cabeza aprendió bajo este objetivo. Se
invierte con `#T-unfreeze-backbone` si el eje 3 (cabeza ×2/×4, job
`antiscale-wide`) aplana la pendiente.

## Estado 2026-09-22 — código listo, medición en cola

El código de los cuatro candidatos está en `main` (`e14a411`): cuatro flags
componibles en `training/python/train_decision.py` con 24 tests verdes. Lo que
falta es exclusivamente la MEDICIÓN, y depende de un recurso ocupado, no de
una decisión: MPS está tomado por el job `antiscale-wide` (eje 3 de
`#T-antiscale-diag`, dos runs de 1 M, ~6,2 h cada uno al ritmo medido de
62 k filas / 1 390 s).

El job `gen-objective-sweep` ya está vivo y espera a que aparezca
`artifacts/runs/antiscale-wide-d1024-modernbert-s20260922/summary.json`;
entonces corre los cinco runs cortos de 62,5 k (base + los cuatro candidatos,
misma seed 20260922, mismo mix `--fence-clean`). Horizonte ≈ 14 h desde las
10:34Z.

Al cerrar el sweep queda por hacer, en este orden:

1. `artifacts/gates/T-gen-objective/gate.json` con el adoptado y el número de
   descarte de cada uno de los otros tres.
2. Run largo a 1 M del candidato adoptado, para que la ganancia se mida en el
   punto donde la pendiente es negativa hoy.
3. Re-evaluar `#T-release-gate` sobre el checkpoint resultante.

Esta task no se desbloquea a mano: se desbloquea cuando el sweep escribe sus
cinco `summary.json`.

## Estado 2026-09-22 — sweep cerrado a 125k, gate escrito

Los cinco brazos terminaron en CPU (ettin-68m, 125 056 decisiones, seed
20260922, `artifacts/runs/genobj-<arm>-ettin-68m-s20260922/summary.json`) y el
gate está en `artifacts/gates/T-gen-objective/gate.json`:

- Adoptado: **prior-penalty** — unseen 0,350 (+53,4 % sobre baseline, 2,12x el
  azar 0,165), ranking 0,350 == accuracy, abstención 0,0.
- Descartados con número: episodic 0,246 (+7,9 %, a 0,104 del ganador),
  contrastive 0,202 (−11,4 % vs baseline), label-dropout 0,169
  (indistinguible del azar, CI95 [0,156, 0,183]).
- Caveat explícito en el gate (`antiscale_delta`): medido a **125k, no a 1M**;
  la caída conocida 0,239 → 0,050 (250k→1M) queda fuera del rango medido y
  nada se extrapola.

Siguiente paso escrito (no entreno nuevo en esta task): run largo a 1M del
brazo prior y re-evaluar `#T-release-gate` sobre ese checkpoint. La GPU sigue
ocupada por `antiscale-wide`; ese run se encola como job, nunca a mano.

## Resolution

Listo: prompt copiable abajo, con números verificados y dudas marcadas. No he commiteado nada ni tocado GPU.

<details><summary>Prompt para el agente — copiar desde aquí</summary>

```
Ejecuta como general coder: (1) commitea el Work in Progress pendiente y (2) da el siguiente paso contra el anti-escalado. Lee este prompt entero antes de tocar nada.

1. EL PROBLEMA (qué está roto)
- El release gate da NO-GO: artifacts/gates/T-release-gate/gate.json, verdict NO-GO (9 criterios fallados en la última lectura; 10/12 en la medición original).
- Curva anti-monótona con backbone congelado: a más filas de entreno, peor accuracy unseen, con seen estable ~0.60-0.66. Medido en artifacts/gates/T-mix-5m/gate.json (250k->1M: ettin-68m 0.239->0.050, modernbert-base 0.194->0.028).
- Sweep de objetivo #T-gen-objective (5 brazos x 125k, device=cpu, azar unseen 0.165): prior 0.350 | episodic 0.246 | baseline 0.228 | contrastive 0.202 | label-dropout 0.169. Adoptado: prior-penalty (gate en artifacts/gates/T-gen-objective/gate.json).
- PERO el run prior-1M (artifacts/runs/genobj-prior-1m-ettin-68m-s20260922/summary.json, finished_utc 2026-09-22T20:36:48Z) confirma que el prior-penalty sube el NIVEL pero NO corrige la PENDIENTE: unseen 0.311 (62k) -> 0.349 (125k) -> 0.321 (250k) -> 0.277 (500k) -> 0.235 (1M), azar 0.165. La caída 125k->1M persiste.
- Contexto de diagnóstico: #T-antiscale-diag (artifacts/gates/T-antiscale-diag/REPORT.md) dice que el eje ranking carga ~75% de la caída y que ~100% de la mezcla 1M se contesta con un mapa texto->etiqueta. #T-labelspace-div mitad 1 encontró inventario pobre (9 taxonomías cubren 83% del 1M) y dejó mezcla alternativa de 20k mini-taxonomías + curva encolada (job labeldiv-curve).

2. NIVEL DE CERTEZA (calibración exigida: duda lo dudoso, afirma lo seguro)
- AFIRMO (medido, con fichero): los 5 números del sweep a 125k; la curva prior-1M de arriba (5 puntos, fichero citado); el NO-GO del release gate; que el backbone estaba congelado en estos runs.
- DUDO (no afirmes sin medir): POR QUÉ persiste la pendiente (¿representación congelada? ¿fuga texto->etiqueta? ¿mezcla pobre? ¿otra causa?); SI descongelar el backbone la corrige; si la curva label-div cambiará el diagnóstico; el estado actual de los jobs de GPU (a tu llegada estaban parados/fallados, re-verifícalo con la API de jobs antes de lanzar nada pesado).

3. TAREA 1 — COMMIT DEL WORK IN PROGRESS (primero, antes de experimentar)
- `git status --porcelain` muestra ~60 ficheros modificados + ~60 untracked de otro agente (tareas .meshkore, crates/jev-model, data/, eval/, tools/data_factory, tools/gen_objective, gates, runs, más basura de build).
- Haz UN commit de todo el trabajo real pendiente: tracked modificados + untracked con valor (código, tasks, gates, summaries de runs). EXCLUYE siempre: target/, .DS_Store, checkpoints binarios sueltos, *.log sueltos. Verifica con `git check-ignore` ante la duda; NUNCA `git add -A`, NUNCA `git commit` a secas (índice git compartido entre agentes: commitea por pathspec explícito y comprueba `git diff --cached --stat` justo antes). NUNCA push. Mensaje con trailers MeshKore (Agent:/Model:/MeshKore:). Si un fichero tiene hunks ajenos a medias, aparta su bloque, commitea lo tuyo y restáuralo byte-idéntico.
- Done cuando: `git status --porcelain` solo muestra lo excluido (build/basura) y el commit existe en `git log --oneline -3`.

4. TAREA 2 — SIGUIENTE PASO (solo después del commit)
- Lee .meshkore/modules/model/tasks/T-unfreeze-backbone.md ENTERA antes de empezar: es el contrato y va DESPUÉS de #T-gen-objective por diseño.
- Siguiente paso: descongelar el backbone (LR pequeña, con el objetivo prior-penalty ya adoptado) y medir si la pendiente 125k->1M se corrige: corre la curva de escala y re-evalúa el release-gate. Mide, no maquilles: reporta unseen/seen por punto + azar, y el veredicto del gate.
- Restricciones: un solo experimento a la vez en GPU; si hay un job de entreno vivo, encola detrás, nunca en paralelo a mano;

…(truncated)
