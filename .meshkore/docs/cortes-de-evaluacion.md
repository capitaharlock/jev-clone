---
title: "Cortes de evaluación — qué contesta cada uno y cuál manda"
updated: 2026-09-24
owner: eval
status: vigente
task: T-eval-cardinality
---

# Cortes de evaluación

> Este repo mide el mismo checkpoint en cuatro cortes distintos con cuatro
> `n` distintos. Ninguno está mal; contestan preguntas distintas. Lo que
> faltaba era decir **cuál contesta cuál** y **cuál manda cuando discrepan**,
> porque sin eso el número que se citaba era el que tuviera a mano quien
> escribiera el informe.

## La jerarquía, en una línea

**Métrica primaria:** accuracy a **cardinalidad completa** sobre etiquetas que
el run no entrenó, con su azar, su `n`, sus aciertos y su IC 95 % al lado.
Todo lo demás es **diagnóstico** y no encabeza un informe (reglas R1 y R2 de
`fase-2-espacio-completo.md` §7).

Un solo comando la produce, con los diagnósticos debajo:

```
.venv-train/bin/python -m eval.scoreboard show --checkpoint <ckpt>
```

## Los cortes

| corte | qué contesta | n | K | dónde |
|---|---|---:|---:|---|
| **primaria · dev** | ¿puntúa el head el espacio completo de etiquetas que no vio? Es donde se eligen los brazos | 1 000 | 77 | `eval.scoreboard --cut dev` |
| **primaria · test (reservado)** | la misma pregunta sobre las filas en las que el mundo publica sus cifras | 3 080 | 77 | `eval.scoreboard --cut test --reason …`, `eval.fullspace` |
| barrido de K | diagnóstico exploratorio: la misma pregunta a K=5/8/20/40/77 | 1 000 | 5-77 | `artifacts/gates/T-teacher-probe/fullspace.json` |
| `eval.unseen` | diagnóstico: ¿señala una opción cuya ETIQUETA no entrenó, en la K en la que se entrenó? | 5 624 (unseen) / 9 646 (seen) | media 5,49 | `artifacts/gates/T-unseen-labels/gate.json` |
| stage eval del trainer | diagnóstico: ¿sigue aprendiendo el run, etapa a etapa? | 3 008 por lado | media 4,37 seen / 5,52 unseen | bloque `metrics` de cada manifest de checkpoint |

Cifras del checkpoint de referencia
(`leverstack-d512-prior-ettin-68m-s20260922/stage-001000000`), cada una con su
azar, como manda R2:

| corte | aciertos | accuracy | IC 95 % | azar | lee |
|---|---:|---:|---:|---:|---|
| primaria · dev (K=77) | 9 / 1 000 | 0,0090 | [0,0047, 0,0170] | 0,0130 | indistinguible del azar |
| primaria · test (K=77) | 38 / 3 080 | 0,0123 | [0,0090, 0,0169] | 0,0130 | indistinguible del azar |
| `eval.unseen` (K media 5,49) | — | 0,3133 | [0,3013, 0,3255] | 0,1662 | bate su azar **en su K** |
| trainer unseen (K media 5,52) | — | 0,2882 | [0,2723, 0,3047] | 0,1652 | bate su azar **en su K** |

Las dos últimas no contradicen a las dos primeras: **miden otra tarea**. Un
número a K≈5 y uno a K=77 no son el mismo experimento aunque compartan
checkpoint, y juntarlos en una frase es el error que esta página existe para
impedir.

## Por qué los `n` no coinciden

- **1 000 / 3 080** — el corte primario es *una fila, un espacio de etiquetas
  completo*. El de desarrollo es una muestra sembrada de 1 000 filas del split
  de train de BANKING77; el reservado son las 3 080 filas oficiales de test.
- **5 624** — `eval.unseen` puntúa varios datasets (banking77, huffpost,
  massive, boolq) y **por etiqueta retenida**: su `n` es la suma de los cortes
  unseen de todos ellos, no de uno. Además parte las filas en metric /
  calibration por grupo sembrado, así que ni siquiera usa todas.
- **3 008** — el stage eval del trainer corta en `--eval-samples`: no es un
  held-out estable, es una muestra del mismo régimen de entreno para ver la
  curva mientras el run avanza.
- **21 694** — el `n_scored` del gate de `eval.unseen` es el total puntuado
  entre todos los cortes (seen + unseen + eval-only). No es ninguna de las
  cifras de arriba y no se cita como tal.

## Quién manda cuando discrepan

1. **La primaria manda sobre cualquier diagnóstico.** Es la tarea que el
   producto promete: el espacio entero de etiquetas, sin que nadie preseleccione
   ocho candidatos.
2. **Entre dev y test manda test** — pero sólo se lee cuando hay algo que
   reportar, y cada lectura se registra (regla R7, abajo).
3. **Un diagnóstico nunca invalida a la primaria**; la explica. Que
   `eval.unseen` dé 0,3133 sobre su azar de 0,1662 y la primaria dé 0,0090
   sobre 0,0130 no es una contradicción: es exactamente el fracaso de
   transferencia que abre la fase 2.
4. **Ningún GO/NO-GO cruza de régimen** (R4). Cada checkpoint publica ahora su
   `cardinality_regime` en `run.json` y en su manifest; un veredicto medido a
   K≤8 se marca como no transferible, no se hereda.

## Desarrollo y test (regla R7)

`eval/cuts.py` define y sella los dos cortes:

- `artifacts/splits/T-eval-cardinality-banking77-dev/` — 1 000 filas del split
  **train**, semilla 20260924, selladas por sha256. Aquí se eligen K, el texto
  de las opciones, el objetivo, el encoder, la mezcla y la semilla, tantas veces
  como haga falta.
- `artifacts/splits/T-eval-cardinality-banking77-test/` — las 3 080 filas
  oficiales de **test**, **reservadas**. Ninguna fila cae en los dos cortes.

Cada lectura del corte reservado se escribe en
`artifacts/gates/T-eval-cardinality/test-queries.json` con fecha, checkpoint y
motivo. El registro arranca documentando las **siete** consultas que ya se
habían hecho antes de que existiera (las de `eval.unseen` del 21, 22 y 23 de
septiembre, la submuestra del teacher probe y las dos de `eval.fullspace`),
marcadas como reconstruidas desde el artefacto: su motivo se lee de la task que
las produjo, no de una nota escrita en su momento. `eval.fullspace` y
`eval.scoreboard --cut test` registran la suya solas; `--cut test` además exige
`--reason`.

El registro no da ni quita permiso —no puede— y tampoco compensa las lecturas
ya hechas: hace visible la selección adaptativa a quien lea los números después.

Una task puede llevar además su propio registro dentro de su gate —
`#T-option-text` lo hace en `artifacts/gates/T-option-text/test-queries.json`
— pero **nunca en vez del de aquí**: escribe en los dos, porque un registro
por task que el recuento del repo no viera haría parecer el corte reservado
menos consultado de lo que está. A 2026-09-24 el registro central lleva 9
consultas: las 7 reconstruidas, la republicación de `eval.fullspace` bajo la
semántica corregida de `beats_chance`, y la de `#T-option-text` — una sola
lectura, con los dos brazos dentro.

## Lo que el gate rechaza desde ahora

`eval/gate_rules.py` añade la regla **C4**: cualquier artefacto que publique una
`accuracy` sin `chance`, sin cardinalidad (`cardinality` o `mean_k`) y sin IC
95 % al lado —en su propio objeto o en alguno que lo contenga— falla, sea verde
o rojo. Se exime la cita ajena que se declara como tal (`citation: true`, o
`who` + `source`), que ya lleva su propio protocolo por R3.

En el árbol de hoy la regla marca tres gates más de los que ya estaban marcados,
entre ellos `T-train-real`, cuyo desglose por dataset publica accuracies sin su
K. El artefacto no se reescribe —un número publicado es parte del historial— y
el código sí: `compose_gate` emite ya `mean_k` en ese bloque.

## Diagnósticos que antes no existían

El scoreboard publica, sobre el mismo forward pass que la primaria:

- **frecuencia de predicción por etiqueta**, contra la frecuencia de gold;
- **matriz de confusión** dispersa, con los pares que dominan;
- **muestra de errores** repartida por el corte, con el rango del gold.

Sobre el checkpoint de referencia dicen algo que ninguna accuracy decía: el head
sólo llega a **26 de las 77 etiquetas** y contesta `lost_or_stolen_phone` en el
**33,9 %** de las filas, donde lo uniforme sería 1,3 %. No es un colapso sobre
una constante ni un error semántico repartido: es un prior fuerte sobre una
parte del espacio. Qué lo causa no lo dice este corte — lo miden
`#T-option-text` y `#T-bigk-optsets`.

`#T-option-text` ya contestó su mitad (2026-09-24,
`artifacts/gates/T-option-text/optiontext.json`): darle al head los nombres
legibles y las 77 definiciones del profesor, o además hasta 24 ejemplos
etiquetados en el `STATE`, deja la cifra donde estaba — 0,0090 / 0,0140 /
0,0130 sobre 1 000 filas de desarrollo contra un azar de 0,012987, los tres
intervalos conteniendo el azar, y la ventaja del mejor brazo sin reproducir
en el corte reservado (0,0123 → 0,0120 sobre 3 080). El prior sobre 26
etiquetas no viene de lo que el modelo lee.
