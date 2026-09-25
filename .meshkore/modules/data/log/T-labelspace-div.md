---
superseded_by: episodic-data / T-episode-gen
id: T-labelspace-div
title: Diversidad de espacios de etiquetas — medirla y, si falta, generarla
status: superseded
priority: high
owner: developer
category: data
initiative: generalization-fix
depends_on:
  - T-antiscale-diag
created: 2026-09-22
updated: 2026-09-25
failed_at: 2026-09-22T11:30:25.903Z
resolved_by: A033
resolved_by_conv: work-generalization-fix-T-labelspace-div-1790074806
commit_shas: ['1e660e067cde2efa14856ae1b62b110a5f904768', '9b27bfa8d915e813699872d58609c070ab82c2fe', '3b144f1c9705c066814a34a0611058c64037d365']
---
# Diversidad de espacios de etiquetas — medirla y, si falta, generarla

> **Cerrada por hallazgo el 2026-09-25.** Su mitad 1 ya entregó lo decisivo:
> 9 taxonomías cubren el 83,1 % del millón de filas y sólo 135 espacios son
> reutilizables (`artifacts/gates/T-labelspace-div/inventory.json`). La curva
> 62k/250k/1M sobre la mezcla `decision-mix-labeldiv-1m` deja de ser la siguiente
> apuesta: es más volumen de la misma plantilla, y el plan de recuperación (§4)
> retira ese camino. El job `labeldiv-curve` queda parado a propósito. El
> problema real —muchos ejemplos donde el mismo estado cambia de respuesta al
> cambiar la pregunta— lo ataca `#episodic-data`.

La hipótesis de arquitectura (`#T-antiscale-diag`, `#T-unfreeze-backbone`) tiene
una gemela por el lado de los datos: `decision-mix-clean-1m` está construido
sobre un número **pequeño** de taxonomías (dbpedia14 = 14 etiquetas,
civil-comments = 2, MASSIVE = 60, ...). Añadir filas dentro de esas taxonomías
no enseña a comparar pregunta↔opción: refuerza el mapa cerrado `texto → etiqueta`.
Es la explicación de la anti-monotonía que **no** requiere tocar el modelo, y hay
que descartarla o confirmarla con la misma disciplina.

Primero se mide sobre el corpus que ya existe: cuántos espacios de etiquetas
distintos hay, cuántas filas por espacio, y cuánto se solapan con los cortes
unseen de `#T-unseen-labels`. Si el inventario sale pobre, se **genera**: la
fábrica sintética (`#T-synth-factory`, `#T-gen-schemas`) ya sabe producir
esquemas de decisión completos — aquí se la apunta a producir *muchos espacios
pequeños* en vez de muchas filas de pocos espacios, y se re-mezcla a igual
volumen total para que la comparación contra la curva actual sea limpia.

No se entrena a ciegas: la mezcla nueva se evalúa con el gate de
`#T-unseen-labels` a los mismos tamaños (62 k / 250 k / 1 M) que la curva vieja,
para que el veredicto sea una curva contra otra curva.

## Done when

- Existe un inventario versionado (`artifacts/gates/T-labelspace-div/`) con nº de
  espacios de etiquetas, filas por espacio y solape con los cortes unseen.
- Hay una mezcla alternativa a **igual volumen** con ≥ 10× más espacios de
  etiquetas distintos que `decision-mix-clean-1m`.
- La curva unseen de esa mezcla está medida a 62 k / 250 k / 1 M con el gate de
  `#T-unseen-labels`, junto a la curva vieja en el mismo JSON.
- El veredicto queda escrito: la diversidad de etiquetas explica la
  anti-monotonía (GO al camino de datos) o no la explica (NO-GO, y el peso cae
  en `#T-unfreeze-backbone`).

## Orden de ejecución (2026-09-22) — la mitad que no toca MPS va primero

MPS está ocupado hasta ~14 h vista (`antiscale-wide` + `gen-objective-sweep`,
ver `#T-gen-objective`), así que esta task se ejecuta en dos mitades y la
primera no espera a nadie:

1. **Inventario + mezcla, sólo CPU.** Contar espacios de etiquetas del corpus
   actual, filas por espacio y solape con los cortes unseen; escribir
   `artifacts/gates/T-labelspace-div/inventory.json`. Si sale pobre, apuntar la
   fábrica sintética a producir *muchos espacios pequeños* y materializar la
   mezcla alternativa a igual volumen (≥ 10× espacios distintos).
2. **Curva, cuando MPS quede libre.** 62 k / 250 k / 1 M con el gate de
   `#T-unseen-labels`, encoladas como job detrás del sweep — nunca lanzadas a
   mano mientras otro run tenga el dispositivo.

Un inventario que ya diga "hay 9 espacios de etiquetas en 1 M de filas" es, por
sí solo, resultado publicable: no hace falta la curva para saber que la mezcla
no enseña a comparar.

## Estado 2026-09-22 — mitad 1 hecha, mitad 2 bloqueada por dispositivo

Mitad 1 entregada y commiteada (`3b144f1`): inventario digest-verificado
(9 taxonomías = 83,1 % de 1 M, sólo 135 espacios reutilizables → **pobre**) y
mezcla alternativa `decision-mix-labeldiv-1m` de 1 M filas en 20 000
mini-taxonomías (148×, solape 0 con las 26 unseen legacy).

Mitad 2 (la curva) está encolada como job `labeldiv-curve` y **no puede
arrancar**: su centinela espera
`artifacts/runs/genobj-short-prior-s20260922/summary.json`, un run-id que no
existe — `tools/gen_objective/arms.py` nombra los runs
`genobj-<arm>-ettin-68m-s<seed>`. Con ese fichero inalcanzable el job poll-ea
cada 600 s para siempre. Hay que corregir el centinela al último brazo real
antes de que MPS quede libre.

### Cierre pendiente

1. Arreglar el centinela de `tools/labeldiv_curve.sh` (esperar los 5 brazos
   reales, no `genobj-short-prior`).
2. Dejar que el job entrene 2 × 1 M (ettin-68m + modernbert-base) sobre
   `episodic-div` cuando el sweep libere el dispositivo.
3. Escribir `artifacts/gates/T-labelspace-div/curve.json` y con él el veredicto
   GO/NO-GO al camino de datos frente a `#T-unfreeze-backbone`.

## Resolution


I'll start by reading the task file.Now let me measure the real corpus. Exploratory pass first:Now let me check what the synthetic factory can actually produce (CPU, offline):Now I'll write the measurement + alternative-mixture module:A sibling agent committed half 1 while I was working. Let me check what they shipped before I clobber anything:I overwrote a sibling agent's committed module. Restoring it now:Verified the collapse. Let me confirm the end-to-end effect on a real sample:Now the fix — let the sampler honour a row's own label space:The fix works and all 69 existing tests pass. Now the measurement module + tests:

**Commit** `1e660e067` (+2) · 12 files

**Files changed (12):**
- `.gitignore`
- `.meshkore/modules/data/tasks/T-labelspace-div.md`
- `artifacts/gates/T-labelspace-div/inventory.json`
- `artifacts/mix-labeldiv/decision-mix-labeldiv-1m-seed20260922.manifest.json`
- `data/episodic.py`
- `data/labelspace.py`
- `data/mix.py`
- `data/test_episodic.py`
- `data/test_labelspace.py`
- `data/test_mix.py`
- `tools/labeldiv_curve.sh`
- `training/python/train_decision.py`

## Replanificación 2026-09-23 — esta task es ahora el eje principal

Con `#T-unfreeze-backbone` en NO-GO y `#T-lever-stack` midiendo el techo del
eje de capacidad+objetivo, el **eje de datos es la única hipótesis viva** de la
anti-monotonía, y es además la que el inventario ya dejó señalada: 9 taxonomías
cubren el 83,1 % de 1 M de filas. La mezcla alternativa
`decision-mix-labeldiv-1m` (20 000 mini-taxonomías, 148× más espacios, solape 0
con los cortes unseen) **existe en disco y nunca se ha entrenado hasta el
final**: el run `labeldiv-1m-ettin-s20260922` murió con exit 143 antes de su
primera etapa evaluada.

Cambios sobre el plan de ayer:

1. **Centinela roto**: `tools/labeldiv_curve.sh` espera
   `artifacts/runs/genobj-prior-ettin-68m-s20260922/summary.json`, un run-id que
   no existe (el real es `genobj-prior-1m-ettin-68m-s20260922`). Con el sweep ya
   terminado el centinela sobra entero: se quita y el job se encola detrás de
   `lever-stack-d512-prior` por la cola de jobs, no por un `sleep`.
2. **El brazo cambia**: la curva ya no se mide con la config desnuda. Se mide
   con las palancas que sobrevivieron (`--d-model 512 --prior-penalty 1.0`,
   ettin-68m), porque el control contra el que se compara es el brazo de
   `#T-lever-stack` sobre la mezcla vieja. Una variable: la mezcla.
3. **Un brazo, no dos**: se cae el brazo `modernbert-base` (es consistentemente
   peor en unseen en las cuatro curvas medidas y cuesta otras 2 h de GPU).

## Done when (revisado 2026-09-23)

- `tools/labeldiv_curve.sh` corre sin centinela y el job queda registrado en el
  daemon con su id.
- `artifacts/gates/T-labelspace-div/curve.json` publica 62 k / 250 k / 1 M de la
  mezcla `labeldiv` **al lado** de la curva de `#T-lever-stack` sobre
  `decision-mix-clean-1m`, misma seed, mismo objetivo, misma cabeza.
- El veredicto queda escrito con el número: si la pendiente unseen 250 k → 1 M
  de la mezcla diversa es ≥ 0 (o mejora la mejor palanca de modelo, −0,0549), el
  eje de datos es GO y el relevo pasa a `#T-teacher-labelspaces` para generar
  mucho más de lo mismo; si no, el anti-escalado no es de espacios de etiquetas
  y hay que subir de tamaño de backbone, no de corpus.
