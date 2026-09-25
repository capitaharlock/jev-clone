---
closed_at: 2026-09-25
superseded_by: cross-encoder-pilot
id: generalization-fix
title: La generalización a etiquetas nuevas (fase 1 — régimen K≤8)
status: superseded
owner: architect-master
modules:
  - model
  - data
created: 2026-09-22
updated: 2026-09-25
---

> **Cerrada el 2026-09-25: el diagnóstico de fase 1 está hecho y su conclusión
> agota la iniciativa.** Las cuatro palancas que abría —objetivo
> (`#T-gen-objective`), capacidad de la cabeza y descongelado
> (`#T-unfreeze-backbone`, `#T-antiscale-diag`), apilado
> (`#T-lever-stack`) y diversidad de espacios (`#T-labelspace-div`)— están
> medidas y ninguna aplana la pendiente hasta algo distinguible del azar a
> cardinalidad real. El inventario de `#T-labelspace-div` deja además el
> hallazgo que decide el siguiente paso: 9 taxonomías cubren el 83,1 % del
> millón de filas, así que el corpus no contiene un millón de lecciones.
> La prueba de mecanismo del operador (§3 del plan de recuperación) descarta
> la explicación posicional: las puntuaciones siguen al texto de la opción,
> lo que falla es la sensibilidad al estado y a la pregunta.
> Continúa en `#cross-encoder-pilot` y `#episodic-data`.

# La generalización a etiquetas nuevas (fase 1 — régimen K≤8)

Work-stream de diagnóstico de la fase 1 (régimen K≤8). El sistema está
entrenado y el gate de release lo mide contra la propiedad que define al
producto —acertar sobre etiquetas nunca vistas— y da NO-GO:
`artifacts/gates/T-release-gate/gate.json`, 10 de 12 criterios por debajo del
umbral, accuracy unseen **0,065** frente a un objetivo de 0,50 y un azar de
**0,165**.

Lo informativo no es el nivel, es la **pendiente**. En la curva de
`artifacts/gates/T-mix-5m/gate.json`, con el mismo corpus, el mismo trainer y
los mismos evals, la accuracy unseen **cae** al añadir datos:

| filas | unseen acc (ettin-68m) | unseen acc (modernbert-base) |
|------:|------:|------:|
| 62 k | 0,141 | 0,080 |
| 250 k | 0,239 | 0,194 |
| 500 k | 0,108 | 0,132 |
| 1 M | 0,050 | 0,028 |

Mientras tanto la accuracy *dentro* del corpus es alta (civil-comments 0,94,
dbpedia14 0,98): en este régimen el sistema aprende el espacio de etiquetas
visto y ese mapa se afila con cada fila nueva, a costa de la propiedad que
queremos. Por eso `#T-mix-5m` registró NO-GO al escalado dentro de la fase 1.

Sospecha principal, a confirmar por medición y no por argumento: el backbone
está **congelado** (`backbone.frozen: true`, ModernBERT-base 149 M) y toda la
capacidad entrenable son ~2,9 M de cabeza pointer (d=256, 2 capas). Una cabeza
pequeña sobre representaciones fijas es exactamente la arquitectura que aprende
un `texto → etiqueta` cerrado en vez de una comparación pregunta↔opción.

Sus mediciones son el prerrequisito de `#oss-release` (en `backlog`) y de
`#T-release-gate`: nada se publica sin una cifra unseen que sostenga la promesa
del producto. Las palancas que esta iniciativa dejó medidas —cabeza d512,
objetivo prior, apilado— y sus veredictos valen dentro del régimen K≤8; la
continuación a cardinalidad completa es `#full-space-training`.

## Done when

- Existe una explicación **medida** (no argumentada) de por qué la accuracy
  unseen cae al escalar, con ablación por eje.
- La curva unseen deja de ser decreciente: a igual corpus, 1 M ≥ 250 k.
- Un checkpoint supera el azar (0,165) en **todos** los cortes unseen y lo
  registra `#T-unseen-labels`.
- `#T-release-gate` puede volver a evaluarse sin que el fallo sea estructural.
- El corpus de entreno contiene espacios de etiquetas **diversos** (no un puñado
  de taxonomías repetidas), medido y registrado por `#T-labelspace-div`.

## Estado 2026-09-23 — tres ejes medidos, dos palancas parciales, uno vivo

Los números de arriba son los del diagnóstico (2026-09-22) y se conservan como
punto de partida. Lo medido desde entonces, con el mismo corpus, la misma seed
y los mismos evals:

| brazo (250 k → 1 M, unseen) | 250 k | 1 M | pendiente |
|---|---:|---:|---:|
| congelado d256 (control) | 0,2394 | 0,0495 | −0,1899 |
| cabeza ×2 (d512) | 0,2387 | 0,1838 | **−0,0549** |
| objetivo prior-penalty | 0,3215 | 0,2347 | **−0,0868** |
| backbone `last-n=2` @1e-5 | 0,3281 | 0,2354 | −0,0927 · **NO-GO** |

Dos palancas parciales (−71 % y −54 % de la caída), ninguna la elimina, y el
backbone entrenable no aporta: `#T-unfreeze-backbone` cerró en NO-GO. El eje de
modelo está cerca de su techo y `#T-lever-stack` lo está midiendo ahora mismo
(las dos palancas apiladas, job `lever-stack-d512-prior`).

Queda **un eje sin medir**: el espacio de etiquetas. El inventario de
`#T-labelspace-div` ya dijo que 9 taxonomías cubren el 83,1 % del corpus y que
la mezcla alternativa de 20 000 mini-taxonomías existe en disco — pero su curva
nunca se llegó a entrenar (el run murió con exit 143). Esa task pasa de
`blocked` a `active` y es ahora la hipótesis principal; su job
(`labeldiv-curve-d512`) está encolado detrás del de `#T-lever-stack`.

Se añade `#T-xlingual-holdout`: el criterio `cross_lingual_holdout` falla en
**todos** los checkpoints medidos porque ningún run excluyó it-IT/pt-PT del
entreno. Es ruido constante en el veredicto de `#T-unseen-labels` y hay que
quitarlo de en medio para poder leer los brazos nuevos.

La referencia externa y el acuerdo con el profesor salen de esta iniciativa y
pasan a `#teacher-distill`, que nace con la API key que aportó el operador.

## Re-encuadre 2026-09-24 — el eje que faltaba no era ninguno de los tres

`eval.fullspace` (2026-09-23) midió por primera vez el régimen que el producto
promete: BANKING77 con sus **77 etiquetas**. El mejor checkpoint da **0,0123
con azar 0,0130**. Toda esta iniciativa —los cuatro brazos, la curva de
escalado, las dos palancas parciales— se midió con **K≤8**, el mismo régimen
con el que se entrenaba en fase 1: mide preferencia entre ocho candidatos, no
ranking de un espacio nuevo.

Ahí termina el alcance de esta iniciativa. La transición a cardinalidad
completa está documentada en `.meshkore/docs/fase-2-espacio-completo.md` y se
ejecuta en `#full-space-training`, que se lleva `#T-bigk-optsets`. Los
veredictos de aquí valen en su régimen y no se heredan allí (regla R4).

Lo que sigue vivo en esta iniciativa: `#T-labelspace-div` (su curva está
corriendo, job `labeldiv-curve-d512`) y `#T-xlingual-holdout`, porque el corte
cross-lingual falla por construcción en cualquier objetivo. `#T-lever-stack`
cierra: su resultado (0,288 unseen @1M) queda como el mejor registro de fase 1.
