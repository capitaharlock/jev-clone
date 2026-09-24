---
title: "Postmortem — entrenamos cuatro días una tarea que no es la que medimos"
updated: 2026-09-24
owner: architect-master
status: vigente
---

# Postmortem — el objetivo de entreno estaba mal

> Este documento existe para que el error quede escrito y no se repita. No es
> un diagnóstico de hiperparámetros: es el registro de una **elección de
> objetivo** que nadie volvió a auditar en cuatro días de trabajo.

## 1. El número que lo destapa

`artifacts/gates/T-teacher-probe/fullspace.json` (2026-09-23), mejor checkpoint
del proyecto (`leverstack-d512-prior`, 1 M filas), sobre las **3 080 filas de
test de BANKING77 con sus 77 etiquetas presentes** — el régimen exacto en el que
el profesor publica 0,924:

| | accuracy | azar | veredicto |
|---|---:|---:|---|
| nuestro head, K=77 | **0,0123** | 0,0130 | por debajo del azar |
| Jev (profesor), K=77 | 0,924 | 0,0130 | ratio nuestro: **1,3 %** |

El barrido de cardinalidad del mismo artefacto, sobre las mismas filas: la
ventaja sobre azar aguanta hasta K=40 y se disuelve en K=77. Con K=5 damos
0,215 contra 0,200 de azar. **Lo que el modelo tiene con pocas opciones es una
preferencia débil, no un ranking del espacio.**

## 2. Qué se clonó y qué no

De Jev se clonó el **runtime**: formato, API, motor de inferencia, `encode_state`
separado de `decide`, caché de estado, gates. Eso funciona y se conserva.

De Jev **no se clonó el entrenamiento**, y fue deliberado. La auditoría del
2026-09-21 (`#decision-rebuild`) definió el producto como "puntuar **opciones
dinámicas** nunca vistas" y de ahí salió una arquitectura distinta a propósito:

| | Jev (el original) | jev-clone (lo que entrenamos) |
|---|---|---|
| encoder | reentrenado entero contra la tarea | **congelado** (`backbone.frozen: true`) |
| espacio de salida | el espacio de etiquetas **completo** | **3–8 opciones muestreadas** (`data/optset.py: k_min=3, k_max=8`) |
| pérdida | sobre el espacio entero | `cross_entropy` sobre esas K opciones |
| capacidad entrenable | 68 M–149 M | ~2,9–11 M de cabeza pointer |

Ninguna de esas cuatro filas es un bug. El bug es que **la de la derecha nunca
se midió contra la de la izquierda hasta el día 4**.

## 3. Por qué se puede caer POR DEBAJO del azar

No es una paradoja ni un error de medida. El azar es una distribución uniforme
sin sesgo; nuestro head sí tiene sesgo — hacia etiquetas frecuentes, cortas y
léxicamente cercanas al texto — aprendido sobre un corpus con 9 taxonomías que
cubren el 83,1 % de las filas. Un sesgo sistemático, evaluado en un espacio
donde ese sesgo ya no correlaciona con el gold, puntúa **peor** que tirar un
dado. No hay ningún suelo que lo impida.

## 4. Por qué más datos empeoraban la métrica

La curva anti-escalado (unseen cae de 0,239 @250 k a 0,050 @1 M con el backbone
congelado) dejó de ser un misterio con lo anterior delante. Un objetivo de 8
opciones premia aprender el **mapa cerrado** `texto → etiqueta` de las
taxonomías vistas, porque con 8 candidatos ese mapa basta para acertar. Cada
fila nueva afila ese mapa. El mapa es exactamente lo que no transfiere a un
espacio nuevo. **Escalar datos sobre un objetivo equivocado escala el error.**

## 5. Causa raíz — es de proceso, no de hiperparámetros

1. **Nunca existió una task que dijera "reproducir la receta de entreno de
   Jev".** El roadmap tenía `#T-pointer-head`, `#T-train-real`, `#T-mix-1m`…
   todas dentro de una arquitectura elegida el día 1 y jamás re-auditada.
2. **La métrica vivía en el mismo régimen que el objetivo.** `eval.unseen`
   medía con K≤8, que es como entrenábamos. Un eval que comparte el sesgo del
   entreno no puede detectar ese sesgo. La divergencia fue invisible cuatro
   días *por construcción del eval*, no por descuido.
3. **Se optimizó dentro de la elección, no sobre ella.** Cuatro brazos
   (cabeza ×2, prior-penalty, unfreeze last-n, apilado) movieron la pendiente
   ±0,05–0,19 — todos medidos con K≤8 y todos irrelevantes para K=77.
4. **La comparación externa llegó la última.** La primera cifra comparable con
   algo publicado por otro es del día 4. Debió ser el día 1, aunque fuera
   contra un baseline tonto.

## 6. Qué queda invalidado (y qué no)

**Invalidado — hay que rehacerlo bajo el objetivo nuevo:**

- El **NO-GO de `#T-unfreeze-backbone`**. Se midió con K≤8, donde descongelar
  no puede ayudar: con 8 candidatos la representación congelada ya basta. La
  conclusión "el backbone entrenable no aporta" **no es transferible** al
  objetivo de espacio completo, que es justamente el que exige que la
  representación cambie. Se reabre como `#T-encoder-finetune`.
- El **NO-GO al escalado de `#T-mix-5m`**. Dice "más datos empeoran", pero mide
  más datos *sobre el objetivo equivocado*. No dice nada sobre escalar bien.
- Toda la **curva anti-escalado** como guía de decisión. Se conserva como
  registro histórico, no como evidencia sobre el modelo correcto.
- Las cifras `unseen` **0,29 / 0,288** que se venían publicando: son con
  K≤8 y **no son comparables con ninguna cifra publicada por nadie**.

**Vivo — no lo toca este postmortem:**

- Todo el runtime Rust/Candle, el servidor, la cuantización, la paridad.
- El corpus: 1 M de filas limpias, con licencias fenced y firewall de
  benchmarks. Lo que cambia es **cómo se muestrean las opciones de cada fila**,
  no las filas.
- La ganancia Metal ×2,7 (`#T-metal-throughput`): hace viable reentrenar.
- El pointer head **como arquitectura**: es label-free por construcción, que
  es la propiedad correcta. Lo que estaba mal es con qué se entrenó.

## 7. Las reglas, a partir de hoy

Son de obligado cumplimiento para cualquier agente que toque entreno o eval.

- **R1 — Se entrena la tarea que se mide.** Si la métrica primaria es
  cardinalidad completa, la pérdida es sobre cardinalidad completa. Cualquier
  divergencia entre régimen de entreno y régimen de eval se declara en el gate.
- **R2 — Toda cifra se publica con su azar al lado** y con `beats_chance`.
  Una accuracy sin su azar no es un resultado, es un número.
- **R3 — Ninguna cifra se compara con el exterior sin el mismo protocolo**:
  mismas filas, misma cardinalidad, mismo split. Si es una cita y no una
  medición, se estampa `same_rows: false`.
- **R4 — Un GO/NO-GO sólo vale dentro del objetivo con el que se midió.** Al
  cambiar el objetivo, los veredictos anteriores se marcan como no
  transferibles en vez de heredarse.
- **R5 — Antes de escalar datos, demostrar la pendiente en pequeño.** No se
  lanza un run de 1 M para una hipótesis que no se haya visto moverse a 62 k.
- **R6 — Una referencia externa desde el primer día.** Ningún work-stream de
  modelo arranca sin una cifra ajena en la misma mesa, aunque sea un baseline.

## 8. Dónde se ejecuta el arreglo

`#full-space-training` — nace de este documento. No ajusta hiperparámetros:
cambia el objetivo, reabre el fine-tune del encoder, y hace de la cardinalidad
completa la métrica primaria en `#honest-eval`.
