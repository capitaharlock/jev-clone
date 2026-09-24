---
title: "Fase 2 del entreno — del régimen de opciones muestreadas al espacio completo"
updated: 2026-09-24
owner: architect-master
status: vigente
---

# Fase 2 — entrenar y medir sobre el espacio de etiquetas completo

> Documento de transición. Cierra la fase de validación (runtime, corpus y
> arquitectura, probados en un régimen de pocas opciones) y abre la fase de
> entreno a cardinalidad completa, con el ajuste del approach de testing que
> hace falta para conducirla.

## 1. Dónde estábamos — fase 1 y para qué servía

La fase 1 tenía un objetivo acotado: levantar el sistema entero end-to-end —
formato, API, motor de inferencia Candle, `encode_state` separado de `decide`,
caché de estado, cuantización, gates, corpus de 1 M de filas limpias con
licencias y firewall de benchmarks — y comprobar que la arquitectura del head
aprende de verdad. Para eso se entrenó y se midió en un régimen barato y
rápido: **3–8 opciones por fila** (`data/optset.py: k_min=3, k_max=8`), con el
backbone **congelado** para que cada iteración cupiera en un Mac.

Ese régimen hizo su trabajo. El sistema está construido y la arquitectura
aprende: dentro del corpus, 0,94–0,98; sobre etiquetas no vistas con K≤8, hasta
0,358 contra un azar de 0,200. Las cuatro palancas medidas (cabeza ×2,
prior-penalty, unfreeze last-n, apilado) y la ganancia Metal ×2,7 salen de ahí.

## 2. La medición que marca el cambio de fase

El plan siempre incluyó una referencia externa. La primera comparación con
protocolo idéntico al del profesor está en
`artifacts/gates/T-teacher-probe/fullspace.json` (2026-09-23): mejor checkpoint
(`leverstack-d512-prior`, 1 M filas) sobre las **3 080 filas de test de
BANKING77 con sus 77 etiquetas presentes** — el régimen exacto en el que el
profesor publica 0,924.

| | aciertos | accuracy | IC 95 % | azar | K |
|---|---:|---:|---:|---:|---:|
| nuestro head | 38 / 3 080 | 0,0123 | [0,0090, 0,0169] | 0,0130 | 77 |
| profesor (cifra publicada) | 2 846 / 3 080 | 0,924 | — | 0,0130 | 77 |

El intervalo **contiene el azar**: la lectura correcta es *indistinguible del
azar*, no «por debajo». La distinción importa, porque «por debajo» sugiere un
sesgo medido y lo único medido es que no hay señal.

El barrido de cardinalidad del mismo artefacto (1 000 filas) se lee como
diagnóstico exploratorio, no como progresión: K=8 y K=40 dan límite inferior por
encima del azar, K=5 (0,215 con IC [0,191, 0,242] contra 0,200) y K=20 no lo
dan. Decir «la ventaja aguanta hasta K=40» describe una monotonía que los
intervalos no sostienen.

Lectura: hay un fracaso de generalización al espacio completo, demostrado. La
hipótesis de trabajo es que lo que el modelo tiene en régimen de pocas opciones
es una **preferencia local** y no un ranking del espacio — que es exactamente lo
que el objetivo de fase 1 optimizaba. El producto promete el ranking, así que el
objetivo de entreno y la métrica primaria suben a cardinalidad completa.

## 3. Lo que la medición demuestra y lo que todavía no aísla

La cifra demuestra el fracaso de transferencia. **No** identifica su causa, y
ninguna parte de este plan puede darla por establecida. Los sospechosos, con su
estado, están tabulados en `revision-externa-2026-09-24.md` §5; los dos que
mandan el orden de trabajo son:

- **El denominador de la pérdida** (entrenar con 3–8 candidatos). Hipótesis
  principal, sin aislar: cambiar el conjunto de opciones cambia también las
  interacciones de la cabeza, que hace atención entre candidatos.
- **La información disponible en las opciones.** Al profesor se le entregan
  definiciones de las 77 categorías y 24 ejemplos etiquetados; a nuestro modelo,
  identificadores crudos (`card_arrival`, `data/adapters.py:146`). Es medible
  sobre el checkpoint que ya existe, sin entrenar nada, y por eso va primero
  (`#T-option-text`).

Que una preferencia sesgada *pueda* caer por debajo del azar sigue siendo cierto
—el azar es uniforme y el head no—, pero es un mecanismo posible, no un
resultado: el IC no lo distingue de la ausencia de señal.

## 4. Por qué más datos no mejoraban la métrica

La curva de escalado (unseen 0,239 @250 k → 0,050 @1 M con backbone congelado)
encaja con lo anterior. Con 8 candidatos, acertar sólo exige descartar 7 cosas,
y el camino más corto es el mapa cerrado `texto → etiqueta` de las taxonomías
vistas. Cada fila nueva afila ese mapa, y el mapa no transfiere a un espacio
nuevo. **Escalar datos amplifica el régimen en el que se entrena**: con el
objetivo de fase 2, esa misma curva hay que volver a trazarla desde cero.

## 5. Qué distingue las dos fases

| | fase 1 (validación) | fase 2 (espacio completo) |
|---|---|---|
| encoder | congelado (`backbone.frozen: true`) | entrenable — se vuelve a medir cuánto |
| espacio de salida | 3–8 opciones muestreadas | el espacio de etiquetas completo |
| pérdida | `cross_entropy` sobre esas K | normalizada sobre el espacio: exacta cuando es enumerable, muestreada con log-Q cuando no |
| capacidad entrenable | ~2,9–11 M (head) | 68 M–149 M (encoder + head) |
| métrica primaria | unseen con K≤8 | accuracy a cardinalidad completa + azar |

La receta de fase 2 es la de recuperación densa (DPR / E5 / SetFit), que es la
que el profesor aplica de facto: normalizar la pérdida sobre el pool entero de
candidatos.

Dos precisiones que la primera redacción de este documento se saltaba, y que
ordenan el trabajo:

- **La vía exacta ya cabe en el código actual.** `cross_entropy` normaliza sobre
  las columnas que se le den: entregar a una fila las 77 etiquetas de su espacio
  ya normaliza sobre ese espacio. No hace falta escribir otra fórmula para la
  primera ablación — la extensión muestreada (negativos in-batch + log-Q) es un
  segundo paso, con su propia especificación.
- **El coste sí crece con K.** Cachear las claves de texto de etiqueta ahorra
  forwards del encoder, pero la cabeza hace `set_attn` entre todas las opciones
  (`model/decision_head.py:105-110`) y ese bloque crece ≈ K² por fila. El coste
  se mide a K=8/41/77 y cientos antes de comprometer un run, y los logits de
  conjuntos distintos no se unen como si fueran un solo softmax.

## 6. Qué se conserva y qué se vuelve a medir

**Se conserva íntegro:**

- Todo el runtime Rust/Candle, el servidor, la cuantización, la paridad.
- El corpus: 1 M de filas limpias, licencias fenced, firewall de benchmarks.
  Cambia **cómo se muestrean las opciones de cada fila**, no las filas.
- La ganancia Metal ×2,7 (`#T-metal-throughput`): es lo que hace viable
  reentrenar el encoder en local.
- El pointer head **como arquitectura**: es label-free por construcción, que es
  la propiedad correcta del producto.

**Se vuelve a medir bajo el objetivo nuevo** (no son veredictos anulados: son
veredictos de fase 1, válidos en su régimen y no transferibles al de fase 2 —
regla R4):

- El **NO-GO de `#T-unfreeze-backbone`**. Con K≤8 la representación congelada
  ya bastaba, así que la pérdida no tenía presión que transmitir al encoder. La
  pregunta se rehace en `#T-encoder-finetune`, donde sí la tiene.
- El **NO-GO al escalado de `#T-mix-5m`**. Mide escalado en régimen de fase 1.
- La **curva de escalado** completa: registro histórico de fase 1, no guía de
  decisión para fase 2.
- Las cifras `unseen` **0,29 / 0,288**: son con K≤8 y sólo se publican con su K
  al lado, nunca frente a una cifra externa medida a 77 vías.

## 7. El ajuste del approach de testing

El eval de fase 1 compartía régimen con el entreno (K≤8), que es lo coherente
mientras el objetivo es K≤8 y también la razón por la que la comparación con el
exterior exige cardinalidad completa. En fase 2 el eval se mueve primero, para
que la métrica fije el objetivo y no al revés. Seis reglas, de obligado
cumplimiento para cualquier agente que toque entreno o eval:

- **R1 — Se entrena la tarea que se mide.** Si la métrica primaria es
  cardinalidad completa, la pérdida es sobre cardinalidad completa. Cualquier
  divergencia entre régimen de entreno y régimen de eval se declara en el gate.
- **R2 — Toda cifra se publica con su azar al lado** y con `beats_chance`. Una
  accuracy sin su azar no es un resultado, es un número.
- **R3 — Ninguna cifra se compara con el exterior sin el mismo protocolo**:
  mismas filas, misma cardinalidad, mismo split. Si es una cita y no una
  medición, se estampa `same_rows: false`.
- **R4 — Un GO/NO-GO sólo vale dentro del régimen con el que se midió.** Al
  cambiar de régimen, los veredictos anteriores se marcan como no transferibles
  en vez de heredarse.
- **R5 — Antes de escalar datos, demostrar la pendiente en pequeño.** No se
  lanza un run de 1 M para una hipótesis que no se haya visto moverse a 62 k.
- **R6 — Una referencia externa en la mesa desde el primer día** de cada
  work-stream de modelo, aunque sea un baseline trivial.
- **R7 — Los brazos se eligen en desarrollo; el test oficial se reserva.**
  Elegir K, log-Q, encoder, descripciones, mezcla o seed mirando las mismas
  3 080 filas convierte el test en conjunto de desarrollo por selección
  adaptativa, aunque nunca entre en el gradiente. Cada consulta al corte final
  se registra con su fecha y su motivo.
- **R8 — La información que recibe el modelo es parte del protocolo.** Qué texto
  llevan las opciones, cuántos ejemplos van en el `STATE` y cuántos tokens
  sobreviven al tokenizador se declaran en el artefacto. Dos cifras con distinta
  información de entrada no son comparables aunque coincidan filas y K.
- **R9 — Un NO-GO barato limita gasto, no establece causa.** El brazo ganador se
  repite en otra seed antes de cualquier veredicto causal, y un brazo que cambia
  el scoring no se llama «misma arquitectura».

## 8. Dónde se ejecuta

`#full-space-training` — nace de este documento y no ajusta hiperparámetros:
cambia el objetivo de entreno, reabre el fine-tune del encoder y trae los
espacios de etiquetas diversos. `#honest-eval` recibe `#T-eval-cardinality`:
cardinalidad completa como métrica primaria, `beats_chance` corregido y
obligatorio en `eval/gate_rules.py`, dev/test separados y un solo scoreboard en
lugar de tres cortes que hay que cruzar a mano.

Orden de ejecución, fijado tras la revisión externa
(`revision-externa-2026-09-24.md`): protocolo (`#T-eval-cardinality`) →
información en las opciones (`#T-option-text`, sin entrenar) → vía exacta
(`#T-bigk-optsets`) → vía muestreada (`#T-fullspace-objective`) → encoder
(`#T-encoder-finetune`) → diversidad → paridad e integración
(`#T-jev-parity`, `#T-serve-engine`).

Riesgo declarado antes de medir: si con el objetivo nuevo la cifra a 77 vías
sigue pegada al azar, la hipótesis del objetivo queda refutada y el siguiente
sospechoso es la capacidad del backbone (68 M → 149 M+), que se escribirá como
NO-GO igual de explícito.

## 9. Revisión externa

Antes del primer run de fase 2 el plan pasó por una auditoría externa e
independiente, con una segunda revisión encima. Sus hallazgos —verificados uno a
uno contra el código— están en `revision-externa-2026-09-24.md` y ya están
incorporados aquí: el matiz del IC sobre la cifra a 77 vías, el coste real de la
cabeza con atención entre opciones, la vía exacta como primera ablación, la
diferencia de información con el profesor, la separación dev/test y tres
hallazgos de runtime (servidor sin motor, cargador sin encoder afinado, colisión
de caché en la API V1) que bloquean servir el modelo de fase 2 cuando exista.
