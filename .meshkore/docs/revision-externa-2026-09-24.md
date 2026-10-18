---
title: "Revisión externa del plan de fase 2 — hallazgos y qué cambia"
updated: 2026-09-24
owner: architect-master
status: vigente
---

# Revisión externa del plan de fase 2

> Una auditoría externa e independiente leyó el árbol, los artefactos y el plan
> de `#full-space-training` antes de que se lanzara el primer run de fase 2, con
> una segunda revisión encima de la primera. No lanzó entreno ni modificó
> código. Este documento recoge lo que encontró, **qué de ello queda verificado
> contra el código**, y qué cambia en el plan. Es el motivo por el que el eval
> se mueve antes que el entreno y por el que el primer brazo de fase 2 no es el
> que estaba escrito.

El informe completo vive fuera del repositorio (carpeta de trabajo del
operador, no versionada). Todo lo que este documento afirma es comprobable
desde el árbol: cada hallazgo lleva su `fichero:línea`.

## 0. Lo que la revisión confirma y lo que corrige

**Confirma** el diagnóstico de fase 2: la capacidad del modelo para la tarea que
el producto promete no está demostrada, el orden «eval antes que entreno» es el
correcto, y el aparato de evaluación honesta (firewall, split por grupos,
holdout de etiquetas, NO-GO explícito) es de lo más sólido del proyecto.

**Corrige tres cosas del plan**, y las tres cambian el trabajo:

1. **La cifra a 77 vías no está «por debajo del azar»: es indistinguible del
   azar.** 38 aciertos en 3 080 = 0,01234, azar 0,01299, IC 95 %
   [0,00900, 0,01689] — el azar cae dentro. Lo que se demuestra es un fracaso de
   generalización, no un sesgo medido.
2. **El plan presuponía una cabeza que puntúa cada etiqueta por separado.** La
   cabeza actual hace atención *entre* opciones: el coste y la semántica del
   objetivo nuevo no son los que la task describía.
3. **Hay una explicación alternativa concreta que nadie había medido**: al
   profesor se le dan definiciones de las 77 categorías y 24 ejemplos
   etiquetados; a nuestro modelo se le dan identificadores crudos
   (`card_arrival`). Antes de atribuir nada al denominador de la pérdida, eso se
   mide — y es barato, no necesita entrenar.

## 1. Hallazgos sobre el objetivo de entreno

### A — la cabeza no puntúa opciones de forma independiente · **verificado**

`model/decision_head.py:105-110` aplica `set_attn` entre todas las opciones
dentro de cada `CrossBlock`, y el logit de abstención se construye con
`ctx.mean(0)` (`:176-182`): el contexto de cada opción **y** el logit `unknown`
dependen del conjunto entero.

Consecuencias, las tres reales:

- Cachear las claves de texto de etiqueta ahorra *forwards del encoder*, no el
  coste de atención por fila, que crece ≈ K² en ese bloque. La promesa «el coste
  por fila no crece con K» no se sigue de esta arquitectura.
- Los logits calculados en conjuntos distintos **no se pueden unir** como si
  fueran un solo softmax. Negativos in-batch exige pasar el conjunto ampliado
  por la cabeza para cada fila, o diseñar una rama de scoring independiente.
- Si se diseña esa rama, el experimento deja de variar sólo el denominador: es
  un cambio de arquitectura y se evalúa como tal.

**Cambia:** `#T-fullspace-objective` mide coste y memoria a K=8/41/77/cientos
antes de prometer nada, y declara si conserva la cabeza o la cambia.

### B — el diseño de negativos y log-Q no estaba especificado · **verificado**

`MixtureStream.epoch()` (`training/python/train_decision.py:399-416`) sirve
batches de **un solo dataset**. En una tarea binaria, la unión de etiquetas de un
batch de 128 filas sigue teniendo dos candidatos: «cientos de etiquetas por
batch» no se sigue del cargador actual.

Además, el plan mezclaba tres universos de negativos (espacio de la fila,
etiquetas del batch, etiquetas de otros espacios) sin decir qué hacer con los
falsos negativos entre espacios, los duplicados, el gold ni `unknown`. Y
`prior_penalty` (`train_decision.py:741-784`) resta un log de frecuencia
empírica, que **no equivale** a una corrección log-Q.

**Cambia:** especificación matemática con dos modos verificables (espacio exacto
/ subconjunto muestreado) como puerta de entrada al run, y medición previa de
etiquetas únicas reales por batch.

### C — ampliar K ya amplía el denominador de la pérdida actual · **verificado**

`cross_entropy(logits, gold)` (`train_decision.py:1543`) normaliza sobre todas
las columnas que se le den. Si a una fila se le entregan las 77 etiquetas de su
espacio, **esa misma llamada ya normaliza sobre el espacio entero**, más el
logit `unknown`. La frase de `#T-bigk-optsets` «subir K sin cambiar la pérdida
sólo encarece cada fila» era incorrecta.

**Cambia el orden de ejecución**: la vía exacta —entregar el espacio enumerable
completo y reutilizar la CE que ya existe— es la primera ablación, más simple y
más barata que añadir a la vez negativos in-batch y log-Q. El muestreado se
especifica aparte y llega después.

### D — el checkpoint de referencia no se entrenó con BANKING77 · **verificado**

`artifacts/runs/leverstack-d512-prior-ettin-68m-s20260922/run.json` declara
`fence.clean_1m: true` y `fenced_datasets: [banking77, helpsteer2, pubmedqa]`; sus
12 datasets no incluyen BANKING77. Eso demuestra **exclusión del dataset**; no
demuestra ausencia literal de cada cadena de etiqueta en las demás fuentes, ni
dice nada de la exposición previa del backbone preentrenado. La regla genérica
de holdout del 25 % (`train_decision.py:129-131`) no describe este run.

**Cambia:** 0,0123 se etiqueta como lo que es —transferencia a un espacio
externo al corpus de ajuste—, la comprobación de solape literal se hace, y los
runs futuros que sí incluyan BANKING77 separan gold visto/no visto dentro del
mismo K=77.

### D2 — al profesor y a nosotros no se nos da la misma información · **verificado**

`data/adapters.py:146` construye cada opción como `Option(id=l, text=l)`: el
modelo ve `direct_debit_payment_not_recognised`, un identificador. El
experimento publicado del profesor entrega **definiciones en lenguaje natural**
de las 77 categorías y recupera **24 ejemplos etiquetados** por predicción.
Igualar filas y K no iguala el protocolo: la información disponible en las
opciones difiere, y es una explicación alternativa concreta del resultado.

Además `eval/calib.py:373-375` trunca el estado a 256 tokens: meter 24 ejemplos
en el `STATE` sin medir qué conserva el tokenizador puede truncar las
demostraciones o la propia consulta.

**Cambia:** task nueva `#T-option-text`, tres brazos sobre el checkpoint que ya
existe, sin entrenar nada. Es lo más barato del plan y lo que decide si la
hipótesis del denominador sigue siendo la primera sospechosa.

## 2. Hallazgos sobre la evaluación

### E — `beats_chance` se decide con el ranking forzado · **verificado**

`eval/fullspace.py:159` compara **`flo`** (límite inferior de
`accuracy_options_only`, el ranking sin `unknown`) contra el azar. Un modelo que
abstenga en todas las filas tendría `accuracy = 0` y podría publicar
`beats_chance: true`. En el artefacto actual `abstain_rate = 0` y el veredicto no
cambia, pero la semántica del gate está mal.

**Cambia:** se separan `beats_chance` (con `lo`, la accuracy real) y
`ranking_beats_chance` (con `flo`), con test que lo fija.

### F — el barrido de K está sobreinterpretado · **verificado**

Sobre 1 000 filas: K=5 da 0,215 con IC [0,1907, 0,2415] contra azar 0,200 — **no
bate el azar por IC**. K=20 tampoco. K=8 y K=40 sí. «La ventaja aguanta hasta
K=40» describe una progresión monótona que los intervalos no sostienen.

**Cambia:** el barrido se publica como diagnóstico exploratorio con conteos e
intervalos, y la «preferencia local» vuelve a ser hipótesis, no conclusión.

### G — el test oficial se está usando para elegir experimentos · **verificado**

`eval/fullspace.py:106` carga el split `test`, y `#T-fullspace-objective` proponía
decidir GO/NO-GO a 62 k mirando esas mismas 3 080 filas. Si contra ellas se
prueban K, log-Q, encoder, descripciones, mezcla y seeds, el test se convierte en
conjunto de desarrollo por selección adaptativa aunque nunca entre en el
gradiente.

**Cambia:** corte de desarrollo congelado para elegir brazos, test oficial
reservado, y **registro de las consultas ya hechas** al test. Regla nueva R7.

### H — el `prior_penalty` del mejor checkpoint necesita interpretación explícita

El trainer resta `log(count)` antes de la CE (`train_decision.py:1539-1543`) pero
`evaluate()` puntúa logits crudos (`:872-884`), igual que el motor. Puede ser una
compensación deliberada, pero no equivale a restar ese prior en inferencia. En
BANKING77, excluido del entreno, `prior_penalty_for()` devuelve `log(1)=0` para
IDs no observados: aplicarlo allí sería un control sin efecto. **No** es un
sospechoso válido del resultado a K=77.

## 3. Hallazgos sobre runtime y release

Los tres son de lectura estática de rutas de código; ninguno explica el 0,0123,
y los tres bloquean servir el modelo de fase 2 cuando exista.

### J — el servidor no está conectado al motor neuronal · **verificado**

Existe implementación Candle real del encoder y del pointer head
(`crates/jev-model/src/{engine,modernbert,pointer}.rs`), pero `jevclone serve`
llama a `jev_server::serve` sin entregarle un `Engine`
(`crates/jev-cli/src/main.rs:37-60`), y `/v1/choice` recibe `weights`, `bias` e
`input` por la petición y los manda al scorer lineal (`v1.rs:28-37, 141-154`).
Las opciones de esa API sólo tienen `id`, sin texto semántico, y `unknown` se
decide por umbral — contrato distinto al del pointer entrenado.

### K — el cargador Rust ignoraría un encoder afinado · **verificado**

El trainer guarda `backbone.safetensors` con su nombre y hash cuando el encoder
es entrenable (`train_decision.py:973-978`). En Rust, `BackboneRef` sólo
deserializa `id`, `hidden_size` y `revision` (`engine.rs:40-44`) y `Engine::load`
carga siempre `artifacts/weights/<id>/pytorch_model.bin` (`:149-170`). Un
checkpoint de fase 2 cargaría con la cabeza nueva y el encoder original. Es
central para fase 2: hay que cargar y verificar ese backbone, o rechazar el
régimen explícitamente.

### L — la clave de caché de la API V1 colisiona entre entradas distintas · **verificado**

`choose()` construye `state_hash` y `tokenizer_hash` como cadenas vacías
(`crates/jev-server/src/v1.rs:144-147`), y esa clave compuesta es la que usa
`Runtime::encode()` para devolver el vector guardado
(`jev-runtime/src/lib.rs:153-162`). Dos peticiones con la misma `model_version` y
entradas diferentes comparten clave: la segunda puede puntuar la entrada de la
primera. Afecta también a `choose_batch()`.

Caso mínimo: dos opciones, `weights=[1,-1]`, `bias=[0,0]`, `input=[1]` y después
`input=[-1]`, misma versión. Lo correcto pasa de ≈`[0,881, 0,119]` a
≈`[0,119, 0,881]`; con esta clave la segunda petición reutiliza la primera.

## 4. Higiene de repositorio

- `git ls-files target` devuelve 9 145 ficheros todavía versionados: ignorar no
  desindexa lo ya indexado (`#T-repo-clean`).
- La model card de 2026-09-21 menciona pesos «pending» y métricas anteriores a
  los checkpoints actuales (`#T-readme-card`).
- `context.md` enlazaba planes bajo la carpeta externa del operador: un clon
  fresco no los resuelve. Las referencias que importan se sustituyen por
  documentos versionados.

## 5. Qué queda claro sobre lo que **no** sabemos

La evidencia actual demuestra un fracaso de generalización en los protocolos
medidos. **No aísla** ninguna de estas causas, y el plan de fase 2 no puede dar
ninguna por establecida:

| sospechoso | estado |
|---|---|
| denominador de la pérdida (K≤8) | hipótesis principal, sin aislar |
| información en el texto de las opciones (ids vs definiciones + ejemplos) | hipótesis nueva, **medible sin entrenar** |
| atención entre opciones de la cabeza | confunde el experimento del denominador |
| encoder congelado | medido sólo en régimen de fase 1, no transferible |
| diversidad de espacios de etiquetas | sin control a igual volumen |
| capacidad del backbone (68 M) | siguiente sospechoso si el objetivo no mueve nada |

Regla de conducta que sale de aquí: un NO-GO a 62 k limita gasto, **no establece
causa**. El brazo ganador se repite en otra seed antes de cualquier veredicto
causal.

## 6. Orden de trabajo que resulta

1. **Protocolo** — `#T-eval-cardinality`: jerarquía de métricas, `beats_chance`
   corregido, dev/test separados con registro de consultas, scoreboard único con
   frecuencia de predicciones y matriz de confusión, baseline reproducible.
2. **Información en las opciones** — `#T-option-text`: tres brazos a K=77 sobre
   el checkpoint actual. Sin entrenar. Decide si el denominador sigue siendo la
   primera hipótesis.
3. **Vía exacta** — `#T-bigk-optsets`: espacio enumerable completo por la CE que
   ya existe, con el coste de la cabeza medido.
4. **Vía muestreada** — `#T-fullspace-objective`: especificación de la pérdida
   con sus tests, medición de etiquetas únicas por batch, y sólo entonces el run.
5. **Encoder** — `#T-encoder-finetune`, sobre el objetivo que haya ganado, con
   memoria medida en esa configuración.
6. **Diversidad** — `#T-labelspace-div` / `#T-labelspace-factory`, con control a
   igual volumen.
7. **Paridad e integración** — `#T-jev-parity`, `#T-serve-engine`.
