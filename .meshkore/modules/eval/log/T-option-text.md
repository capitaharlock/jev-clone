---
id: T-option-text
title: Qué información recibe el modelo en las opciones — tres brazos a K=77 sin entrenar
status: done
priority: high
owner: unassigned
category: eval
initiative: full-space-training
depends_on:
  - T-eval-cardinality
created: 2026-09-24
updated: 2026-09-24
completed_at: 2026-09-24T12:25:00.656Z
resolved_by: A036
resolved_by_conv: work-full-space-training-T-option-text-1790251049
commit_shas: ['92480b1a9f8c4a5d3bf61a17b56aa3c25ff8d976']
---
# Qué información recibe el modelo en las opciones

La revisión externa (`.meshkore/docs/revision-externa-2026-09-24.md`, hallazgo
D2) encontró una diferencia de protocolo que no estaba medida y que compite
directamente con la hipótesis del denominador:

- **Nosotros**: `data/adapters.py:146` construye cada opción como
  `Option(id=l, text=l)`. El modelo ve la cadena `direct_debit_payment_not_recognised`
  — un identificador, no una descripción.
- **El profesor**: su experimento publicado entrega **definiciones en lenguaje
  natural** de las 77 categorías y recupera **24 ejemplos etiquetados** por
  predicción.

Igualar filas y K no iguala el protocolo. Antes de atribuir el 0,0123 al
objetivo de entreno hay que saber cuánto de la distancia lo explica la
información disponible en la entrada. Es la medición más barata de todo el plan:
**no entrena nada**, corre sobre el checkpoint que ya existe.

## Brazos

Mismo checkpoint (`leverstack-d512-prior-ettin-68m-s20260922`), mismas filas,
K=77 siempre, azar 0,0130 impreso al lado:

| brazo | texto de la opción | `STATE` |
|---|---|---|
| **A — actual** | `card_arrival` | sólo la consulta |
| **B — legible** | nombre legible + descripción de la categoría | sólo la consulta |
| **C — con ejemplos** | nombre legible + descripción | consulta + ejemplos etiquetados |

El brazo C reproduce el régimen del profesor hasta donde la ventana lo permita.

## El presupuesto de contexto es parte de la medición (regla R8)

`eval/calib.py:373-375` trunca el estado a `TRAIN_MAX_LENGTH = 256` tokens. Meter
ejemplos en el `STATE` sin mirar qué sobrevive puede truncar las demostraciones —
o la propia consulta, que es peor. El artefacto registra, por brazo: tokens
retenidos, cuántos ejemplos entraron completos, dónde queda la consulta dentro
de la ventana, y el presupuesto total. Si el brazo C no cabe, eso **es** el
resultado y se publica como tal, no se fuerza.

## Disciplina de corte

Los tres brazos se eligen y se comparan en el **corte de desarrollo** de
`#T-eval-cardinality`. El test final se toca una sola vez, con el brazo ya
elegido, y la consulta queda registrada (regla R7).

## Qué decide

- Si B o C mueven la cifra de forma apreciable, la explicación del resultado a
  77 vías se reparte y `#T-fullspace-objective` deja de ser la primera hipótesis
  — además, el texto de etiqueta pasa a ser materia del corpus, no del eval.
- Si los tres brazos se quedan pegados al azar, la información de entrada queda
  descartada como causa y el denominador de la pérdida conserva la prioridad,
  ahora con un competidor menos.

De paso, deja escrito qué texto de opción usa el repo por defecto: si las
descripciones ayudan en eval, entrenar con identificadores crudos es una
decisión que hay que tomar a propósito, no por omisión.

## Done when

- Los tres brazos medidos sobre el mismo checkpoint y las mismas filas a K=77,
  con accuracy, aciertos, IC 95 %, azar y tasa de abstención en un solo
  artefacto.
- El presupuesto de contexto publicado por brazo: tokens retenidos, ejemplos
  completos, posición de la consulta.
- La fuente de las definiciones y de los ejemplos declarada y versionada, con su
  licencia, para que el brazo sea reproducible.
- Veredicto escrito en una línea: cuánta de la distancia con el profesor explica
  la información de entrada, y qué prioridad le queda al denominador.

## Resultado (2026-09-24)

Artefacto único: `artifacts/gates/T-option-text/optiontext.json`. Comando:
`.venv-train/bin/python -m eval.optiontext run --checkpoint <ckpt>`. No
entrena nada: un forward por brazo sobre
`leverstack-d512-prior-ettin-68m-s20260922/stage-001000000`.

Corte de desarrollo sellado (`banking77-dev`, 1 000 filas, K=77, azar
0,012987):

| brazo | aciertos | accuracy | IC 95 % | bate azar | ejemplos completos | consulta entera |
|---|---:|---:|---:|---|---:|---:|
| A — actual | 9 / 1 000 | 0,0090 | [0,0047, 0,0170] | no | — | 100 % |
| B — legible + definición | 14 / 1 000 | 0,0140 | [0,0084, 0,0234] | no | — | 100 % |
| C — con ejemplos | 13 / 1 000 | 0,0130 | [0,0076, 0,0221] | no | 11,9 / 24,0 | 100 % |
| C-orden-profesor *(diagnóstico)* | 12 / 1 000 | 0,0120 | [0,0069, 0,0209] | no | 12,7 / 24,0 | **0,1 %** |

Los cuatro intervalos contienen el azar y se solapan entre sí. El mejor
punto en desarrollo era B; **no reproduce**. Una sola consulta al corte
reservado (`banking77-test`, 3 080 filas, registrada en
`artifacts/gates/T-option-text/test-queries.json` y en el registro R7 de
`#T-eval-cardinality`), con el brazo ya elegido y A como control pareado:
A 38/3 080 = 0,0123 [0,0090, 0,0169] — reproduce exactamente la cifra
publicada — y B 37/3 080 = 0,0120 [0,0087, 0,0165]. Azar 0,012987 en ambos.

### El presupuesto de contexto (R8)

La ventana es `TRAIN_MAX_LENGTH = 256`, leída, nunca subida. Con 24 ejemplos
recuperados el estado pide 487 tokens de media y retiene 256: se caen
231 453 tokens en 1 000 filas y **sólo la mitad de las demostraciones entra
entera** (49,6 %). Y el orden decide qué mitad: con la consulta primero
sobrevive en el 100 % de las filas; en el orden del profesor —ejemplos
primero, consulta al final, que es lo que «reproducir el régimen» significa
literalmente— la consulta entera sobrevive en **1 fila de 1 000**. El
régimen del profesor no cabe en esta ventana, y eso es el resultado, no un
obstáculo que se rodee.

El texto de las opciones sí cabe: van por `embed_texts` con ventana 512 y el
brazo B gasta 21,6 tokens de media (máximo 36), cero truncados.

### Fuentes

- Definiciones: `data/taxonomies/banking77-jev/descriptions.json`, las 77 del
  propio experimento del profesor, byte a byte, commit
  `5cac4ff7783a4cfc0badba4124a309dd2a2b9862`, sha256 `ea930901…`, CC-BY-4.0
  con la salvedad declarada en `card.json` (el repo de origen no lleva
  LICENSE en la raíz; la CC-BY-4.0 es la de los datos de los que derivan).
- Ejemplos: filas de train de BANKING77 (CC-BY-4.0), menos las 1 000 del
  corte sellado; BM25 sobre palabras y pares adyacentes, ≤24, ≤4 por clase,
  reimplementado desde `PROTOCOL.md`.

### Veredicto

La información de entrada no explica nada de la distancia con el profesor:
desplaza el punto +0,0050 en desarrollo (0,5 % del hueco de 0,915) y −0,0003
en el corte reservado, ambos dentro del ruido. `#T-fullspace-objective`
conserva la primera prioridad, ahora con la hipótesis de la información de
entrada **descartada** en vez de sin ordenar. Corolario para el corpus:
entrenar con identificadores crudos deja de ser una decisión por omisión —
medida, no cuesta nada a K=77 sobre este checkpoint.

Límite declarado (R9): una semilla y un checkpoint. Esto pone precio a la
entrada; no establece una causa.

## Resolution

✓ task #T-option-text done. files: 14. commit: `92480b1`. pytest eval/ + data/: 3 fallos en eval/, los 3 pre-existentes; suite global 886 passed / 7 failed, exactamente los 7 pre-existentes que avisaste. ruff limpio. Artefacto en disco y legible.

**Los tres brazos se quedan pegados al azar.** La información de entrada no explica nada de la distancia con el profesor — `#T-fullspace-objective` conserva la primera prioridad, ahora con un competidor **descartado** y no sólo sin ordenar.

| brazo | dev (n=1000) | IC 95 % | test reservado (n=3080) | IC 95 % |
|---|---:|---:|---:|---:|
| A — actual | 9 → 0,0090 | [0,0047, 0,0170] | 38 → 0,0123 | [0,0090, 0,0169] |
| B — legible + definición | 14 → 0,0140 | [0,0084, 0,0234] | 37 → 0,0120 | [0,0087, 0,0165] |
| C — con ejemplos | 13 → 0,0130 | [0,0076, 0,0221] | — | — |

Azar 0,012987 en todas. K=77 siempre. El mejor punto en dev era B; **no reproduce** en el corte reservado.

<details><summary>Presupuesto de contexto (R8) — el brazo C no cabe, y eso es el resultado</summary>

Ventana `TRAIN_MAX_LENGTH = 256`, leída, nunca subida.

| brazo | tokens pedidos | retenidos | filas truncadas | ejemplos enteros | consulta entera |
|---|---:|---:|---:|---:|---:|
| A / B | 16,0 | 16,0 | 0 / 1000 | — | 100 % |
| C (consulta primero) | 487,3 | 255,8 | 999 / 1000 | 11,9 / 24,0 (49,6 %) | 100 % |
| C-orden-profesor *(diagnóstico)* | 486,3 | 255,8 | 999 / 1000 | 12,7 / 24,0 | **0,1 %** |

Se caen 231 453 tokens en 1000 filas. El orden decide qué mitad muere: con la consulta primero sobrevive siempre; en el orden literal del profesor —ejemplos primero, consulta al final— la consulta entera sobrevive en **1 fila de 1000**. El texto de opción sí cabe: va por `embed_texts` (ventana 512), brazo B 21,6 tokens de media, máx 36, cero truncados.
</details>

<details><summary>Fuentes — declaradas, pinneadas, con licencia y su salvedad</summary>

- **Definiciones**: `data/taxonomies/banking77-jev/descriptions.json` — las 77 del propio experimento del profesor, byte a byte, commit `5cac4ff7783a4cfc0badba4124a309dd2a2b9862`, sha256 `ea930901…`. Licencia CC-BY-4.0 **de los datos de los que derivan** (`sources/DATA_LICENSE` del mismo repo). **Salvedad declarada en `card.json`**: el repo de origen no lleva LICENSE en la raíz, así que eso no es una concesión explícita sobre el fichero de definiciones; se redistribuye sin modificar, atribuido y pinneado, como cita de una configuración experimental publicada. Marcado `eval-only`: son las definiciones del experimento con el que nos comparamos.
- **Ejemplos**: filas de train de BANKING77 (CC-BY-4.0), menos las 1000 del corte sellado, menos cualquier fila cuya forma normalizada coincida con la consulta. BM25 sobre palabras y pares adyacentes, ≤24, ≤4 por clase — reimplementado desde el `PROTOCOL.md` del profesor, no vendorizado, y dicho así en la card.
- `data/taxonomy.py` verifica el sha256 antes de devolver nada: un fichero editado en local deja de ser la fuente que la card cita.
</details>

<details><summary>Corte reservado — una sola consulta, registrada en dos sitios</summary>

Una lectura, con el brazo ya elegido en dev (B) y A como control pareado en la misma pasada. A reproduce exactamente la cifra publicada (38/3080 = 0,0123), lo que valida la máquina. Sin brazo C: su veredicto es un hallazgo de presupuesto ya cerrado en dev.

Registrada en `artifacts/gates/T-option-text/test-queries.json` **y** en el registro R7 del repo (`artifacts/gates/T-eval-cardinality/test-queries.json`, que pasa de 8 a 9 consultas). Escribe en los dos a propósito: un registro por task que el recuento del repo no viera haría parecer el corte reservado menos consultado de lo que está.
</details>

<details><summary>Ficheros y verificación</summary>

Nuevos: `eval/optiontext.py`, `eval/test_optiontext.py`, `data/taxonomy.py`, `data/test_taxonomy.py`, `data/taxonomies/banking77-jev/{descriptions,card}.json`, `artifacts/gates/T-option-text/{optiontext,test-queries}.json`.
Tocados: `artifacts/

…(truncated)

**Commit** `92480b1a9` · 14 files · 16M tokens

**Files changed (14):**
- `.meshkore/docs/cortes-de-evaluacion.md`
- `.meshkore/docs/coverage.md`
- `.meshkore/docs/fase-2-espacio-completo.md`
- `.meshkore/docs/source-register.md`
- `.meshkore/modules/eval/tasks/T-option-text.md`
- `artifacts/gates/T-eval-cardinality/test-queries.json`
- `artifacts/gates/T-option-text/optiontext.json`
- `artifacts/gates/T-option-text/test-queries.json`
- `data/taxonomies/banking77-jev/card.json`
- `data/taxonomies/banking77-jev/descriptions.json`
- `data/taxonomy.py`
- `data/test_taxonomy.py`
- `eval/optiontext.py`
- `eval/test_optiontext.py`
