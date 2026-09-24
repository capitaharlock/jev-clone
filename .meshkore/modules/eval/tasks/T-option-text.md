---
id: T-option-text
title: Qué información recibe el modelo en las opciones — tres brazos a K=77 sin entrenar
status: next
priority: high
owner: unassigned
category: eval
initiative: full-space-training
depends_on:
  - T-eval-cardinality
created: 2026-09-24
updated: 2026-09-24
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
