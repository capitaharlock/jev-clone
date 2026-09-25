---
id: T-teacher-probe
title: Cablear el profesor y medir la distancia real — mismo corte, mismas filas
status: backlog
priority: high
owner: unassigned
category: eval
initiative: teacher-distill
created: 2026-09-23
updated: 2026-09-25
---

# Cablear el profesor y medir la distancia real

Primera task de `#teacher-distill` y prerrequisito de las otras dos. Hace dos
cosas y ninguna toca pesos.

**1. El cliente.** Un módulo `eval/teacher.py` con: la key leída de entorno o de
un fichero fuera de git (nunca en el repo, nunca en un JSON de gate), caché en
disco por hash de `(modelo, prompt, opciones)` para que re-ejecutar un
experimento cueste 0, reintento con backoff, y un contador de tokens y coste que
cada gate que lo use tiene que publicar. Falta por confirmar con el operador el
**endpoint base y el nombre del modelo**: la key está, la URL no aparece en el
repo ni en los documentos de plan.

**2. La medición que contesta la pregunta del operador.** Se le pasa al profesor
el mismo corte `unseen` de `#T-unseen-labels` —las mismas filas, el mismo
`STATE`, el mismo conjunto de opciones, el mismo formato de respuesta forzada— y
se publica su accuracy al lado de la nuestra, por corte (banking77, huffpost,
massive, ...). Es la única forma honesta de decir "estamos a N puntos": el
umbral de release lo escribimos nosotros, el profesor no.

Muestreo: no se puntúan las 5 624 filas del corte completo de entrada. Se
empieza por una submuestra estratificada por corte con IC95 declarado (≈ 400
filas), se publica el coste, y sólo se amplía si el operador lo aprueba viendo
el gasto por punto de precisión.

## Estado 2026-09-23

**Hecho.** `eval/teacher.py` (cliente: caché en disco por hash de
`(modelo, state, pregunta, opciones)`, presupuesto duro en llamadas y en
dólares, reintento con backoff, contador de tokens/coste, clave fuera de git
y redactada de todo lo que sale del módulo) y `eval/teacher_probe.py` (corte
`unseen`, submuestra estratificada y sembrada, nuestro head re-puntuado sobre
ESAS filas, tabla de distancia por corte). 29 tests verdes contra un servidor
HTTP local que habla el formato System-One; nada stubbed en producción.
Submuestra versionada: 399 filas, coste estimado **$0,0018** — el corte
completo de 5 624 filas costaría ~$0,03, así que el muestreo deja de ser una
restricción de presupuesto en cuanto haya endpoint.

**Bloqueado.** La key que dio el operador (`apikey_2217…f0e797dc`) devuelve
401 en los seis endpoints públicos de Jev que existen: `api.typesafe.ai/v1/
systemone`, `tokenra.io/v1/decisions`, `jev-ai.pro/api/v1/systemone`,
`thejevai.com/v1/systemone`, `api.venice.ai/api/v1/decisions` y
`jevai.org/api/v1/decisions`. Falta que el operador diga **de qué proveedor
es la key**; con eso son dos variables de entorno y un `gate`.

**Medido mientras tanto** (`fullspace.json`, `eval/fullspace.py`): nuestro
head sobre las 3 080 filas de test de BANKING77 con las **77 etiquetas** como
opciones — el régimen en el que Jev publica 0,924 — da **0,0123, por debajo
del azar (0,0130)**. El barrido de cardinalidad sobre las mismas filas
(K=5/8/20/40/77) muestra que la ventaja sobre azar aguanta hasta K=40 y
desaparece en K=77: lo que el head tiene con pocas opciones es una
preferencia débil, no un ranking del espacio real de etiquetas. Eso abre
`#T-bigk-optsets`.

## Done when

- `eval/teacher.py` existe con caché en disco, presupuesto y contador de coste, y
  su suite verde; la key no aparece en ningún fichero versionado.
- `artifacts/gates/T-teacher-probe/gate.json` publica accuracy del profesor por
  corte unseen, con n, IC95, coste en tokens y en dinero, y el `model_version`
  del checkpoint nuestro contra el que se compara.
- El gate incluye la línea de distancia: profesor − nosotros, por corte, y el
  corte donde la diferencia es mayor.
- La submuestra y su semilla quedan versionadas para que la medición sea
  repetible sin volver a gastar saldo.
