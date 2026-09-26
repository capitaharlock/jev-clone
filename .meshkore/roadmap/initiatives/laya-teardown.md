---
id: laya-teardown
title: Laya — el clon de Jev que ya funciona: qué copiar, qué medimos mal
status: next
owner: architect-master
modules:
  - model
  - eval
created: 2026-09-27
updated: 2026-09-27
---

# Laya — el clon de Jev que ya funciona: qué copiar, qué medimos mal

`github.com/NandhaKishorM/laya` (Apache-2.0, `convaiinnovations/laya*` en HF) es
el mismo producto que intentamos construir: System 1 no autorregresivo, preguntas
tipadas (`choice` / `score` / `noul`), una sola pasada, 33 ms, y un benchmark
público contra Jev. Está **publicado, medido y parcialmente por delante de Jev**
(typed-decisions 0,766 vs 0,727; ECE 0,081 vs 0,246).

La razón de abrir este work-stream no es copiar código: es que Laya **nos da una
cota superior externa y reproducible** para las tres cosas que ahora mismo
tenemos rotas o sin referencia — la transferencia a espacios de etiquetas
grandes, la sensibilidad al cambio decisivo, y la calibración.

**El dato que obliga a mirarlo.** En BANKING77 a cardinalidad completa (77
opciones) Laya mide **0,425** con una cabeza sobre encoder congelable, y declara
que su techo ahí es presupuesto de tokens por opción, no capacidad. Nuestro
pointer head en el mismo espacio mide **0,0123 contra un azar de 0,0130**
(`artifacts/gates/T-fullspace-objective/`). No es la misma distancia al problema:
es la diferencia entre «hay que afinar» y «no aprende».

## Lo que este work-stream tiene que contestar

1. **¿La diferencia es la arquitectura o el entreno?** Laya no pone una cabeza
   sobre un backbone congelado: mete un `[MASK]` por opción **dentro de la
   secuencia**, antes del estado, y lee el hidden state de ese marcador. El
   backbone ve estado, pregunta y opciones a la vez y se entrena entero.
2. **¿Laya acierta donde nosotros fallamos?** Los contrafactuales del reanálisis
   del operador (verde/rojo, cuál cuesta menos vs cuál dura más) son ejecutables
   contra su checkpoint sin entrenar nada.
3. **¿Su objetivo (RLCD) aporta algo sobre nuestro CE listwise?** Regla de
   puntuación estrictamente propia + CE suave + temperatura por (tipo, K).

## Restricciones

- Apache-2.0: se puede leer, medir y adaptar citando. **No se copia código al
  repo sin decisión explícita del operador**; el valor por defecto de este
  work-stream es medir y aprender.
- Nada aquí promete latencia ni sustituye a `#cross-encoder-pilot`: si Laya gana,
  informa a esa hipótesis; si pierde donde nosotros perdemos, confirma que el
  problema es el dato, no la arquitectura.

## Done when

- Está publicada la tabla de diferencias arquitectónicas Laya vs nuestro pointer
  head, con la lista de las que son **medibles** y cuál se elige probar.
- El checkpoint de Laya está medido, sin ajustar, sobre nuestra batería privada
  de `#honest-eval` y sobre los pares contrafactuales, con IC95 %.
- Está medido su BANKING77 a cardinalidad completa en nuestro arnés, para saber
  si su 0,425 reproduce o es de su propio harness.
- Existe un veredicto escrito: qué mecanismo adoptamos, cuál descartamos y por
  qué — o un NO-GO con su causa.
