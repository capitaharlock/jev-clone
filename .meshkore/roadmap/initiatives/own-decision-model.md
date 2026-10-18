---
id: own-decision-model
title: Modelo de decisión propio
status: active
owner: architect-master
modules:
  - model
  - data
  - eval
  - runtime
created: 2026-10-18
updated: 2026-10-18
---

# Modelo de decisión propio

El producto es una red de decisión diseñada aquí, inicializada al azar y
entrenada aquí. Sus pesos publicados proceden solo de ese entrenamiento. El
repositorio público contiene la arquitectura, el entrenador, el formato de
pesos, la evaluación y el runtime necesarios para reproducirla.

TinyJev, JEv, Qwen, Laya y otros modelos pueden actuar localmente como
profesores, referencias, generadores, críticos o controles. Su código y sus
pesos no forman parte del producto, del checkpoint publicado ni de la ruta de
inferencia.

## Primera arquitectura

Un encoder bidireccional pequeño produce una representación compartida del
estado y la pregunta. Un encoder de opciones comparte parámetros y una cabeza
de puntero puntúa cualquier conjunto dinámico de opciones. No hay vocabulario
cerrado de etiquetas ni decodificación autorregresiva.

El tokenizer se entrena con el corpus admitido y se publica junto al modelo.
La inicialización del encoder y de la cabeza es aleatoria. PyTorch, MLX,
Candle y bibliotecas de tokenización son herramientas de ejecución, no pesos
ni arquitecturas externas incorporadas al modelo.

## Escalera local

1. Probar que una red de unos 15 M de parámetros aprende las familias
   sintéticas y supera el azar en etiquetas no vistas.
2. Escalar a unos 60 M con el corpus admitido y distilación local.
3. Escalar hacia 150-250 M solo cuando v3 muestre que la capacidad limita.
4. Alquilar GPU únicamente después de medir tokens, memoria, tiempo y coste
   con la red pequeña. No se presupone un preentrenamiento de 4B.

## Datos y profesores

- La fábrica publica corpus fuente, proyecciones y licencias.
- Los ejemplos externos entran solo tras deduplicación y firewall.
- Los profesores pueden aportar etiquetas blandas, preferencias y
  explicaciones; el estudiante empieza desde pesos aleatorios.
- El corte sellado y JEv nunca entrenan.

## Done when

- Un checkout limpio puede entrenar la red desde inicialización aleatoria.
- El checkpoint publicado no necesita pesos, código ni descarga de un modelo
  externo.
- El gate de linaje prueba hashes de datos, tokenizer, configuración y pesos.
- El modelo supera el azar en etiquetas no vistas y tiene resultado v3
  reproducible.
- El runtime Rust carga ese mismo checkpoint y coincide con la referencia.

## Camino TinyJev conservado

El autoloop TinyJev queda aparcado como laboratorio y posible producto
derivado sujeto a su licencia. Su historial, métricas, código de integración y
artefactos locales se conservan, pero no deciden el campeón de esta iniciativa.
