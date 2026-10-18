---
id: T-own-backbone
title: Entrenar desde cero el primer backbone de decisión propio
status: active
priority: critical
owner: unassigned
category: model
initiative: own-decision-model
depends_on: []
created: 2026-10-18
updated: 2026-10-18
---

# Primer backbone propio

Construir la ruta mínima que parte de pesos aleatorios y termina en un
checkpoint servible sin descargar ni cargar otro modelo.

## Alcance inicial

- Tokenizer entrenado con corpus admitido y manifiesto reproducible.
- Encoder bidireccional de unos 15 M de parámetros.
- Estado compartido, encoder de opciones con parámetros compartidos y cabeza
  de puntero.
- Entrenamiento supervisado con episodios propios y distilación opcional.
- Exportación a safetensors con configuración y hashes.
- Evaluación de orden, cardinalidad, etiquetas no vistas y sonda v3.

## Restricciones

- Inicialización aleatoria de todos los pesos publicables.
- Ningún checkpoint externo entra en la red estudiante.
- Los profesores se consultan fuera del forward del producto y sus salidas
  quedan ligadas a fuente, licencia y receta.
- La batería sellada no entrena ni elige arquitectura.
- TinyJev continúa disponible solo en el laboratorio aparcado.

## Gate de la fase 1

- La pérdida baja en un conjunto pequeño que el modelo puede memorizar.
- Supera el azar con margen en el corte de etiquetas no vistas.
- Mantiene la elección ante permutaciones de opciones.
- Acepta K dinámico dentro del contrato de memoria.
- El checkpoint carga sin red y sin cachés de modelos externos.
- Se publican parámetros, tokens, horas, energía aproximada y hashes.

## Siguiente decisión

Si 15 M aprende el contrato, entrenar 60 M con más corpus. Si no aprende, se
corrige arquitectura u objetivo antes de comprar cómputo. Solo se plantea GPU
alquilada después de medir el cuello con esta fase.

## Estado del 18 de octubre

- `model/own_backbone.py`: primer peldaño de 16.763.138 parámetros.
- `training/python/own_train.py`: lectura de `episode-v1`, entrenamiento desde
  cero, holdout, hashes y exportación safetensors.
- Smoke de 42.498 parámetros y 20 pasos: pérdida 1,3871 a 1,3552; holdout
  2/4. Prueba que la ruta completa ejecuta y guarda pesos.
- Gate de capacidad con 16.394.498 parámetros, 29 filas y 500 pasos: pérdida
  1,0984 a 0; precisión de entrenamiento 0,9655; holdout 2/3. PASS.
- Nueve tests pasan, incluidos el corte limpio y el lote de igual K.
- Gate de etiquetas no vistas, 18 de octubre, 19:14: 16.419.074 parámetros,
  2.500 pasos, lote 8, Metal, 451 s. Entrenamiento 284 filas, precisión 1,000.
  Evaluación 308 filas cuyos textos no aparecen en el entrenamiento:
  130/308 = 0,4221, azar 0,3013, cota inferior de Wilson 0,3682. Corte limpio.
  PASS. Es un lote sintético de reglas, no el benchmark v3.
