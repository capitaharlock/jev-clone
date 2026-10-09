---
id: T-open-decision-backbone
title: Sustituir el encoder listwise por un Qwen con cabeza puntero
status: active
priority: critical
owner: unassigned
category: model
initiative: daily-learning-loop
depends_on:
  - T-aligned-ranking-v2
  - T-dev-rotation
created: 2026-10-02
updated: 2026-10-03
---

# Sustituir el encoder listwise por un Qwen con cabeza puntero

## Decisión

El brazo principal pasa de mDeBERTa listwise (280M) a TinyJev 4B v1:
Qwen3-4B-Base, LoRA rango 16 y una cabeza puntero que puntúa el estado de
`<decide>` contra el final de cada opción. Se ejecuta con MLX e INT8.

El bucle listwise quedó parado después del ciclo 74. Ese ciclo retuvo 0,7225.
Los ciclos 68 a 73, con pérdida pareada 0,3, promediaron 0,740 y no movieron
el nivel.

## Medición del 2 de octubre

Las 400 filas de `battery_dev` se puntuaron con el mismo estado, pregunta,
opciones e IDs. El corte sellado no se leyó.

| Modelo | Forced | Joint | K=2 | K=3 | K=8 | Tiempo |
|---|---:|---:|---:|---:|---:|---:|
| `loop-0024` | 0,7325 | 0,5857 | 0,7875 | 0,6917 | 0,7000 | referencia |
| TinyJev 0,6B, MLX INT8 | 0,7325 | 0,5571 | 0,8063 | 0,7833 | 0,5833 | 6,7 s |
| **TinyJev 4B v1, MLX INT8** | **0,9250** | **0,8571** | **0,9625** | **0,8917** | **0,9083** | **27,9 s** |

El gate pareado contra `loop-0024` da GO:

- forced: +0,1925, IC95 [0,1475, 0,2425];
- joint: +0,2714, IC95 [0,1786, 0,3571];
- ninguna familia ni idioma cae;
- comparación: 0,60 a 0,95;
- prioridad: 0,5375 a 0,80.

La evaluación reproducible es:

```bash
PYTHONPATH=. .venv-train/bin/python -m eval.open_decision_baseline \
  --model TinyJev-4B-v1 \
  --source tmp/jev-research/tinyjev \
  --backend mlx --quantize 8 \
  --out artifacts/gates/T-open-decision-backbone/eval-dev.json
```

## Riesgos antes de release

El orden de entrada dio 0,925. Invertir las opciones dio 0,9075. Cuatro
permutaciones con semilla dieron entre 0,8875 y 0,9075, con 20 a 26 ganadores
distintos sobre 400. Sigue por encima del objetivo, pero el orden no es
invariante. Producción debe imponer un orden estable y el gate debe publicar
también el peor orden.

Una búsqueda literal no encontró tres frases de muestra de la batería en los
repositorios descargados. Eso no descarta solapes de plantilla con
`decision-v7`, el corpus con el que se entrenó TinyJev. La batería se escribió
después de la primera versión del modelo, pero el siguiente veredicto debe
salir de `T-dev-rotation`, no de volver a elegir con estas mismas 400 filas.

## Siguiente pase

1. Cerrar la auditoría de fuga por estado, pregunta, opciones, n-gramas y
   plantilla contra los datos públicos de Kev y TinyJev.
2. Ejecutar el dev rotado y las seis permutaciones. Umbral: forced mayor que
   0,80, joint mayor que 0,70 y ninguna familia menor que 0,75.
3. El 3 de octubre el operador ordenó el LoRA de continuación. Corre
   `tools/pointer_loop.py`: TinyJev-4B v1, LoRA rango 16 en atención y MLP,
   cabeza puntero, entropía cruzada, orden barajado y opción ninguna en
   el 25% de las vistas. Train: 215906 episodios de regla. Holdout: 24094,
   por grupo, sólo como monitor. `battery_dev` y el sellado no entran.
   El lote 96 agotó la memoria al tercer paso. El lote vivo es 48.
   Un dev rotado sigue siendo el veredicto, no este holdout.
   El reinicio conserva paso global, decisiones, época, mejor holdout y estado
   del optimizador. Guarda un checkpoint cada 10 pasos. La migración recuperó
   el adaptador del paso global 528 sin estado del optimizador; el paso 530 ya
   escribió el primer checkpoint completo.    El panel separa ahora la batería
   medida, el holdout de regla y el encoder listwise legado (0,7325).
   El adaptador del paso 996 puntúa la batería en 0,9525 (381/400) en el
   orden escrito. Cuatro permutaciones dan 0,9425, 0,9475, 0,9525 y 0,9600.
   El 0,95 no se sostiene al mover las opciones. Prioridad queda en 70/80
   y negación en 73/80.
4. Ajustar una temperatura en el dev rotado y publicar ECE, Brier y cobertura.
5. Abrir el corte sellado una sola vez cuando los puntos anteriores pasen.

