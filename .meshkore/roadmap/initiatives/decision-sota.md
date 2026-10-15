---
id: decision-sota
title: Llevar TinyJev al nivel de JEv con datos reales y un benchmark que no se sature
status: proposed
owner: architect-master
modules:
  - data
  - eval
  - model
created: 2026-10-15
updated: 2026-10-15
---

# Llevar TinyJev al nivel de JEv

## Diagnóstico del 15 de octubre

- La batería de 400 filas y el benchmark v2 están saturados. JEv sacó 400/400
  en la batería el 28 de septiembre; el campeón, 95,5 %; el modelo base sin
  ajustar, 92,5 %. En el v2 de 10.000 casos el campeón ya está en 99,09 %.
  Ninguno de los dos puede medir la distancia que importa.
- El modelo ha aprendido de muy pocos datos reales. Cada ciclo arranca desde el
  campeón del ciclo 5 y ve entre 2.400 y 14.000 decisiones. La mezcla pública
  tiene cinco fuentes y unas 40.000 filas; el pool verificado de Qwen, unas
  8.600; la fábrica v2, 12.430. Los 51 GB de episodios por regla son plantillas
  de cinco formas de tarea y casi no se usan.
- No hay productos con especificaciones reales, código, preguntas y respuestas
  de comunidades técnicas, literatura ni preferencias humanas a escala.
- El rendimiento medido en esta máquina es de unas 8.000 decisiones por hora.
  Un día da para 150.000 a 200.000; una semana, para algo más de un millón. El
  límite actual es el diseño de ciclos cortos, no el hardware.
- La comparativa con JEv, Qwen y los modelos externos sigue en la pestaña
  Evaluation, pero está congelada en la batería de 400 filas. Queda en espera
  hasta que exista el benchmark v3.
- El proxy de esta máquina bloquea TypeSafe y también huggingface.co (503).
  Las mediciones de JEv se hacen en otro equipo y vuelven por Git, como en
  `#T-teacher-probe`. Los datos se bajan de su host de origen: GitHub,
  archive.org, Gutenberg, Open Food Facts.

## Estado al cierre del 15 de octubre

- v3 alfa en la fábrica: 18.710 casos de desarrollo, 6.316 sellados, sonda de
  1.350; nueve arenas y 87 dominios. Fuentes: MMLU, ARC, ANLI, GSM8K,
  HumanEval, MBPP, 20 webs de Stack Exchange y Open Food Facts.
- El autoloop mide v3 desde el ciclo 31: el campeón en todo el desarrollo una
  vez por versión, cada candidato en la sonda. El panel lo muestra en la
  tarjeta del campeón y en Evaluation.
- Pools de entrenamiento separados por entidad: 153.750 decisiones reales,
  pendientes de revisión de licencia y firewall antes de entrar en la mezcla.
- Pendiente: Amazon Reviews y Project CodeNet no respondieron en sus URL;
  falta la arena de velocidad de código, la de reseñas y la de literatura.

## Qué tiene que saber hacer el modelo

Un modelo de decisión elige entre opciones a partir del contexto. Si recibe las
fichas de dos secadores, compara potencia, peso, ruido y precio con el criterio
de la pregunta. Si solo recibe los nombres, depende de lo que el backbone de 4B
aprendió en el preentrenamiento, y eso es débil para productos concretos. El
ajuste no le enseña el catálogo del mundo: le enseña a leer evidencia, aplicar
criterios, ponderar varios y abstenerse cuando la evidencia no alcanza. Para
eso necesita cientos de miles de decisiones reales y variadas, con gold que no
dependa de la opinión de otro modelo.

## Benchmark v3, el definitivo

El benchmark se construye antes de ampliar el entrenamiento y no se toca
después de ver resultados.

- Fuentes reales con split por fuente o por entidad, nunca por fila: un
  producto, repositorio, pregunta o libro que aparece en evaluación no aparece
  en entrenamiento.
- Doce arenas, al menos 1.000 casos cada una, en inglés y español:
  1. productos con fichas técnicas;
  2. preferencia de producto según reseñas;
  3. corrección de código por ejecución;
  4. velocidad de código por ejecución medida;
  5. calidad de respuestas técnicas;
  6. conocimiento multidisciplinar;
  7. ciencia y razonamiento;
  8. matemáticas;
  9. lectura, literatura y estilo;
  10. inferencia, negación y verificación de hechos;
  11. decisión multicriterio y preferencia humana;
  12. abstención: casos sin respuesta correcta entre las opciones.
- 30.000 casos de desarrollo y 10.000 sellados.
- Un benchmark solo es aceptable si deja margen: JEv, Qwen 35B, el modelo base
  y el campeón se miden sobre las mismas filas, y ninguna arena puede tener al
  campeón por encima del 90 % si JEv no está también cerca del techo.
- Cada informe trae intervalo de confianza, resultados por arena, por idioma y
  por número de opciones, y curva riesgo-cobertura.

Cuando v3 esté publicado, la comparativa con JEv y los modelos externos vuelve
al panel con los números nuevos, y la batería de 400 filas pasa a control de
regresión.

## Datos de entrenamiento

Objetivo: un millón de decisiones verificables en ocho semanas, con no más del
15 % de una sola fuente y gold por programa, ejecución o etiqueta humana de la
fuente. Ninguna fuente entra sin revisión de licencia, deduplicación y firewall
contra v3.

| Familia | Fuentes candidatas | Gold | Objetivo |
|---|---|---|---|
| Productos con especificaciones | Open Food Facts, Open Icecat, fichas de fabricantes con licencia | Comparación programática sobre atributos | 150.000 |
| Reseñas y preferencia de producto | Amazon Reviews 2023, LISTEN | Valoración y utilidad de la reseña, con margen mínimo | 100.000 |
| Corrección de código | Project CodeNet, MBPP, soluciones generadas con tests | Ejecución en sandbox | 120.000 |
| Velocidad y calidad de código | Project CodeNet, variantes generadas | Tiempo medido en sandbox, repeticiones y margen | 80.000 |
| Respuestas técnicas | Volcado de Stack Exchange, CC BY-SA | Respuesta aceptada y votos, con margen | 150.000 |
| Preferencia general | HelpSteer2 y 3, UltraFeedback, SHP | Etiqueta humana de la fuente | 120.000 |
| Conocimiento y ciencia | ARC, OpenBookQA, SciQ, MedMCQA, GSM8K, MATH | Respuesta de la fuente | 150.000 |
| Literatura y estilo | Proyecto Gutenberg | Metadatos: autor, época, género | 60.000 |
| Inferencia y hechos | MNLI, ANLI, FEVER, VitaminC, XNLI | Etiqueta de la fuente | 70.000 |

Qwen solo reescribe enunciados y produce distractores; nunca decide el gold.
Las tablas de licencias quedan en `config/` de la fábrica con revisión fijada y
hash, como ya hacen GSM8K y OpenBookQA.

### JEv como profesor

Con la API se pueden pedir las probabilidades de JEv sobre 100.000 a 300.000
decisiones y usarlas como objetivos suaves. Antes hay que confirmar dos cosas:
que las condiciones de TypeSafe permiten entrenar con sus salidas y el
presupuesto disponible. Hasta entonces JEv solo mide.

## Entrenamiento

- Los ciclos cortos desde el campeón sirven como control y se mantienen, pero
  la ganancia grande tiene que venir de una ejecución larga y continua: varios
  días sobre la mezcla nueva, checkpoint cada 2.000 pasos y evaluación en una
  muestra fija de v3.
- Se compara el LoRA de rango 16 contra rango 64 y contra un backbone mayor si
  el de 4B se queda corto en conocimiento.
- Se promueve por v3 con intervalo de confianza y sin regresión en la batería
  de 400 filas. Un candidato que empata dentro del ruido no se promueve.
- Si una semana de datos nuevos no mueve v3 fuera del ruido, la auditoría
  cambia de hipótesis: backbone, formato o calidad del gold.

## Fases

| Semana | Trabajo | Salida comprobable |
|---|---|---|
| 1 | Sonda de 200 casos a mano: secadores, funciones, respuestas técnicas | Primera lista de fallos reales |
| 1 y 2 | Benchmark v3 y medición de JEv, Qwen, base y campeón | Margen medido por arena |
| 2 a 4 | Adaptadores de las nueve familias de datos, licencias y firewall | 300.000 decisiones verificadas |
| 3 a 6 | Ejecución larga y comparación de rango y backbone | Curva v3 por checkpoint |
| 4 a 8 | Escala a un millón, minado de errores por arena, destilación de JEv si se aprueba | Distancia a JEv por arena |
| 8 | Abrir el corte sellado una vez | Resultado publicable |

## Hecho cuando

- v3 existe, está fijado por hash y deja margen medido en todas las arenas.
- JEv y el campeón están medidos sobre las mismas filas de v3.
- El campeón queda dentro de 3 puntos de JEv en la media de v3 y no más de 8
  por debajo en ninguna arena, confirmado en el corte sellado.
