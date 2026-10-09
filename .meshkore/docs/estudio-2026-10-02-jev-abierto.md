---
title: Qué publicaron Jeff, Kev y AutoJev, y qué cambia en nuestro bucle
updated: 2026-10-02
owner: developer
status: active
task: T-aligned-ranking-v2
initiative: daily-learning-loop
---

# Qué publicaron Jeff, Kev y AutoJev

Estudio del 2 de octubre de 2026. Jev sigue cerrado. Desde el 17 de
septiembre han salido recetas abiertas que se miden contra él, y el patrón
que se repite no es el nuestro.

## Qué se sabe de Jev

TypeSafe no ha publicado pesos, tamaño ni la receta. El anuncio dice que Jev
devuelve todas las probabilidades en paralelo, sin generar texto, y llama al
entrenamiento RLCD (reinforcement learning for calibrated decisions). La
receta exacta no está publicada.

La sonda pública más cuidadosa es [Jev's Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked/)
(17 de septiembre, unos 10.000 llamadas a Jev 1.13.0). Lo que aguanta:

- La latencia crece con la entrada, no con el número de opciones. Un campo
  `output_tokens` cuenta la respuesta serializada, no tokens generados.
- El estado se comparte y las preguntas no se leen entre sí.
- Dentro de una pregunta las opciones se leen juntas. Añadir una opción
  irrelevante cambió el log-odds entre dos opciones ya presentes en los diez
  bloques (cambio medio -0,28). Un scorer que puntúa cada opción por separado
  y solo normaliza al final no puede hacer eso.
- Cabem dos lecturas: una cabeza sobre el token de decisión, o un puntero que
  compara ese estado con el de cada opción. Con 200 opciones los errores no
  saltaban al vecino, lo que favorece el puntero. Ninguna de las dos queda
  cerrada.
- La confianza del API no es la probabilidad. Para una elección con K>1 es
  `(p_max - 1/K) / (1 - 1/K)`.
- El orden de las opciones mueve la probabilidad. Una política con umbral
  cerca de 0,9 puede cambiar de acción con las mismas etiquetas.

## Las recetas que se acercan a Jev

Todas parten de un modelo causal ya entrenado (Qwen3.5 o Qwen3.8), hacen una
sola pasada y leen una distribución sobre las opciones. Ninguna entrena un
encoder de 280M desde una cabeza NLI.

| Proyecto | Fecha | Qué es | Entreno | Dónde queda respecto a Jev |
|---|---|---|---|---|
| [Kev](https://github.com/jaredpalmer/kev) | 17 sep | LoRA rango 16 más cabeza puntero sobre Qwen3.5-Base (0,8B, 4B, 9B) y Qwen3.8-27B | CE, 2 épocas. Conjunto `decision-v7`: 10.000 filas de diez datasets públicos, 896 políticas y 1.680 reglas. Temperatura ajustada después (4B: 2,41). No cambia el ganador | Fuentes nuevas, dev/test: 4B 0,817/0,838, 9B 0,822/0,852, 27B 0,848/0,896. Jev en ese dev: 0,857. El 0,8B se queda en 0,648/0,697 |
| [AutoJev-27B](https://github.com/denis-pplx/autojev) | 19 sep | Ajuste completo de Qwen3.8-27B | 73.000 ejemplos, 286 pasos, CE, temperatura aparte. El corpus no se publica | Panel de cinco benchmarks: 84,9, por encima del 83,0 publicado de Jev. Muestras distintas |
| [Jeff](https://github.com/firelex/jeff) v1.2 | 1 oct | Ajuste completo de Qwen3.5-0,8B, 2B y Gemma 4 E2B. Parte de AutoJev | 1 época, lotes de 256, CE sobre la letra de la opción, temperatura al final. 284.747 preguntas. El panel no elige checkpoint | Panel de 4.599: 0,8B 78,7, 2B 81,7, Gemma 81,6, frente al 83,0 publicado de Jev. JevBench hard: 44,8 / 57,1 / 48,6 frente a 73,3 |
| [SemIf](https://github.com/TheoLeeCJ/SemIf) | sep | Sin entreno: logits de la última posición, restringidos a las letras declaradas | Ninguno | Línea base. Las probabilidades salen sin calibrar |
| Tev1-4B (Together) | sep | Ajuste de Qwen3.5-4B | Unos 38.000 ejemplos (MultiNLI, BoolQ, Banking77, AG News, SST-5, políticas y enrutado). Coste publicado: 17 USD | Receta barata, sin tabla comparable a la de Kev |

Los números de Jev en esas tablas no son nuestra batería de 400 filas. Sirven
para ordenar recetas entre sí, no para decir que ya superan a `loop-0024`.

## Datos

Jeff publica la lista y no los textos (`docs/data-sources.md`). La base mezcla
datasets con etiqueta humana (BoolQ, SQuAD2, PAWS, MultiNLI, SNLI, MASSIVE,
CommonsenseQA, ARC, SocialIQA, CosmosQA, WinoGrande, RAGTruth, HaluEval,
VitaminC) con unas 31.000 preguntas sintéticas de un profesor abierto
(Qwen3.8-Flash-Next) y listas largas de 20 a 254 opciones (MASSIVE completo y
CLINC150). En v1.2 quitaron solapes con los tests, atajos de longitud, de
letra y de número de opciones. El acierto en documentos largos bajó de 82,8 a
66,1 al quitar la fuga de MAUD. Destilar el 2B en el 0,8B no movió ningún test.

Kev entrena con unas 12.600 filas y dos épocas. El salto de tamaño importa
más que el volumen: en fuentes nuevas el 0,8B se queda en 0,65 y el 4B pasa
de 0,82. Un ajuste posterior de dominio, partiendo del checkpoint publicado,
sube la tarea nueva sin tirar lo anterior. Partir de la base virgen, en una
prueba que publican, cayó a 0,33 en su propia evaluación.

Jeff además publica adaptadores LoRA de unos 41 MB, una época, una tarea
cada uno. En el test propio pasan de un 20-70 del base a 85-98, con ECE de
0,004 a 0,020. La emoción (27 clases) se queda en 60,6. El truco es el
adaptador de dominio, no más épocas del modelo general.

## Qué probaron y no se quedaron

- Kev documenta pérdidas de orden (`perm_kl`, `ord_w`). Los modelos publicados
  no las usan. Encaja con nuestros ciclos 68 a 73: peso pareado 0,3, media
  0,740, igual que el listwise puro.
- Jeff v1.2 publica el checkpoint del final de la época. El panel no selecciona.
- La temperatura se ajusta después, en un corte apartado. Mejora Brier y ECE.
  El argmax no cambia, así que no sube el acierto forzado.
- Jeff avisa de que un modelo pequeño no razona. En BBH, JudgeBench y
  JevBench hard se queda 16 a 30 puntos por debajo de Jev. Gana en
  clasificación y en anclaje a un pasaje (Financial PhraseBank, RAGTruth).

## Qué cambia aquí

Nuestro scorer listwise sobre mDeBERTa-v3 (280M, marcadores MASK, CE) es un
pariente del puntero de Laya y de Kev, pero el cuerpo no lo es. Llevamos el
ciclo 74 reiniciando desde `loop-0024` (0,7325) con miles de filas numéricas
generadas para parecerse a la batería. El holdout llega a 0,87 y dev no pasa
de 0,77. Las familias que fallan son comparación y prioridad.

Eso es el síntoma que Jeff midió al quitar la fuga, y el que Kev separa como
"fuentes entrenadas" frente a "fuentes nuevas". Más ciclos de la misma mezcla
no son el brazo que la literatura está usando para acercarse a Jev.

La primera prueba ya cambia la ruta. TinyJev 0,6B empató a `loop-0024` con
0,7325. TinyJev 4B v1, Qwen3-4B con LoRA y cabeza puntero, dio 0,9250 forced y
0,8571 joint. El gate pareado dio GO: +0,1925 forced, IC95 de 0,1475 a 0,2425,
y +0,2714 joint, IC95 de 0,1786 a 0,3571. No cayó ninguna familia ni idioma.

La medición inversa y cuatro permutaciones dieron entre 0,8875 y 0,9075. El
nuevo nivel no elimina la sensibilidad al orden, pero el peor pase sigue por
encima del objetivo. La ruta queda así:

1. TinyJev 4B v1 pasa a ser la arquitectura activa de investigación.
2. Auditar fuga y medir un dev rotado antes de ajustar el modelo.
3. Abrir un LoRA propio sólo si prioridad no llega a 0,75 en el dev rotado.
4. Ajustar temperatura fuera de la batería y abrir el sellado una sola vez.
