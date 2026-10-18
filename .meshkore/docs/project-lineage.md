# Linaje del proyecto

## Producto principal

El objetivo es demostrar que podemos diseñar, entrenar y servir un modelo de
decisión propio. La arquitectura publicable, el tokenizer y todos sus pesos
empiezan aquí. El repositorio debe permitir reproducir el entrenamiento y la
inferencia sin descargar checkpoints de terceros.

## Qué significa propio

- El diseño de la red y el código de entrenamiento son nuestros.
- Todos los pesos publicables parten de inicialización aleatoria.
- El tokenizer se ajusta con corpus admitido por el proyecto.
- Los datasets llevan fuente, licencia, transformación y hashes.
- El runtime carga el formato propio y no depende de otro modelo.

Las bibliotecas generales de cálculo, serialización y tokenización siguen
siendo dependencias normales de software.

## Referencias y profesores locales

JEv se usa para medir. No tenemos su código fuente ni sus pesos, y sus
respuestas no entran en el corte sellado.

TinyJev se clonó localmente y sus pesos se descargaron para estudiar una
arquitectura de decisión abierta y obtener una referencia fuerte. El autoloop
entrenó adaptadores LoRA sobre TinyJev-4B. Ese trabajo logró buenos resultados
en las baterías antiguas y se conserva como laboratorio y posible producto
derivado sujeto a licencia. No es el modelo propio.

Qwen y otros modelos pueden generar, verificar o etiquetar datos. Sus pesos y
su código no forman parte del estudiante publicado.

## Historia técnica

El proyecto empezó como una investigación inspirada por JEv. Construyó
esquemas de episodios, una cabeza de decisión, scorers cross-encoder y
listwise, evaluación, gates y un runtime Rust. El camino propio llegó a
0,7325 en la batería antigua, pero falló la generalización a etiquetas no
vistas.

El 2 de octubre de 2026 se cambió el brazo operativo a TinyJev-4B porque
obtuvo 0,9250 sin ajuste en esa batería. El campeón posterior llegó a 0,9553,
pero el benchmark v3 mostró alrededor de 0,62 y el bucle acumuló más de 65
ciclos sin promoción.

El 18 de octubre de 2026 se corrigió el objetivo: TinyJev queda aparcado como
referencia local. El producto vuelve a una red propia entrenada desde cero,
con una escalera de capacidad que empieza en unos 15 M de parámetros.

## Publicación

El repositorio de laboratorio puede conservar clientes de evaluación,
entrenadores de referencias y documentos históricos. El repositorio o corte
de release del modelo contiene solo código propio, pesos propios y las
licencias de sus dependencias de software y datos.
