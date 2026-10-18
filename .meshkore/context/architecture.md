# Tres frentes: datos, entrenamiento y ejecución

El proyecto avanza por tres frentes separados. Cada uno tiene su repositorio o
módulo, su iniciativa, su servicio en marcha continua y un contrato de entrada
y salida con los otros dos. Ningún frente espera a otro para seguir
trabajando: lo que consume es siempre el último artefacto publicado, no una
copia en curso.

La fábrica de datos publica datasets versionados; el entrenamiento los
consume y promueve un campeón; el runtime sirve ese campeón.

## Datos: `decision-dataset-factory`

- Repositorio propio en `~/Prj/asimovia/decision-dataset-factory`, privado.
- Iniciativa `#decision-dataset-factory`; tareas `#T-world-corpus-engine`,
  `#T-real-sources`, `#T-parser-queue`, `#T-source-corpus-projections` y
  `#T-frontier-data-map`.
- Servicio `com.decisiondata.worker` (launchd, KeepAlive). Cada ciclo descarga
  en paralelo las fuentes activas que tocan, normaliza en paralelo las que han
  cambiado, rehace el inventario y deja un latido en
  `work/world/worker-state.json`. Si no hay trabajo, espera diez minutos.
- Entrada: el catálogo `config/source-catalog.json` y el árbol
  `config/world-taxonomy.json`.
- Salida base: corpus neutral `source-document-v1` en
  `work/world/corpus/<fuente>.jsonl.gz`, inventario y un manifiesto con
  `corpus_id`. Los datasets de decisión son proyecciones que fijan ese ID y su
  receta. Una futura proyección para un modelo de lenguaje reutiliza los mismos
  documentos y raw locales.
- Regla de frontera: descargar no es admitir. Una fuente solo llega a
  entrenamiento tras revisión de licencia, deduplicación, firewall contra el
  corte sellado y una ganancia medida en v3.

## Entrenamiento: modelo y evaluación en `jev.clone`

- Módulos `model` y `eval`. Iniciativa `#own-decision-model`; tarea principal
  `#T-own-backbone`.
- El estudiante empieza desde pesos aleatorios. Los modelos externos solo
  generan, verifican, enseñan o sirven como referencias locales.
- El autoloop TinyJev queda parado y conservado como laboratorio. El siguiente
  bucle continuo se activa cuando el trainer propio supere sus gates de
  aprendizaje y linaje.
- Entrada: los datasets publicados por la fábrica y el benchmark v3.
- Salida: un checkpoint propio con tokenizer, configuración, informe v3 y
  hashes. El corte sellado no se abre nunca; JEv solo mide.

## Ejecución: runtime en `jev.clone/crates`

- Módulo `runtime`. El trabajo Qwen/TinyJev de `#champion-runtime` queda
  aparcado. El runtime propio existente en `crates/jev-model` se conecta al
  formato del nuevo backbone.
- Entrada: el último campeón promovido, exportado a safetensors.
- Salida: un servicio con el checkpoint propio, paridad numérica frente a la
  referencia Python y latencias medidas.

## Publicación del modelo

El repositorio que publique el modelo lleva solo código nuestro: la red, los
pesos y el código que la ejecuta. Ningún modelo ajeno entra ahí, ni como
código, ni como pesos de base, ni como dependencia de la inferencia.

Usar un modelo ajeno para enseñar, aprender o comprobar se queda en esta
máquina. El clon de TinyJev, su caché de pesos y los adaptadores entrenados
encima no se suben. `jev.clone` es el laboratorio. El modelo sale de aquí
cuando la red que puntúa sea nuestra y todos sus pesos hayan empezado desde
inicialización aleatoria.

## Reparto de recursos de la máquina

| Recurso | Dueño | Regla |
| --- | --- | --- |
| GPU (Metal) | entrenamiento y evaluación | Ningún proceso de datos carga un modelo. El runtime solo usa la GPU en las ventanas de paridad. |
| CPU, 18 núcleos | datos | El worker usa `núcleos - 6` procesos para normalizar, con `nice 10`, y deja el resto al entrenamiento y al panel. |
| Red | datos | Cuatro descargas simultáneas, reanudables, con cuota por fuente. |
| Disco | datos | Techo de 250 GB para el corpus, 5 GB por fuente y 150 GB siempre libres. |

## Trabajo continuo

Cada frente tiene una cola que no se vacía:

- Datos: la cola de parsers y fuentes de `#T-parser-queue`. Cuando una fuente
  queda activa y normalizada, se toma la siguiente.
- Entrenamiento: la escalera propia avanza de 15 M a 60 M y después escala
  solo cuando la capacidad medida lo pide. Cada candidato se puntúa en v3.
- Ejecución: cada promoción del campeón dispara una exportación y una pasada
  de paridad.
