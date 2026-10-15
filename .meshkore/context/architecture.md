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
  `#T-real-sources`, `#T-parser-queue`.
- Servicio `com.decisiondata.worker` (launchd, KeepAlive). Cada ciclo descarga
  en paralelo las fuentes activas que tocan, normaliza en paralelo las que han
  cambiado, rehace el inventario y deja un latido en
  `work/world/worker-state.json`. Si no hay trabajo, espera diez minutos.
- Entrada: el catálogo `config/source-catalog.json` y el árbol
  `config/world-taxonomy.json`.
- Salida: corpus normalizado `work/world/corpus/<fuente>.jsonl.gz`, inventario,
  y los paquetes de episodios y el benchmark v3 que consume el entrenamiento.
- Regla de frontera: descargar no es admitir. Una fuente solo llega a
  entrenamiento tras revisión de licencia, deduplicación, firewall contra el
  corte sellado y una ganancia medida en v3.

## Entrenamiento: modelo y evaluación en `jev.clone`

- Módulos `model` y `eval`. Iniciativa `#decision-sota`; tareas
  `#T-autoloop`, `#T-benchmark-v3`, `#T-backbone-ladder`, `#T-devloop`.
- Servicios `com.jevclone.autoloop` (ciclos de ajuste, evaluación y
  promoción), `com.jevclone.devloop`, `com.jevclone.research` y el panel
  `com.jevclone.dashboard`.
- Entrada: los datasets publicados por la fábrica y el benchmark v3.
- Salida: un campeón promovido con adaptador, cabeza de puntero, informe v3 y
  hash. El corte sellado no se abre nunca; JEv solo mide.

## Ejecución: runtime en `jev.clone/crates`

- Módulo `runtime`. Iniciativa `#champion-runtime`; tareas
  `#T-champion-export`, `#T-candle-pointer`, `#T-runtime-parity-v3`.
- El runtime de septiembre (`#local-runtime`) sirve el cross-encoder antiguo
  con Candle. El campeón actual es TinyJev-4B con LoRA y cabeza de puntero en
  MLX, y todavía no se puede servir desde Rust. Cerrar esa distancia es el
  trabajo de este frente.
- Entrada: el último campeón promovido, exportado a safetensors.
- Salida: `jevclone serve` con el campeón, paridad numérica frente a MLX sobre
  la sonda v3 y latencias medidas.

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
- Entrenamiento: el autoloop encadena ciclos; cada candidato se puntúa en la
  sonda v3 y solo promueve si mejora.
- Ejecución: cada promoción del campeón dispara una exportación y una pasada
  de paridad.
