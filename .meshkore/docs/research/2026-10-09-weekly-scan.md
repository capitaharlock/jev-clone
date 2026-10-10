# Investigación semanal del 9 de octubre de 2026

Ventana principal: 26 de septiembre a 9 de octubre de 2026.
Task: `#T-evidence-cadence`.

## Veredicto

No hay evidencia pública que justifique abandonar ahora el Qwen con cabeza
puntero. La investigación sí cambia el benchmark: debe medir riesgo selectivo,
cobertura y calibración, no solo accuracy. También aparece una arena útil de
decisión multicriterio, LISTEN, que conviene adaptar como evaluación externa.

## JEv

- [Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
  sigue siendo la explicación primaria: preguntas tipadas, probabilidades,
  confianza y RLCD. Es una descripción de alto nivel, no una receta
  reproducible.
- [Models - TypeSafe AI](https://docs.typesafe.ai/models) publica
  `jev-1.13.0`, contexto de 64k, límites y precio. `jev-latest` y
  `jev-preview` apuntan a esa versión en la consulta.
- No se encontró en la ventana un paper técnico nuevo, pesos abiertos ni una
  implementación reproducible de RLCD.
- La explicación secundaria de Sanity del 6 de octubre no añade evidencia
  arquitectónica suficiente para cambiar el brazo.

Decisión: mantener la comparación remota de JEv como referencia medida cuando
el protocolo coincida y como cita cuando no. No inferir la arquitectura desde
marketing o documentación de API.

## Investigación reciente

### Abstención y control de riesgo

- [Unanimously Wrong: Certified Abstention from How Medical LLM Consensus Forms](https://arxiv.org/abs/2610.07570),
  enviado el 6 de octubre. ProbeGuard muestra que el consenso puede ser
  unánime y erróneo, y calibra abstención con Learn-then-Test.
- [Careful Judge: Safe and Efficient Human-AI Collaborative Decision Making](https://arxiv.org/abs/2610.09043),
  6 de octubre. CARE combina calibración adaptativa, escalado a humanos y
  aprendizaje de feedback con control de riesgo.
- [Rubric-Calibrated Preferences](https://arxiv.org/abs/2609.35739) combina
  torneos Bradley-Terry, criterios binarios e Item Response Theory para hacer
  comparables los scores entre preguntas.

Aplicación: añadir al benchmark curvas riesgo-cobertura, error condicionado a
aceptación e intervalos de confianza. La confianza del modelo no se considera
calibrada porque el propio modelo la produzca.

### Repositorios

- [EIT-NLP/Direct-Rank](https://github.com/EIT-NLP/Direct-Rank) informa que
  los rerankers directos superan a variantes con chain-of-thought y conservan
  mejor calibración y generalización fuera de dominio. Refuerza la ruta de
  salida directa del puntero.
- [rank_llm](https://github.com/david23131/rank_llm) ofrece evaluadores
  pointwise, pairwise y listwise, incluidos métodos basados en logits del
  primer token. Es candidato para contrastar métricas, no una dependencia
  necesaria.
- [listwise-crossencoder-reranking](https://github.com/matanf-healthee/listwise-crossencoder-reranking)
  compara un cross-encoder especializado con un reranker generalista y aporta
  construcción de datos, cinco seeds y evaluación holdout. Puede servir como
  referencia de protocolo.

La búsqueda no encontró un repositorio nuevo de las últimas dos semanas que
reemplace de forma reproducible la arquitectura actual. La CLI local de GitHub
no tenía sesión, así que commits, licencias y fechas deben verificarse antes de
adoptar código.

## Datasets

### Prioridad alta

- [LISTEN benchmark](https://huggingface.co/datasets/AdamJovine/LISTEN-benchmark):
  MIT, revisión `8ab430c3bb66`, 0,8 MB, cuatro escenarios de selección
  multicriterio y cinco rerankings humanos de shortlists. Es adecuado como
  arena externa de preferencias y comparación de productos. No es nuevo en la
  ventana: última modificación 29 de mayo de 2026.
- [MixReward](https://huggingface.co/datasets/SUSTech-NLP/MixReward):
  Apache-2.0, revisión `93b5c90974ba`, 64.528 pares, 103 idiomas y seis
  dominios. Puede aportar transferencia multilingual, pero sus preferencias
  juzgan respuestas generadas, no decisiones arbitrarias. Requiere piloto y
  medición de transferencia antes de entrar en train.

### No ingerir ahora

- [Innoduel RLHF](https://huggingface.co/datasets/NordosoftOy/innoduel-rlhf):
  1,5 millones de preferencias organizativas reales, pero acceso manual y
  licencia comercial `other`. Es relevante, pero no se descarga ni
  redistribuye sin acuerdo explícito.
- Los datasets de ranking de vídeo son multimodales y no encajan con la entrada
  textual del puntero actual.

## Acciones

1. Continuar el entrenamiento actual hasta la siguiente auditoría de 48 horas.
2. Añadir riesgo-cobertura y calibración de abstención al benchmark.
3. Preparar un adaptador eval-only para LISTEN con revisión humana de gold y
   protocolo fijado antes de puntuar.
4. Probar una muestra estratificada de MixReward; no mezclarla con train hasta
   demostrar transferencia a las cinco familias.
5. No integrar repositorios externos hasta verificar commit, licencia, tests y
   paridad en una rama aislada.
6. Preparar para publicación el contrato, validadores y datos sintéticos
   verificables; mantener fuera las baterías selladas y cualquier fuente que
   no permita redistribución.
