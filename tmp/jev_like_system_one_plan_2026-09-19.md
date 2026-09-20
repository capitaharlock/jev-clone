# Plan maestro: modelo local tipo Jev / System‑One
## Arquitectura no autoregresiva, decisiones tipadas, probabilidades calibradas y latencia local mínima

**Fecha de revisión:** 19 de septiembre de 2026  
**Objetivo del documento:** convertir la idea de un “Jev abierto” en un proyecto ejecutable: arquitectura, datos, entrenamiento, evaluación, optimización, hardware, backlog y criterios de decisión.

---

## 0. Resumen ejecutivo

La recomendación es **no intentar clonar Jev literalmente** ni entrenar un foundation model desde cero. Tampoco conviene construir el producto final como un wrapper de Qwen/Gemma/Llama, porque aunque eliminemos la generación token-a-token, seguimos pagando el coste del backbone grande.

La dirección con mejor relación riesgo/resultado es:

1. **Reutilizar conocimiento de un encoder moderno preentrenado**.
2. **Separar físicamente el encoding del `state` del procesamiento de las preguntas**.
3. Codificar el `state` **una sola vez**.
4. Representar preguntas y opciones dinámicas como consultas.
5. Hacer sólo **1–3 capas ligeras de cross-attention** desde preguntas/opciones hacia la memoria del estado.
6. Permitir interacción **listwise** entre opciones de una misma pregunta.
7. Puntuar opciones con un **pointer/dynamic-label head**, no con un classifier de clases fijas.
8. Producir probabilidades directamente; **cero decoding autoregresivo**.
9. Entrenar primero con supervised learning + proper scoring rules + distillation.
10. Añadir abstención, calibración, early exit, quantization y cascada a un LLM únicamente para los casos difíciles.

La primera decisión que **no** debe darse por hecha es el backbone. En septiembre de 2026 hay opciones más interesantes que el ModernBERT original:

- **LFM2.5-Encoder 230M / 350M** — julio de 2026, encoder híbrido atención + convolución corta, 8K, 15 idiomas, orientado explícitamente a baja latencia y on-device.
- **Ettin Encoder** — 17M, 32M, 68M, 150M, 400M y 1B; licencia MIT y datos/checkpoints abiertos.
- **NeoBERT 250M** — MIT, 4K, encoder moderno compacto.
- **ModernBERT 149M / 395M** — Apache 2.0, 8K, muy maduro y fácil de modificar.
- **mmBERT** — MIT y multilingüe masivo; interesante si español/multilingüe importa desde el principio.

**Mi hipótesis inicial:** comenzar con **Ettin-68M/150M + ModernBERT-base + LFM2.5-230M** como carrera de backbones. No elegir ninguno hasta medirlos con nuestro propio head y nuestro propio hardware.

El principal reto **no será escribir el código del modelo**. Los problemas difíciles serán:

- conservar inteligencia/generalización en un modelo pequeño;
- construir datos diversos sin contaminar benchmarks;
- generar hard negatives y `unknown` creíbles;
- obtener probabilidades realmente calibradas fuera de distribución;
- impedir que el modelo memorice artefactos de datasets;
- mantener licencia comercial limpia;
- hacer que la ventaja de “state encode once” sea real en el runtime;
- optimizar kernels, formas dinámicas y quantization en Apple Silicon y CUDA.

---

# 1. Objetivo del producto

Queremos construir un **motor de decisión semántico local**.

No queremos un chatbot.

Entrada conceptual:

```text
STATE
+
N QUESTIONS
+
dynamic OPTIONS per question
```

Salida:

```text
probability distribution per question
```

Ejemplo:

```json
{
  "state": "Customer says the transfer was charged twice...",
  "questions": [
    {
      "id": "intent",
      "type": "choice",
      "prompt": "What is the primary intent?",
      "options": [
        {"id": "duplicate_transfer", "text": "Duplicate bank transfer"},
        {"id": "cash_withdrawal", "text": "Cash withdrawal issue"},
        {"id": "card_payment", "text": "Card payment issue"}
      ]
    },
    {
      "id": "urgent",
      "type": "boolean",
      "prompt": "Does this need urgent intervention?"
    }
  ]
}
```

Salida:

```json
{
  "intent": {
    "duplicate_transfer": 0.962,
    "cash_withdrawal": 0.008,
    "card_payment": 0.030
  },
  "urgent": {
    "true": 0.74,
    "false": 0.26
  }
}
```

No hay texto generado. No existe:

```text
"The customer appears to have..."
```

salvo que una aplicación posterior quiera convertir el resultado a texto.

---

# 2. Qué queremos optimizar

Orden recomendado de prioridades:

1. **Calidad semántica útil**
2. **Calibración**
3. **Latencia single-request**
4. **Escalado con muchas preguntas sobre el mismo estado**
5. **Abstención/OOD**
6. **Memoria**
7. **Throughput**
8. **Multilingüe**
9. **Long context**
10. **Tamaño del checkpoint**

El error más fácil sería obsesionarse con `5 ms` y terminar con un classifier rápido pero incapaz de generalizar.

## 2.1 Targets de investigación

No son promesas; son objetivos para decidir si continuamos.

### V0 — smoke test

- 2–42 opciones dinámicas.
- `choice` y `boolean`.
- accuracy competitiva en datasets sencillos.
- probabilidades correctas.
- inferencia sin generación.
- Apple MPS + CUDA.

### V1 — útil

Objetivos de ingeniería:

- `choice`, `boolean`, `score`.
- `state` codificado una vez.
- 1–50 preguntas por state.
- opciones nunca vistas durante training.
- explícito `unknown/insufficient evidence`.
- p50 local < 50 ms en una pregunta corta en al menos uno de nuestros equipos.
- 10 preguntas claramente más baratas que 10 ejecuciones independientes.
- ECE in-domain < 0.05 como objetivo inicial.
- fuerte curva risk/coverage con abstención.

### V2 — interesante frente a Jev-like existentes

- p50 ideal < 20 ms para contexto corto.
- state caching real.
- escalado sublineal al aumentar `Q`.
- buena transferencia a familias de tareas no vistas.
- cuantización INT8 sin degradación material.
- funcionamiento español + inglés.
- fallback a LLM sólo en fracción pequeña de casos.

---

# 3. Qué sabemos y qué no sabemos de Jev

Fuente principal:

- TypeSafe — *Introducing System One Models and Jev*:  
  https://typesafe.ai/blog/introducing-system-one-models-and-jev
- TypeSafe:  
  https://typesafe.ai/

Lo público indica:

- modelo orientado a decisiones, no a generación de strings;
- sampler paralelo;
- salidas tipadas/probabilísticas;
- método de training denominado **RLCD — Reinforcement Learning for Calibrated Decisions**;
- gran énfasis en latencia/coste y calibración.

Lo que **no** está publicado suficientemente como para reproducirlo:

- arquitectura exacta;
- número de parámetros;
- mezcla de datos;
- objective exacto de RLCD;
- routing/MoE si existiese;
- estructura interna del sampler;
- distillation/teacher pipeline;
- optimizaciones específicas de serving.

Por tanto, debemos tratar cualquier “reverse engineering” como hipótesis, no como especificación.

---

# 4. Panorama de implementaciones Jev-like

Estas implementaciones sirven para aprender, benchmarkear y robar buenas ideas arquitectónicas. No asumir que ninguna sea “el clon correcto”.

## 4.1 Laya

- Modelo: https://huggingface.co/convaiinnovations/laya
- Repo: https://github.com/NandhaKishorM/laya

Ideas valiosas:

- encoder bidireccional;
- opciones marcadas en la entrada;
- output directo;
- proper-scoring-oriented training;
- evaluación explícita de calibración;
- no decoding.

Problema para nuestro objetivo:

- si cada pregunta incluye de nuevo el estado en la secuencia, un batch de preguntas no equivale a **state compute once**;
- debemos medirlo y no asumir ventaja por el mero hecho de hacer un único kernel launch/batch.

## 4.2 Verdict / OpenJev

- Modelo: https://huggingface.co/heman10x/rlcd-modernbert-151m
- Repo indicado por el autor: https://github.com/Heman10x-NGU/Verdict-open-jev

Ideas valiosas:

- ModernBERT/GLiClass;
- candidatos dinámicos;
- abstención;
- CE + Brier;
- temperature scaling;
- export ONNX/WebGPU;
- modelo pequeño.

Limitación:

- resultados publicados principalmente sobre intent classification; todavía no demuestran generalidad tipo foundation decision model.

## 4.3 `kev`

- Repo: https://github.com/jaredpalmer/kev

Probablemente el repo más valioso **arquitectónicamente**, aunque usa un backbone Qwen pequeño.

Ideas que debemos replicar sin Qwen:

- state/prefix compartido;
- preguntas aisladas entre sí;
- block/branch attention;
- un único prefill;
- pointer-style decision head;
- zero decoding;
- calibración posterior.

Nos interesa más su **estructura de atención** que su backbone.

## 4.4 `jevlike`

- Repo: https://github.com/vinnylarouge/jevlike

Valioso porque demuestra el esqueleto mínimo:

- context encoder;
- option/query representation;
- scoring compartido;
- softmax;
- CPU/MPS/CUDA;
- arquitectura pequeña.

Puede ser nuestro **baseline mínimo from-scratch**, no el modelo final.

## 4.5 `open-jev`

- Repo: https://github.com/daseinlabs/open-jev

Usa un LLM y elimina decoding mediante scoring de opciones/prefix caching.

Útil para:

- upper bound semántico;
- comparar “gran backbone sin decoding” contra nuestro encoder pequeño;
- estudiar KV/prefix reuse.

No es nuestra solución final por coste del backbone.

## 4.6 NanoJev

- Repo: https://github.com/TianyuCodings/NanoJev

Interesante para:

- datasets/formato;
- decisiones paralelas;
- experiments pequeños;
- comparar calibration recipes.

De nuevo, no debemos heredar automáticamente su backbone.

## 4.7 openjev-sglang

- Repo: https://github.com/ekzhang/openjev-sglang

Útil para serving ideas:

- radix/prefix caching;
- batching;
- prefill-only;
- API compatibility.

No es objetivo local: usa infraestructura/modelos mucho mayores.

## 4.8 Índice vivo de alternativas

- https://systemonemodels.org/examples/alternatives/

Crear un job/manual checklist para revisar este índice durante el proyecto; el espacio está evolucionando muy rápido.

---

# 5. Carrera de backbones: no casarnos con ModernBERT

Ésta es una mejora importante respecto a la primera idea.

## 5.1 LFM2.5-Encoder 230M / 350M — candidato prioritario

- Blog, julio 2026:  
  https://www.liquid.ai/blog/lfm2-5-encoders
- 230M:  
  https://huggingface.co/LiquidAI/LFM2.5-Encoder-230M
- 350M:  
  https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M

Características relevantes:

- bidireccional;
- 8,192 tokens;
- 15 idiomas, incluido español;
- arquitectura híbrida que intercala convoluciones cortas gated con atención;
- pensado explícitamente para clasificación, NLI, retrieval y uso on-device;
- 230M es un tamaño razonable para nuestros Macs.

**Riesgo:** usa LFM Open License v1.0, no una licencia permisiva clásica MIT/Apache. Antes de convertirlo en base del producto comercial hay que revisar la licencia completa.

### Hipótesis

La convolución/hybrid backbone podría ser especialmente atractiva cuando el `state` crece, porque no todas las capas pagan self-attention global.

---

## 5.2 Ettin — candidato prioritario y probablemente el mejor laboratorio

- HF 150M: https://huggingface.co/jhu-clsp/ettin-encoder-150m
- Colección/modelos: https://huggingface.co/models?other=ettin
- Blog: https://huggingface.co/blog/ettin

Tamaños:

- 17M
- 32M
- 68M
- 150M
- 400M
- 1B

Ventajas:

- MIT;
- datos de pretraining abiertos;
- muchos checkpoints intermedios;
- arquitectura moderna;
- permite estudiar **quality/latency scaling** con la misma familia.

Esto es ideal para responder científicamente:

```text
¿cuántos parámetros necesitamos realmente?
```

Podemos entrenar exactamente el mismo decision head encima de:

```text
17M → 32M → 68M → 150M → 400M
```

y obtener una frontera Pareto real.

### Recomendación

Usar **68M y 150M** desde el primer sprint.

17M/32M sirven para medir el suelo de latencia.

400M sólo entra si la calidad de 150M deja demasiado gap.

---

## 5.3 NeoBERT 250M

- Modelo: https://huggingface.co/chandar-lab/NeoBERT
- Paper: https://arxiv.org/abs/2502.19587

Características:

- 250M;
- MIT;
- 4,096 tokens;
- RoPE;
- SwiGLU;
- Pre-RMSNorm;
- FlashAttention;
- entrenado sobre ~2.1T tokens según model card.

Es un buen candidato si buscamos una arquitectura limpia y permisiva con un tamaño intermedio.

---

## 5.4 ModernBERT

- Base 149M: https://huggingface.co/answerdotai/ModernBERT-base
- Large 395M: https://huggingface.co/answerdotai/ModernBERT-large
- Repo/FlexBERT: https://github.com/answerdotai/modernbert
- Paper ACL 2025: https://aclanthology.org/2025.acl-long.127/

Ventajas:

- Apache 2.0;
- 8,192 tokens;
- maduro;
- soporte Transformers;
- RoPE;
- local/global alternating attention;
- FlashAttention/unpadding;
- buena documentación;
- FlexBERT facilita experiments de arquitectura.

Sigue siendo probablemente nuestro **baseline más seguro**.

---

## 5.5 mmBERT

- https://huggingface.co/jhu-clsp/mmBERT-base
- Repo enlazado por model card: https://github.com/jhu-clsp/mmBERT
- Paper: https://arxiv.org/abs/2509.06888

Ventajas:

- MIT;
- >1,800 idiomas según model card;
- datos de training abiertos;
- arquitectura ModernBERT;
- útil si queremos español/internacional desde V1.

No lo usaría para el primer benchmark si el modelo es sensiblemente mayor que Ettin/ModernBERT-base, pero sí como **multilingual quality baseline**.

---

# 6. Experimento cero: seleccionar backbone antes de construir demasiado

No comparar modelos con sus heads nativos. Compararlos **con exactamente nuestra misma tarea y head**.

## 6.1 Backbones mínimos

```text
Ettin-32M
Ettin-68M
Ettin-150M
ModernBERT-base 149M
NeoBERT 250M
LFM2.5-Encoder-230M
```

Opcional:

```text
ModernBERT-large 395M
Ettin-400M
LFM2.5-Encoder-350M
mmBERT
```

## 6.2 Dataset de selección

Primera pasada:

- HuffPost
- Banking77
- BoolQ
- Civil Comments subset
- HelpSteer2 subset

Así obligamos al backbone a demostrar:

- multiclass;
- dynamic labels;
- boolean reasoning;
- score/binary semantics;
- preference/quality.

## 6.3 Mismo training budget

Ejemplo:

- 1 epoch warmup dataset mix;
- mismo tokenizer budget cuando sea posible;
- mismo decision head;
- misma seed set;
- misma cantidad de updates;
- misma precision;
- mismo max state length.

## 6.4 Métricas

No seleccionar sólo por accuracy.

Score compuesto interno:

```text
quality:
  macro accuracy / F1
  NLL
  Brier
  zero-shot task family transfer

systems:
  p50
  p95
  peak memory
  1Q latency
  10Q latency
  50Q latency
  long-state latency
```

No convertir este score en una métrica pública única; conservar el Pareto.

---

# 7. Arquitectura recomendada

## 7.1 Principio central: separar STATE de QUESTION

Este es el punto más importante del diseño.

Un encoder bidireccional normal:

```text
[state + question + options] → encoder
```

**no permite cachear de verdad el state**, porque las representaciones del state cambian al poder atender bidireccionalmente a la pregunta y opciones.

Por tanto:

```text
QUESTION 1: encoder(state + q1 + opts)
QUESTION 2: encoder(state + q2 + opts)
QUESTION 3: encoder(state + q3 + opts)
```

repite gran parte del compute.

Aunque lo metamos en un batch, no hemos solucionado la raíz.

### Diseño recomendado

```text
                    ┌─────────────────────┐
STATE ─────────────►│  STATE ENCODER E_s  │
                    └──────────┬──────────┘
                               │
                      state memory S
                      computed ONCE
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
      Question 1            Question 2           Question N
       + options             + options            + options
          │                    │                    │
          ▼                    ▼                    ▼
       E_q / embeds          E_q / embeds         E_q / embeds
          │                    │                    │
          └───── shallow cross-attention to S ─────┘
                               │
                         option mixer
                               │
                         pointer scores
                               │
                    calibrated probabilities
```

El state puede reutilizarse durante:

- múltiples preguntas en una request;
- múltiples turns si el state no cambia;
- varios módulos/agentes que consultan la misma memoria;
- re-scoring con diferentes option sets.

---

# 8. State encoder

## 8.1 V1

Usar el backbone ganador como `E_s`.

Entrada:

```text
state text / serialized JSON / messages
```

Salida:

```text
S ∈ R^(Ls × d)
```

donde `Ls` es longitud del state.

Conservar token-level memory, no sólo `[CLS]`.

Razón: un único vector destruye demasiado detalle para QA/NLI/reasoning.

## 8.2 Pool global adicional

Añadir:

```text
s_global = attention_pool(S)
```

Útil para:

- routing;
- gating;
- early exit;
- OOD;
- bi-encoder fast path.

## 8.3 Cache key

Runtime:

```text
state_hash
model_version
tokenizer_version
max_length
```

No mezclar caches de checkpoints incompatibles.

---

# 9. Question / option representation

Cada pregunta contiene:

```text
prompt
type
options[]
optional descriptions
optional metadata
```

No usar IDs numéricos como semántica.

Mala opción:

```text
class_17
```

Buena opción:

```text
{
  "id": "duplicate_card_payment",
  "text": "Duplicate card payment",
  "description": "The same card purchase appears more than once."
}
```

## 9.1 Option encoder

Dos posibilidades:

### A — encoder compartido

Mismos pesos base para state y questions.

Ventajas:

- conocimiento alineado;
- menos parámetros.

### B — question encoder ligero

State encoder grande, questions con 2–6 capas.

Ventajas:

- questions son cortas;
- menos latencia por branch;
- permite cachear option labels.

**Recomendación:** empezar compartiendo embeddings/tokenizer, pero probar `E_q` reducido.

---

# 10. Cross-attention superficial

Cada question branch debería pagar sólo un pequeño coste sobre el state:

```text
Q/O hidden
   │
cross-attn → STATE MEMORY
   │
FFN
   │
cross-attn → STATE MEMORY
   │
FFN
```

Comenzar con:

```text
1 layer
2 layers
3 layers
```

No 12–24.

Pregunta experimental clave:

> ¿Cuántas capas de fusión necesitamos para recuperar la calidad de un cross-encoder completo?

Esta será probablemente una de las curvas más importantes del proyecto.

---

# 11. State memory compression

Para states largos, incluso cross-attention superficial sobre 8K tokens puede ser caro.

Introducir sólo después de tener baseline:

```text
S: 8192 token vectors
     ↓
learned latent compressor
     ↓
M: 32 / 64 / 128 / 256 memory vectors
```

Inspiración:

- Perceiver-like latent bottlenecks;
- learned pooling;
- token pruning;
- hierarchical pooling.

Experimentos:

```text
raw state tokens
64 latents
128 latents
256 latents
```

Medir calidad específicamente en tareas que necesiten detalles raros del contexto.

No usar compresión agresiva como default hasta demostrar que no destruye recall.

---

# 12. Dynamic option head

No usar:

```python
Linear(hidden_size, 77)
```

porque las clases quedarían fijadas.

Necesitamos labels arbitrarios.

## 12.1 Pointer-style scorer

Para cada opción:

```text
o_i = contextualized option representation
q   = question/decision representation
```

Score:

```text
z_i = qᵀ W o_i
```

o:

```text
z_i = MLP([q, o_i, q*o_i, |q-o_i|])
```

Luego:

```text
p = softmax(z)
```

Referencia fundamental:

- Pointer Networks: https://arxiv.org/abs/1506.03134

No necesitamos copiar Pointer Networks; necesitamos el principio de **output vocabulary dinámico**.

---

# 13. Interacción entre opciones: listwise, no independientes

Una opción puede cambiar el significado de otra.

Ejemplo:

```text
A = card payment
B = transfer
C = other
```

`other` sólo tiene sentido relativo al conjunto.

Por tanto, después de obtener option embeddings:

```text
[o1, o2, o3, ...]
       ↓
option mixer
       ↓
contextualized options
```

## 13.1 Primera implementación

Un pequeño Transformer de 1–2 capas sobre las opciones.

Como `K` suele ser pequeño, el coste es insignificante.

## 13.2 Alternativa barata

Deep Sets:

```text
g = sum(phi(o_i))
z_i = rho(o_i, g, q, state_global)
```

Referencia:

- Deep Sets: https://arxiv.org/abs/1703.06114

## 13.3 Alternativa más expresiva

Set Transformer:

- https://arxiv.org/abs/1810.00825
- https://proceedings.mlr.press/v97/lee19d.html

---

# 14. Invariancia al orden

Para `choice`, el orden de las opciones normalmente no debería afectar a la probabilidad semántica.

Entrenar con permutaciones:

```text
options = shuffle(options)
```

y añadir:

```text
L_perm = KL(
    P(original),
    unshuffle(P(permuted))
)
```

No aplicar exactamente la misma regla a un `score` ordinal, donde el orden contiene semántica.

Medir:

```text
Permutation Stability Error
```

como benchmark propio.

---

# 15. Tipos de decisión

## 15.1 `choice`

Una sola opción correcta/óptima.

```text
softmax
```

## 15.2 `boolean`

Internamente sigue siendo:

```text
[true, false]
```

pero puede tener head/calibración específica.

## 15.3 `score`

Ejemplo:

```text
1 2 3 4 5
```

No tratarlo únicamente como cinco labels nominales.

Opciones:

- cumulative ordinal regression;
- CORAL-like thresholds;
- distribution over bins + expected value;
- Ranked Probability Score / EMD.

Salida útil:

```json
{
  "distribution": [0.01, 0.04, 0.20, 0.51, 0.24],
  "expected": 3.93
}
```

## 15.4 `multiselect` — V2

No softmax.

Cada opción puede ser verdadera:

```text
sigmoid per option
```

pero mantener option interaction.

## 15.5 `bounded_numeric` — futuro

Distribución discretizada o mixture head.

No priorizar.

---

# 16. Abstención y `unknown`

Debe ser una capacidad entrenada, no un `if max_prob < x` añadido al final.

Tipos de ejemplos negativos:

1. la respuesta correcta no está entre las opciones;
2. state insuficiente;
3. state contradictorio;
4. pregunta ambigua;
5. opción correcta demasiado genérica;
6. dominio completamente OOD;
7. inputs corruptos/truncados;
8. options duplicadas o casi equivalentes.

Añadir pseudo-opción:

```text
__insufficient_evidence__
```

o un **separate abstention head**.

Hay que probar ambos.

---

# 17. Dos niveles de incertidumbre

No asumir:

```text
softmax entropy == epistemic uncertainty
```

Construir:

### Nivel A — distributional confidence

Derivada del softmax/listwise output.

### Nivel B — trust/OOD head

Entrenado con:

- OOD datasets;
- corrupted examples;
- missing-answer examples;
- teacher disagreement;
- embedding distance;
- state/question incompatibility.

Salida:

```text
trust_probability
```

Separar “no sé cuál opción” de “este input no se parece a nada que entiendo”.

---

# 18. Cascada inteligente

El producto final no necesita que el pequeño modelo resuelva el 100%.

```text
                   FAST DECISION MODEL
                          │
             ┌────────────┴─────────────┐
          confident                  uncertain
             │                           │
          execute                larger model/agent
```

Esto puede hacer el sistema mucho mejor que intentar inflar el modelo pequeño.

Target eventual:

```text
80–95% decisions local fast path
5–20% escalated
```

El porcentaje debe surgir de datos, no fijarse dogmáticamente.

---

# 19. Early exit

Instalar heads provisionales en varias profundidades del state/question encoder.

Referencia:

- DeeBERT: https://arxiv.org/abs/2004.12993

Ejemplo:

```text
layer 4  → confidence?
layer 8  → confidence?
layer 12 → final
```

Sólo salir si:

- confianza calibrada;
- trust score suficiente;
- prediction estable entre dos heads;
- risk target satisfecho.

No implementar antes de tener una buena baseline; complicaría debugging.

---

# 20. Bi-encoder fast path

Para preguntas extremadamente simples:

```text
state_global
    ↕ dot/cosine
option embeddings
```

Latencia mínima.

Si la diferencia entre top-1 y top-2 es grande:

```text
return
```

Si no:

```text
cross-attention reranker
```

Esto crea una cascada interna:

```text
cheap bi-encoder → shallow cross-attention → external LLM
```

Especialmente útil con 100+ opciones.

---

# 21. Late interaction

Una solución intermedia entre bi-encoder y cross-encoder:

- State se codifica una vez en token vectors.
- Options/questions se codifican aparte.
- Score mediante MaxSim/late interaction.

Referencia:

- ColBERT: https://github.com/stanford-futuredata/ColBERT

No usar necesariamente el score exacto de ColBERT; reutilizar el concepto de **representaciones independientes pero interacción token-level barata**.

---

# 22. Shared-state branch attention

`kev` demuestra una idea importante:

- state visible para cada branch;
- cada question no ve otras questions;
- options de su propia question sí interactúan.

Para un decoder causal esto puede implementarse mediante block masks.

En nuestro diseño de encoder split, la separación es más limpia:

```text
State memory is immutable.
Each question branch cross-attends to the same memory.
```

Ventaja adicional: no necesitamos replicar el state en la matriz de tokens de cada branch.

Referencias útiles para shared-prefix/tree attention:

- Hydragen: https://arxiv.org/abs/2402.05099
- Repo Hydragen: https://github.com/ScalingIntelligence/hydragen
- DeFT: https://github.com/LINs-lab/DeFT
- DeFT ICLR 2025: https://proceedings.iclr.cc/paper_files/paper/2025/hash/a6df53f082619d02b9fad64a022e5de3-Abstract-Conference.html

No copiar estas implementaciones directamente: sus kernels están orientados a LLM/KV trees. Extraer principios de reducción de IO/reuso.

---

# 23. Losses

## 23.1 Categorical log loss

Base:

```text
L_ce = -log p(y)
```

Es una proper scoring rule.

## 23.2 Brier

```text
L_brier = Σ_i (p_i - y_i)^2
```

Útil para calibration-sensitive training.

No reemplazar automáticamente CE; probar:

```text
CE
CE + λ Brier
distill KL + CE
distill KL + CE + Brier
```

## 23.3 Distillation KL

Con distribución teacher:

```text
L_kd = KL(P_teacher || P_student)
```

Probar temperaturas:

```text
T = 1, 2, 4
```

## 23.4 Permutation consistency

```text
L_perm
```

para options nominales.

## 23.5 Ordinal loss

Para `score`:

- Ranked Probability Score;
- earth-mover distance;
- cumulative threshold objective.

## 23.6 OOD / abstention

Binary focal/BCE o selective-risk objective.

## 23.7 Total inicial

No hipercomplejizar:

```text
L = CE
  + λ_kd * KL
  + λ_brier * Brier
  + λ_perm * PermConsistency
  + λ_ood * OOD
```

Empezar con λ pequeños y ablation individual.

---

# 24. RLCD-like: qué hacer y qué NO hacer

TypeSafe usa el nombre RLCD, pero el algoritmo exacto no es público.

No empezar con PPO/GRPO sólo porque aparece la palabra “Reinforcement Learning”.

Si tenemos `ground truth y`, podemos entrenar directamente una distribución con proper scoring rules, con mucha menos varianza.

### Fase 1

Supervised:

```text
ground-truth + CE/Brier
```

### Fase 2

Teacher distributions:

```text
soft probability distillation
```

### Fase 3

Calibration fine-tune:

- frozen/low LR backbone;
- calibration-aware dataset;
- hard/OOD examples;
- proper scoring losses.

### Fase 4 — sólo si aparece necesidad real

RL/bandit cuando el outcome:

- llega tarde;
- depende de acciones;
- no existe label directo;
- queremos optimizar utilidad/risk.

---

# 25. Calibration

Referencia clásica:

- Guo et al., *On Calibration of Modern Neural Networks*:  
  https://arxiv.org/abs/1706.04599

## 25.1 Temperature scaling

```text
P = softmax(logits / T)
```

Debe ser el primer calibrador.

## 25.2 Variantes posteriores

Sólo si mejoran held-out calibration:

- vector scaling;
- matrix scaling;
- Dirichlet calibration;
- isotonic;
- beta calibration.

## 25.3 No usar un único T para todo necesariamente

Probar:

```text
T_global
T_choice
T_boolean
T_score
T_by_option_cardinality
T_by_domain
```

Con regularización para evitar overfit.

## 25.4 Métricas

Siempre:

```text
NLL
Brier
ECE
adaptive ECE
classwise ECE
reliability diagram
```

Para abstención:

```text
risk-coverage
selective accuracy
coverage @ target risk
AUROC OOD
AUPRC OOD
```

---

# 26. Conformal/selective risk — V2

Una vez tengamos buen calibrador, estudiar control formal del riesgo.

Fuentes recientes:

- Selective Conformal Risk Control (2025):  
  https://arxiv.org/abs/2512.12844
- Conformal Selective Prediction with General Risk Control (2026):  
  https://arxiv.org/abs/2603.24704
- Conformal Risk Control for Non-Monotonic Losses (2026):  
  https://arxiv.org/abs/2602.20151

Esto puede ser especialmente útil si el motor se usa para acciones automáticas:

```text
"execute only when estimated error risk < 0.5%"
```

No forma parte del MVP.

---

# 27. Data schema universal

Crear un formato propio desde el día 1.

```json
{
  "id": "banking77:test:123",
  "state": {
    "text": "I see the same transfer twice."
  },
  "questions": [
    {
      "id": "intent",
      "type": "choice",
      "prompt": "What best describes the customer's request?",
      "options": [
        {
          "id": "duplicate_transfer",
          "text": "Duplicate transfer",
          "description": "The same transfer appears more than once."
        },
        {
          "id": "cash_withdrawal",
          "text": "Cash withdrawal"
        }
      ],
      "target": {
        "option_id": "duplicate_transfer"
      }
    }
  ],
  "meta": {
    "source": "banking77",
    "source_split": "test",
    "language": "en",
    "license": "CC-BY-4.0",
    "commercial_pool": true,
    "group_id": "..."
  }
}
```

Para distillation:

```json
"teacher": {
  "model": "...",
  "distribution": {
    "duplicate_transfer": 0.94,
    "cash_withdrawal": 0.01,
    "...": 0.05
  },
  "entropy": 0.31,
  "timestamp": "..."
}
```

---

# 28. Data governance

Crear:

```text
data/
  manifests/
    huffpost.yaml
    banking77.yaml
    massive.yaml
    ...
  commercial/
  research_only/
  eval_only/
```

Cada manifest debe contener:

```yaml
name:
version:
source_url:
license:
license_url:
commercial_use:
attribution_required:
share_alike:
download_date:
sha256:
converter_version:
allowed_uses:
notes:
```

**Nunca** mezclar `research_only` con el checkpoint destinado a uso comercial.

---

# 29. Dataset 1 — HuffPost News Category

Fuentes:

- Kaggle: https://www.kaggle.com/datasets/rmisra/news-category-dataset
- HF mirror: https://huggingface.co/datasets/khalidalt/HuffPost

Aproximadamente 200K+ artículos/entradas en la versión habitual del dataset, con headline, short description y categorías.

## 29.1 Por qué sí usarlo

Excelente para validar:

- labels dinámicos;
- K variable;
- option shuffling;
- listwise discrimination;
- hard negatives;
- calibration básica;
- latency con muchas options.

## 29.2 Cómo transformarlo

State:

```text
headline + "\n" + short_description
```

Question:

```text
Which category best describes this article?
```

No entrenar siempre con las 42 categorías.

Curriculum:

```text
K=2
K=4
K=8
K=16
K=32
K=all
```

## 29.3 Hard negatives

No muestrear sólo aleatoriamente.

Ejemplo:

```text
POLITICS
WORLD NEWS
BUSINESS
MEDIA
CRIME
```

es más difícil que:

```text
POLITICS
WEDDINGS
STYLE
FOOD
PARENTING
```

Construir semantic-negative miner.

## 29.4 Limitaciones

- es fundamentalmente topic classification;
- datos históricos;
- posibles señales de vocabulario superficial;
- no enseña reasoning general;
- no es suficiente para calibración OOD.

**Conclusión:** smoke-test y parte del mix, nunca núcleo del proyecto.

**Nota legal:** el mirror de HF muestra metadatos de licencia; antes de un release comercial verificar también los términos de la fuente original/Kaggle y conservar evidencia de la versión descargada.

---

# 30. Dataset 2 — Banking77

- https://huggingface.co/datasets/PolyAI/banking77

Características:

- 77 intents;
- ~13K ejemplos;
- licencia CC-BY-4.0 en la card de HF.

Perfecto para:

- semantic routing;
- dynamic labels;
- near-neighbour hard negatives;
- OOD artificial;
- hierarchical labels futuros.

Transformaciones:

```text
K=4/8/16/32/77
```

Crear descripciones humanas/teacher para cada intent, no usar sólo nombres con underscores.

---

# 31. Dataset 3 — CLINC150 / OOS

Mirrors útiles:

- https://huggingface.co/datasets/DeepPavlov/clinc150
- https://huggingface.co/datasets/contemmcm/clinc150

~150 intents + OOS en variantes habituales.

Especialmente útil para:

- explicit out-of-scope;
- domain + intent hierarchy;
- `unknown`;
- routing.

**Antes de incorporar al checkpoint comercial:** revisar licencia de la distribución/original exacta; no confiar sólo en un mirror.

---

# 32. Dataset 4 — MASSIVE

- https://huggingface.co/datasets/AmazonScience/massive

Características:

- >1M utterances;
- 52 idiomas;
- intents + slots;
- CC-BY-4.0 según dataset card.

Muy importante para:

- multilingüe;
- routing;
- Spanish transfer;
- muchos paraphrases;
- intent descriptions.

No usar todos los idiomas inicialmente. Primera fase:

```text
en
es
fr
de
pt
```

Luego ampliar.

---

# 33. Dataset 5 — BoolQ

- https://huggingface.co/datasets/google/boolq

~16K preguntas yes/no con passage.

Convertir directamente:

```text
state = passage
question = question
options = [true, false]
```

Excelente para enseñar que el modelo debe **leer state**, no sólo clasificar una frase.

---

# 34. Dataset 6 — NLI

## ANLI

- https://huggingface.co/datasets/facebook/anli

Muy valioso porque es adversarial.

Formato:

```text
state = premise
question = "What is the relation of the hypothesis to the state?"
hypothesis = ...
options = [entailment, neutral, contradiction]
```

**Problema:** CC-BY-NC-4.0 según card.  
Mantener en `research_only`, no en checkpoint comercial.

Buscar además alternativas NLI con licencia permisiva para el pool comercial.

---

# 35. Dataset 7 — LogiQA 2.0

- https://huggingface.co/datasets/datatune/LogiQA2.0

Útil para:

- logical reasoning;
- multiple choice;
- distractors semánticamente fuertes.

La card consultada indica MIT.

Puede entrar en train/validation, pero separar una parte por fuentes/grupos para evitar memorizar patterns.

---

# 36. Dataset 8 — ReClor

- https://huggingface.co/datasets/sxiong/ReClor

Logical reasoning multiple choice.

Útil para:

- reasoning;
- option interaction;
- distillation.

Mantener un split intacto para generalization tests.

---

# 37. Dataset 9 — Civil Comments

- https://huggingface.co/datasets/google/civil_comments

Gran dataset de comments con señales de toxicidad.

Card: CC0-1.0.

Usos:

### boolean

```text
Is this comment toxic?
```

### score

Convertir score continuo a bins:

```text
0..4
```

### multi-question

Mismo state:

```text
toxic?
insult?
threat?
identity attack?
```

Éste es precisamente un buen test para **una sola codificación de state + múltiples preguntas**.

---

# 38. Dataset 10 — GoEmotions

- Repo oficial: https://github.com/google-research/google-research/tree/master/goemotions

~58K comments con 27 emociones + neutral según la publicación del dataset.

Usos:

- multiple questions over same state;
- multi-label futuro;
- fine-grained semantic distinctions.

Antes de commercial training: verificar licencia/dataset terms exactos de la distribución utilizada.

---

# 39. Dataset 11 — SST-5

- https://huggingface.co/datasets/SetFit/sst5

5 niveles de sentiment.

Útil para `score` ordinal:

```text
very negative
negative
neutral
positive
very positive
```

No limitarse a categorical CE; probar RPS/ordinal heads.

Verificar licencia exacta antes de commercial pool.

---

# 40. Dataset 12 — NVIDIA HelpSteer2

- https://huggingface.co/datasets/nvidia/HelpSteer2

Muy importante.

Contiene human feedback / quality axes y preference data.

Card: CC-BY-4.0.

Usos:

- `score`;
- pairwise `choice`;
- quality dimensions;
- teacher calibration;
- aprender preferencias no meramente temáticas.

Convertir una misma respuesta en varias preguntas:

```text
How helpful?
How correct?
How coherent?
How verbose/appropriate?
Which response is preferred?
```

Esto explota nuestra ventaja de multi-question shared-state.

---

# 41. Dataset 13 — FEVER

- https://huggingface.co/datasets/fever/fever

Fact verification.

Formato natural:

```text
state = evidence
question = claim
options = [supported, refuted, insufficient]
```

Muy útil para `insufficient evidence`.

Licencia/derivación de Wikipedia requiere revisión legal y de attribution/share-alike antes de un checkpoint comercial.

---

# 42. Datasets de benchmark — NO ENTRENAR

Crear lista de hashes/IDs prohibidos en training.

## MMLU-Pro

- https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro

Usar como evaluación de conocimiento/reasoning.

No entrenar.

## GPQA

- https://huggingface.co/datasets/Idavidrein/gpqa

448 preguntas expert-level en la versión original.

No entrenar.

## SimpleQA

- https://huggingface.co/datasets/OpenEvals/SimpleQA

Factual short-answer benchmark.

Para nuestro modelo hay que convertir respuestas a distractors **sin usar el test para entrenamiento**.

Mejor crear el option set sólo durante evaluation.

## MuSR

- https://huggingface.co/datasets/TAUR-Lab/MuSR

Reasoning/common-sense.

Eval-only.

## RewardBench 2

- https://huggingface.co/datasets/allenai/reward-bench-2

Eval de preference/reward models.

Eval-only.

## ARC Challenge

- https://huggingface.co/datasets/allenai/ai2_arc

Usar al menos ARC-Challenge como held-out.

## OpenBookQA

- https://huggingface.co/datasets/allenai/openbookqa

Eval/held-out inicialmente; revisar licencia antes de cualquier training.

---

# 43. Dataset matrix

| Dataset | Skill | Train commercial | Research | Eval-only | Prioridad |
|---|---|---:|---:|---:|---:|
| HuffPost | topic/dynamic choice | verificar origen | sí | no | P0 |
| Banking77 | intent | sí, CC-BY-4 | sí | parcial | P0 |
| CLINC150 | intent/OOD | revisar | sí | parcial | P0 |
| MASSIVE | multilingual intent | sí, CC-BY-4 | sí | parcial | P1 |
| BoolQ | boolean QA | revisar SA obligations | sí | parcial | P0 |
| Civil Comments | boolean/score | sí, CC0 | sí | parcial | P0 |
| HelpSteer2 | score/preference | sí, CC-BY-4 | sí | parcial | P0 |
| LogiQA2 | reasoning | según MIT card | sí | parcial | P1 |
| ReClor | reasoning | revisar distribución | sí | parcial | P1 |
| GoEmotions | multilabel/semantic | revisar | sí | parcial | P1 |
| SST-5 | ordinal | revisar | sí | parcial | P1 |
| ANLI | NLI | **NO** (NC) | sí | sí | P1 |
| ToxicChat | safety/OOD | **NO** (NC) | sí | sí | P2 |
| FEVER | fact verification | legal review | sí | sí | P1 |
| MMLU-Pro | reasoning | no | no | **sí** | P0 eval |
| GPQA | expert reasoning | no | no | **sí** | P0 eval |
| SimpleQA | factual | no | no | **sí** | P0 eval |
| MuSR | reasoning | no | no | **sí** | P1 eval |
| RewardBench2 | preference | no | no | **sí** | P1 eval |

---

# 44. Datos sintéticos: probablemente la pieza que más valor aportará

Los datasets existentes tienen labels fijas y estilos artificialmente separados.

Nuestro modelo necesita aprender el **meta-task**:

> “Lee estado + pregunta + opciones arbitrarias y produce una distribución calibrada.”

Por ello necesitamos convertir y sintetizar.

## 44.1 Transformaciones gratuitas

Para cada ejemplo:

- variar wording de la pregunta;
- variar descripción de labels;
- variar K;
- permutar options;
- añadir distractors;
- eliminar la respuesta correcta;
- añadir `unknown`;
- combinar varias questions sobre un mismo state;
- truncar state;
- introducir evidencia irrelevante;
- introducir contradicciones controladas.

## 44.2 Hard-negative mining

Pipeline:

```text
label descriptions
    ↓
embedding model
    ↓
nearest semantic labels
    ↓
construct difficult option set
```

El 50% de training no debería estar formado por negatives aleatorios fáciles.

Curriculum sugerido:

```text
25% random negatives
50% semantic hard negatives
15% adversarial teacher-generated
10% unknown/OOD
```

Ajustar por resultados.

---

# 45. Teacher distillation

Ésta puede ser la forma barata de transferir “inteligencia general” a un encoder pequeño.

No pedir al teacher sólo:

```text
answer = B
```

Pedir:

```json
{
  "probabilities": {
    "A": 0.03,
    "B": 0.74,
    "C": 0.20,
    "D": 0.03
  },
  "insufficient_evidence": 0.07
}
```

Pero **no confiar ciegamente en esas probabilidades**: los LLMs también están mal calibrados.

## 45.1 Ensemble de teachers

Para un subset importante:

```text
cheap teacher A
cheap teacher B
strong teacher C
```

Guardar:

```text
mean distribution
variance/disagreement
majority answer
```

El disagreement es una señal de dificultad muy útil.

---

# 46. APIs económicas para datos sintéticos

Precios cambian; comprobar de nuevo justo antes de lanzar una generación grande.

## Google Gemini 2.5 Flash-Lite

Pricing actual consultado el 19-09-2026:

- estándar: $0.10 / 1M input text tokens, $0.40 / 1M output;
- batch: $0.05 / 1M input, $0.20 / 1M output.

Fuente oficial:

https://ai.google.dev/gemini-api/docs/pricing

Muy atractivo para:

- paraphrases;
- distractors;
- label descriptions;
- bulk weak labels.

## OpenAI GPT-5.6 Luna

Pricing oficial actual:

- estándar short-context mostrado: $0.20 / 1M input y $1.20 / 1M output;
- la tabla oficial también muestra tiers de procesamiento con precios distintos.

Fuente:

https://developers.openai.com/api/docs/pricing

Usarlo para:

- segundo teacher;
- adjudicación;
- ejemplos donde Gemini/ground truth discrepen.

## Modelos fuertes

No usarlos para millones de examples.

Usarlos en:

```text
top 1–5% hardest examples
gold calibration set
teacher disagreement
reasoning dataset generation
```

---

# 47. Presupuesto sintético inicial recomendado

No gastar cientos de euros sin haber probado el pipeline.

### Fase A — €0–20

- datasets públicos;
- generación gratuita/tier barato;
- 20K synthetic examples;
- validar que distillation mejora algo.

### Fase B — €20–100

- 100K–500K transforms/examples, según longitud;
- 2 teachers en subset;
- hard-negative generation;
- calibration cases.

### Fase C — sólo si hay señal

- 1M+ synthetic decisions;
- teacher ensemble selectivo;
- human validation.

El cuello de botella probablemente no será el precio por token sino **calidad y contaminación de los labels**.

---

# 48. Human gold set

No confiar sólo en teachers.

Una pequeña muestra humana puede valer muchísimo.

Proveedor:

- Prolific: https://www.prolific.com/pricing

A fecha de consulta:

- pay-as-you-go;
- corporate platform fee normalmente 42.8% sobre participant rewards;
- recomiendan aproximadamente $12/h mínimo recomendado para participantes, más para expertise.

Usarlo para:

- 1K–5K ejemplos difíciles;
- pairwise preference;
- confidence/ambiguity labels;
- verificar `unknown`;
- calibration set independiente.

No usar humanos para etiquetar 500K ejemplos simples.

---

# 49. Gold set propio

Crear un conjunto que **nunca** se use para gradient updates.

Ideal:

```text
5K–20K decisions
```

Repartidas:

```text
routing
NLI
boolean QA
factual
reasoning
preference
ordinal
OOD
ambiguous
Spanish
long-context
```

Cada ejemplo difícil:

- 2–3 annotators o expert adjudication;
- registrar disagreement;
- si existe disagreement legítimo, conservar una distribución humana, no forzar one-hot.

Éste será probablemente el asset más valioso del proyecto.

---

# 50. Curriculum de training

## Stage 0 — head warmup

Freeze backbone.

Entrenar:

- question encoder/head;
- cross attention;
- option mixer;
- pointer scorer.

Datasets fáciles.

## Stage 1 — multi-dataset supervised

Unfreeze últimas capas del backbone.

Mix:

```text
HuffPost
Banking77
BoolQ
Civil Comments
HelpSteer2
```

## Stage 2 — full semantic mix

Añadir:

```text
MASSIVE
NLI
reasoning
fact verification
ordinal
preference
Spanish
```

## Stage 3 — hard negatives

Incrementar:

- semantic distractors;
- unknown;
- OOD;
- truncations;
- contradictions;
- reordered options.

## Stage 4 — distillation

Soft distributions.

## Stage 5 — calibration

Low LR.

Calibration-specific held-out data.

## Stage 6 — compression

Quantize / prune / distill to smaller backbone.

---

# 51. Dataset sampling

No samplear proporcional al tamaño.

MASSIVE/Civil Comments destruirían datasets pequeños.

Usar algo tipo:

```text
p(dataset) ∝ n_dataset^α
```

con:

```text
α ≈ 0.3–0.6
```

a tunear.

También quotas por capability:

```text
routing      20%
boolean/NLI  20%
reasoning    20%
score        15%
preference   10%
OOD          10%
multilingual 5%
```

Sólo punto de partida.

---

# 52. Curriculum de option cardinality

Entrenar explícitamente:

```text
K=2
K=3–4
K=5–8
K=9–16
K=17–32
K=33+
```

No permitir que el modelo aprenda:

```text
"si hay 77 opciones = Banking77"
```

Para ello variar K dentro de cada dataset.

---

# 53. Curriculum multi-question

Una ventaja del modelo es responder muchas preguntas sobre el mismo state.

Training batches deben incluir:

```text
Q = 1
Q = 2
Q = 4
Q = 8
Q = 16
```

Ejemplos naturales:

Civil Comments:

```text
toxicity?
insult?
threat?
identity attack?
```

HelpSteer:

```text
correctness?
helpfulness?
coherence?
style?
```

Synthetic:

generar diferentes preguntas sobre el mismo article/passsage.

---

# 54. Pretraining adicional opcional: text ↔ label contrastive

Antes del full decision objective:

```text
state/query
↔
correct label description
```

Contrastive loss:

```text
InfoNCE
```

Esto puede mejorar dynamic-label generalization.

Luego pasar a listwise decision training.

No es obligatorio; medir.

---

# 55. From-scratch: cuándo tendría sentido

No ahora.

Entrenar desde cero un encoder competente requiere:

- corpus enorme;
- tokenizer;
- pretraining recipe;
- miles de millones/trillones de tokens;
- compute muy superior al hardware disponible.

Tus equipos sí permiten entrenar:

- heads;
- adapters;
- LoRA;
- full fine-tune de encoders pequeños/medios;
- distillation;
- continued pretraining relativamente pequeño.

No permiten competir razonablemente con un pretraining tipo 2T tokens.

## Excepción

Entrenar un **tiny 5–30M model from scratch** sí tiene sentido como experimento de latencia y para validar nuestra arquitectura.

No como modelo principal.

---

# 56. Hardware disponible y uso recomendado

Disponemos de:

- Mac M4 Max, 48 GB unified memory;
- Mac M5, 48 GB unified memory;
- Windows con NVIDIA CUDA, GPU exacta todavía por identificar.

## 56.1 Primer task del Windows

Guardar output de:

```bash
nvidia-smi
```

y:

```bash
python - <<'PY'
import torch
print(torch.cuda.get_device_name())
print(torch.cuda.get_device_properties(0).total_memory / 2**30)
print(torch.cuda.get_device_capability())
PY
```

Necesitamos:

- modelo;
- VRAM;
- compute capability;
- driver/CUDA;
- bf16 support.

No diseñar training alrededor de una GPU que aún no hemos identificado.

---

# 57. Apple Silicon stack

## PyTorch MPS

Docs:

https://docs.pytorch.org/docs/stable/notes/mps.html

Usar para:

- desarrollo;
- correctness;
- training pequeño;
- comparar M4 vs M5.

## MLX

Repo:

https://github.com/ml-explore/mlx

Examples:

https://github.com/ml-explore/mlx-examples

Usar más adelante para:

- inference optimizado;
- memory-efficient Apple implementation;
- explorar custom kernels si compensa.

## ONNX + CoreML EP

https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html

Útil para evaluar:

- GPU;
- ANE donde sea compatible;
- graph optimization.

No asumir que un modelo exportado automáticamente será más rápido que PyTorch/MLX; medir.

---

# 58. CUDA stack

Referencia:

- ONNX Runtime CUDA/TensorRT:  
  https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- TensorRT EP:  
  https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html

Orden:

```text
PyTorch eager
torch.compile
ONNX Runtime CUDA
TensorRT / ORT TensorRT
```

Medir cada uno.

---

# 59. Quantization

## Fase 1

FP16/BF16.

## Fase 2

INT8 static.

## Fase 3

INT8 weight + activations o mixed precision.

## Fase 4

INT4 sólo si quality se mantiene.

HF Optimum/ORT:

https://huggingface.co/docs/optimum-onnx/onnxruntime/usage_guides/quantization

ModernBERT ya tiene comunidad/exports ONNX y quantized variants; aprovechar tooling, pero quantizar nuestro modelo final con nuestro calibration corpus.

---

# 60. Benchmark de latencia correcto

No publicar un único número.

Medir combinaciones:

## State length

```text
128
512
2048
8192
```

## Questions

```text
1
2
10
50
```

## Options/question

```text
2
4
8
16
32
64
```

## Batch

```text
1
4
16
```

## Precision

```text
fp32
fp16/bf16
int8
```

## Plataformas

```text
M4 Max
M5
CUDA GPU
CPU
```

---

# 61. Descomponer la latencia

Registrar:

```text
tokenization
state encode
question encode
cross attention
option mixing/head
calibration
serialization
total
```

Dos modos:

### cold state

```text
includes state encode
```

### warm/cached state

```text
state memory already available
```

Esto es crítico. Nuestro modelo debe ser especialmente fuerte en warm-state/multi-question.

---

# 62. Benchmark energético opcional

En local/on-device puede ser relevante:

```text
joules / decision
battery impact
power draw
```

No P0.

---

# 63. Métricas de calidad

Por task:

```text
accuracy
macro-F1
micro-F1
AUROC where applicable
Spearman/Pearson for scores
```

Global:

```text
NLL
Brier
adaptive ECE
```

Dynamic option behavior:

```text
accuracy vs K
latency vs K
calibration vs K
```

Multi-question:

```text
quality vs Q
latency vs Q
```

Long context:

```text
quality vs state length
```

---

# 64. Métricas nuevas propias

## Option Permutation Stability

```text
mean JS divergence after permutations
top-1 flip rate
```

## Option Insertion Stability

Añadir una opción irrelevante:

```text
does ranking of existing options change irrationally?
```

No exigir invariancia absoluta: algunas option sets deben cambiar probabilidades.

## Missing Correct Answer Detection

Retirar el gold option.

Medir:

```text
unknown recall
unknown precision
false confident rate
```

## Shared-State Efficiency

```text
cost(Q questions)
/
(Q * cost(single independent question))
```

Queremos que caiga claramente al crecer Q.

## Semantic Label Transfer

Train label descriptions A; eval con paraphrased/novel descriptions B.

---

# 65. Benchmarks internos críticos

Crear pruebas que los datasets clásicos no tienen.

## 65.1 Label rename

```text
"cash withdrawal"
→
"ATM cash removal"
```

¿Sigue funcionando?

## 65.2 Opaque ID trap

El modelo no debe depender de `label_17`.

## 65.3 Option shuffle

Ya descrito.

## 65.4 Hard semantic siblings

Sólo labels cercanos.

## 65.5 Irrelevant-state injection

Añadir 2K tokens irrelevantes.

## 65.6 Evidence near end

Información decisiva al final del state.

## 65.7 Contradictory evidence

Dos hechos incompatibles.

## 65.8 Multilingual mismatch

State español, labels inglés y viceversa.

## 65.9 Typo/noisy text

Chats reales.

---

# 66. Baselines obligatorias

No podremos saber si construimos algo interesante sin comparar.

### Baseline 0

Random / majority.

### Baseline 1

Fixed-head fine-tuned encoder por dataset.

### Baseline 2

GLiClass.

- Repo: https://github.com/Knowledgator/GLiClass

### Baseline 3

Laya.

### Baseline 4

Verdict.

### Baseline 5

`kev`.

### Baseline 6

Small decoder + first-token/prefill decision.

Ejemplo Qwen pequeño.

### Baseline 7

Gemma/open-jev cached likelihood.

Sirve como quality upper bound con backbone mayor.

---

# 67. GLiClass como fuente de código, no como arquitectura final

Repo:

https://github.com/Knowledgator/GLiClass

Piezas reutilizables/conceptos:

- dynamic labels;
- uni-encoder;
- bi-encoder;
- fusion;
- scorers;
- hierarchical labels;
- serving;
- continual learning ideas.

También existe backend C/ONNX:

https://github.com/Knowledgator/GLiClass.c

Podemos reutilizar adapters/ideas, pero nuestro diferenciador debe ser:

```text
true state encode once
+
multi-question branches
+
calibrated decision training
+
abstention
+
latency-first runtime
```

---

# 68. Repo propuesto

```text
system-one/
├── README.md
├── pyproject.toml
├── configs/
│   ├── model/
│   ├── data/
│   ├── train/
│   └── eval/
├── src/
│   ├── model/
│   │   ├── state_encoder.py
│   │   ├── question_encoder.py
│   │   ├── cross_attention.py
│   │   ├── option_mixer.py
│   │   ├── pointer_head.py
│   │   ├── ordinal_head.py
│   │   ├── uncertainty_head.py
│   │   └── model.py
│   ├── data/
│   │   ├── schema.py
│   │   ├── registry.py
│   │   ├── sampler.py
│   │   ├── hard_negatives.py
│   │   └── adapters/
│   │       ├── huffpost.py
│   │       ├── banking77.py
│   │       ├── boolq.py
│   │       ├── civil_comments.py
│   │       └── helpsteer2.py
│   ├── train/
│   │   ├── losses.py
│   │   ├── distill.py
│   │   ├── trainer.py
│   │   └── calibration.py
│   ├── eval/
│   │   ├── metrics.py
│   │   ├── calibration.py
│   │   ├── ood.py
│   │   ├── invariance.py
│   │   └── latency.py
│   ├── runtime/
│   │   ├── pytorch.py
│   │   ├── onnx.py
│   │   ├── mlx.py
│   │   └── cache.py
│   └── api/
│       ├── schema.py
│       └── engine.py
├── data/
│   └── manifests/
├── benchmarks/
├── experiments/
├── tests/
└── scripts/
```

---

# 69. Testing

## Unit

- masks;
- option indexing;
- permutation restore;
- missing option;
- padding;
- multiple Q;
- score ordering;
- cache validity.

## Numerical

- single vs batch result consistency;
- cache vs uncached consistency;
- fp32 vs fp16 deviation;
- ONNX vs PyTorch;
- MLX vs PyTorch.

## Dataset

- no ID overlap;
- no eval contamination;
- label distributions;
- licenses/manifests present.

## Regression

Cada model change debe ejecutar un pequeño fixed benchmark.

---

# 70. Reproducibility

Cada run:

```yaml
git_commit:
dataset_manifest_hash:
model_backbone:
backbone_revision:
seed:
hardware:
precision:
batch:
optimizer:
lr:
loss_weights:
timestamp:
```

Guardar:

```text
metrics.json
latency.json
config.yaml
calibration.json
```

No depender de WandB para reconstruir un run; logs locales deben bastar.

---

# 71. Iniciativa I0 — Reconocimiento y baselines

**Objetivo:** saber qué existe y qué rendimiento real tiene en nuestro hardware.

### I0-T1 — inventariar hardware

- `nvidia-smi`;
- macOS/chip exacto;
- RAM;
- versions;
- disk.

**Done:** `benchmarks/hardware.json`.

### I0-T2 — ejecutar GLiClass

Medir:

- 1Q;
- 10Q;
- K variable.

### I0-T3 — ejecutar Laya

Mismo benchmark.

### I0-T4 — ejecutar Verdict

Mismo benchmark.

### I0-T5 — ejecutar `kev`

Medir específicamente state reuse.

### I0-T6 — ejecutar `jevlike`

Medir mínimo posible de latencia.

### I0-T7 — ejecutar open-jev / small LLM baseline

Calidad/latencia upper bound.

### I0-T8 — informe

Crear:

```text
benchmarks/baseline_2026-09.md
```

**Acceptance:** tabla reproducible de quality/latency/memory en al menos Mac M4 Max y Windows CUDA.

---

# 72. Iniciativa I1 — Data foundation

**Objetivo:** ningún dataset entra al modelo sin trazabilidad.

### I1-T1 — schema universal

Implementar JSONL/Arrow schema.

### I1-T2 — data registry

`DATA_SOURCES.yaml`.

### I1-T3 — license fence

Flags:

```text
commercial
research_only
eval_only
```

Training commercial falla si intenta cargar NC/eval.

### I1-T4 — hashes

Hash de source + converted split.

### I1-T5 — leakage detector

Exact match + normalized text match + approximate similarity para detectar solapamientos con eval.

### I1-T6 — dataset cards internas

Generadas automáticamente.

**Acceptance:** reproducir exactamente un training mix desde manifest.

---

# 73. Iniciativa I2 — Adapters P0

### I2-T1 HuffPost

- parse;
- normalize labels;
- K curriculum;
- semantic negatives.

### I2-T2 Banking77

- label descriptions;
- K curriculum;
- hard siblings.

### I2-T3 BoolQ

- boolean schema.

### I2-T4 Civil Comments

- boolean + score;
- multi-Q.

### I2-T5 HelpSteer2

- score;
- pairwise;
- multi-Q.

### I2-T6 Mixed loader

Balance por capability/dataset.

**Acceptance:** >100K normalized decision examples y smoke-training completo.

---

# 74. Iniciativa I3 — Backbone bake-off

### I3-T1 Ettin 32M
### I3-T2 Ettin 68M
### I3-T3 Ettin 150M
### I3-T4 ModernBERT 149M
### I3-T5 NeoBERT 250M
### I3-T6 LFM2.5 230M

Mismo head provisional.

### I3-T7 Pareto report

Ejes:

```text
quality
latency
memory
training speed
license
export difficulty
multilingual
```

### I3-T8 seleccionar top 2

No elegir uno aún. Mantener top-2 hasta shared-state test.

**Acceptance:** una selección basada en medidas, no intuición.

---

# 75. Iniciativa I4 — Baseline dynamic-choice model

Arquitectura simple:

```text
state + question + option markers
→ encoder
→ option representations
→ shared scorer
```

Sí, re-encodifica state; es deliberadamente simple.

### Objetivo

Crear referencia de calidad para saber cuánto perdemos al separar state/question.

### Tasks

- option markers;
- dynamic K;
- CE;
- Brier;
- shuffle;
- score head;
- calibration;
- benchmark.

**Acceptance:** funciona en cinco datasets y supera fixed random/naive baselines.

---

# 76. Iniciativa I5 — Arquitectura shared-state V1

Ésta es la iniciativa central.

### I5-T1 state encoder API

```python
memory = model.encode_state(state)
```

### I5-T2 question encoder

```python
queries = model.encode_questions(questions)
```

### I5-T3 one-layer cross attention

Q/options → S.

### I5-T4 option pooling

Representación estable por option.

### I5-T5 pointer head

Dynamic score.

### I5-T6 branch batching

Todas las questions en una sola estructura batch.

### I5-T7 cache

State memory reutilizable.

### I5-T8 cache invalidation

Version/hash-safe.

### I5-T9 parity benchmark

Comparar contra full cross-encoder baseline.

**Acceptance:**

- estado se calcula exactamente una vez;
- quality gap aceptable;
- `Q=10` mejora clara de latencia vs 10 independent forwards.

---

# 77. Iniciativa I6 — Option mixer

### I6-T1 independent scorer baseline
### I6-T2 DeepSets mixer
### I6-T3 1-layer Set Transformer
### I6-T4 2-layer Set Transformer
### I6-T5 option insertion benchmark

**Acceptance:** elegir la opción que aporte mayor robustness por <10% de overhead del branch.

---

# 78. Iniciativa I7 — Calibration

### I7-T1 NLL/Brier/ECE toolkit
### I7-T2 temperature scaling
### I7-T3 per-type temperature
### I7-T4 cardinality calibration
### I7-T5 reliability plots
### I7-T6 calibration drift by domain
### I7-T7 calibrator serialization

**Acceptance:** calibration reproducible en held-out sin degradar materialmente accuracy.

---

# 79. Iniciativa I8 — OOD / abstention

### I8-T1 missing-answer generator
### I8-T2 unrelated-question examples
### I8-T3 corrupted state
### I8-T4 contradiction examples
### I8-T5 explicit unknown option
### I8-T6 separate trust head
### I8-T7 compare strategies
### I8-T8 risk-coverage curves

**Acceptance:** reducir drásticamente respuestas erróneas de alta confianza en OOD.

---

# 80. Iniciativa I9 — Hard-negative engine

### I9-T1 label embedding index
### I9-T2 nearest label sampler
### I9-T3 in-domain hard negatives
### I9-T4 cross-dataset labels
### I9-T5 teacher adversarial distractors
### I9-T6 difficulty score

Guardar difficulty para curriculum.

**Acceptance:** dataset puede generar K arbitrario con dificultad controlada.

---

# 81. Iniciativa I10 — Distillation

### I10-T1 teacher schema
### I10-T2 Gemini cheap pipeline
### I10-T3 second teacher
### I10-T4 disagreement detection
### I10-T5 soft KL loss
### I10-T6 confidence weighting

No dar el mismo peso a teacher examples donde teachers se contradicen.

### I10-T7 strong-teacher adjudication

Sólo hard subset.

### I10-T8 cost accounting

Guardar coste estimado/real por dataset.

**Acceptance:** demostrar mejora zero-shot/calibration, no sólo training accuracy.

---

# 82. Iniciativa I11 — Reasoning/generalization mix

Añadir:

- LogiQA2;
- ReClor;
- NLI;
- FEVER research/commercial according to license;
- multilingual;
- synthetic transformations.

### Critical split

Hold out **entire task families**, no sólo random rows.

Ejemplo:

```text
train:
routing + sentiment + NLI

test zero-shot:
new reasoning family
```

**Acceptance:** medir si hemos creado un meta-decision model o sólo un multi-dataset classifier.

---

# 83. Iniciativa I12 — Evaluation firewall

### I12-T1 benchmark registry

Mark:

```text
eval_only = true
```

### I12-T2 contamination scanner

Comparar train text contra:

- MMLU-Pro;
- GPQA;
- SimpleQA;
- MuSR;
- RewardBench2.

### I12-T3 immutable benchmark versions

Pin revisions/hashes.

### I12-T4 report contamination

Si aparece overlap, excluir items afectados.

**Acceptance:** resultados defendibles.

---

# 84. Iniciativa I13 — Multilingual

No empezar entrenando 52 idiomas.

### Phase 1

```text
English
Spanish
```

### Phase 2

```text
French
German
Portuguese
```

### Tasks

- MASSIVE;
- translated/generated decision examples;
- bilingual labels;
- cross-language question/state.

Test importante:

```text
state=Spanish
question=English
options=English
```

y viceversa.

**Acceptance:** español no es simplemente translation-to-English hidden pipeline.

---

# 85. Iniciativa I14 — Long state

### I14-T1 512-token baseline
### I14-T2 2K
### I14-T3 4K
### I14-T4 8K
### I14-T5 evidence-position benchmark

Insertar evidence:

```text
0%
25%
50%
75%
95%
```

### I14-T6 latency scaling

**Acceptance:** no perder de forma catastrófica información al final del contexto.

---

# 86. Iniciativa I15 — Latent state compression

Sólo después de I14.

### I15-T1 learned 256 memories
### I15-T2 128
### I15-T3 64
### I15-T4 query-dependent retrieval into state
### I15-T5 hybrid local memory + global pool

**Acceptance:** 2K–8K states obtienen mejora de latency suficiente para justificar quality loss.

---

# 87. Iniciativa I16 — Fast path / reranking cascade

### Stage 1

Bi-encoder score.

### Stage 2

Top-N candidates:

```text
N = 4/8/16
```

Cross-attend sólo top-N.

### Stage 3

Unknown/LLM escalation.

Ideal con 100–10,000 labels.

**Acceptance:** large-label routing sin coste lineal completo innecesario.

---

# 88. Iniciativa I17 — Quantization

### I17-T1 FP16/BF16 reference
### I17-T2 dynamic INT8
### I17-T3 static INT8
### I17-T4 QAT if needed
### I17-T5 INT4 experimental
### I17-T6 calibrator after quantization

Importantísimo:

**recalibrar después de quantization**.

**Acceptance:** INT8 obtiene mejora de latency/memory sin degradación significativa de NLL/Brier/accuracy.

---

# 89. Iniciativa I18 — ONNX runtime

### I18-T1 export fixed/simple shapes
### I18-T2 dynamic K/Q
### I18-T3 graph optimization
### I18-T4 CUDA EP
### I18-T5 CoreML EP
### I18-T6 benchmark vs PyTorch

Cuidado con dynamic shapes: pueden impedir optimizaciones.

Probar bucketing:

```text
K buckets = 2,4,8,16,32
Q buckets = 1,4,8,16,32
```

con padding controlado.

---

# 90. Iniciativa I19 — MLX

Sólo si PyTorch/ONNX no aprovecha bien Apple.

### I19-T1 port core model
### I19-T2 weight converter
### I19-T3 parity tests
### I19-T4 fused pointer/mixer
### I19-T5 state-memory cache
### I19-T6 M4/M5 benchmark

**Acceptance:** mejora real; si no, abandonar MLX port y evitar mantenimiento doble.

---

# 91. Iniciativa I20 — Early exit

### I20-T1 intermediate heads
### I20-T2 stability criterion
### I20-T3 calibration by exit depth
### I20-T4 risk-safe exit policy

No salir sólo porque `max softmax > 0.9`.

Usar:

```text
confidence
+
trust
+
head agreement
```

**Acceptance:** reducir mean compute manteniendo target risk.

---

# 92. Iniciativa I21 — Human gold / calibration data

### I21-T1 annotation UI/schema
### I21-T2 500 pilot items
### I21-T3 inter-annotator agreement
### I21-T4 ambiguity policy
### I21-T5 5K gold target
### I21-T6 reserve calibration/test splits

Guardar distributions cuando exista disagreement.

---

# 93. Iniciativa I22 — Public API

Mantener API extremadamente simple.

```python
result = engine.decide(
    state=state,
    questions=questions,
)
```

Advanced:

```python
memory = engine.encode_state(state)
result = engine.decide_from_memory(memory, questions)
```

Esto expone explícitamente nuestra ventaja.

## Output

Siempre incluir:

```text
probabilities
selected option
confidence
trust/abstain
model version
```

No incluir chain-of-thought.

---

# 94. Iniciativa I23 — Compatibility / Jev-like adapter

No copiar APIs propietarias exactamente si hay riesgos de marca/compatibilidad contractual.

Pero crear adapter conceptual:

```text
state
questions
choice / boolean / score
```

Así es fácil probar aplicaciones destinadas a System-One models.

---

# 95. Iniciativa I24 — Continual improvement

Guardar casos escalados al LLM:

```text
small model uncertain
→ teacher resolves
→ anonymized training candidate
```

Esto crea flywheel:

```text
production hard cases
→ human/teacher review
→ curated replay
→ next checkpoint
```

Evitar self-training automático sin quality gate.

---

# 96. Iniciativa I25 — Research: sparse MoE

No MVP.

Idea:

```text
shared encoder
→ router
→ small semantic experts
```

Puede aumentar capacidad total sin activar todos los parámetros.

Pero introduce:

- routing instability;
- export complexity;
- Apple kernel problems;
- calibration by expert;
- load imbalance.

Sólo explorar si dense 150–350M queda claramente limitado.

---

# 97. Iniciativa I26 — Research: tiny model from scratch

Objetivo científico:

```text
¿Cuál es la latencia mínima posible con nuestra arquitectura?
```

Tamaños:

```text
5M
15M
30M
60M
```

Pretrain pequeño o train únicamente sobre decision corpus.

No esperar general knowledge fuerte.

Puede ser excelente como:

- domain-specific router;
- embedded/on-device agent;
- benchmark del architecture overhead.

---

# 98. Iniciativa I27 — Research: convert decoder knowledge to encoder

Ettin incluye investigación encoder/decoder comparable y checkpoints cruzados.

Explorar más adelante:

```text
small decoder pretrained
→ bidirectional conversion / continued MLM
→ decision training
```

Puede ser una ruta para absorber conocimiento de mejores decoder checkpoints sin pagar decoding runtime.

No P0.

---

# 99. Iniciativa I28 — Safety contra shortcut learning

Construir adversarial tests.

### Label-name leakage

Cambiar nombres.

### Dataset fingerprint

Paraphrase prompts.

### Option position

Balancear gold position.

### Length

Evitar que la opción correcta tenga más palabras sistemáticamente.

### Punctuation

Randomizar formato.

### Dataset-specific template

Generar múltiples prompts.

### Source identity

Hold out sources/domains.

---

# 100. Iniciativa I29 — Knowledge freshness

Un encoder pequeño no tendrá “conocimiento actual” por magia.

Separar:

```text
semantic decision ability
```

de:

```text
world knowledge
```

Para datos actuales, pasar la información en `state`.

Ésta es una propiedad deseable:

```text
retriever/tool → state
fast decision model → decision
```

No intentar convertirlo en una base de conocimiento viva.

---

# 101. Arquitectura V1 concreta

Primera implementación seria:

```text
Backbone:
  top-2 winner from bake-off

State:
  full pretrained encoder
  token memory + global pooled vector

Question:
  shared tokenizer/embeddings
  2–4 lightweight transformer blocks

Fusion:
  2 cross-attention blocks Q/O -> state memory

Options:
  one marker/pool per option

Option mixer:
  1 transformer layer

Head:
  bilinear pointer scorer

Auxiliary:
  unknown/trust head

Loss:
  CE
  + 0.1-ish Brier [tune]
  + KD when available
  + permutation consistency
```

No fijar coeficientes sin sweep.

---

# 102. Arquitectura V1.5 para mayor velocidad

```text
state encoder
   ↓
state token memory
   ↓
token selector / compressor
   ↓
128 state memories

question encoder:
2 blocks

cross-attn:
1–2 blocks

option mixer:
DeepSets or 1 attention block

pointer head
```

---

# 103. Arquitectura V2 para muchas labels

```text
label text
  ↓
cached label encoder embeddings

state
  ↓
state global
  ↓
ANN / bi-encoder shortlist top 16

top 16
  ↓
cross-attention reranker

pointer head
```

Esto convierte 10,000 labels en un problema barato.

---

# 104. Arquitectura V2 para agentes

La use case más interesante:

```text
AGENT STATE
- user request
- tool results
- memory
- environment
- policy

questions:
- which tool?
- should we ask clarification?
- is action safe?
- is task complete?
- which next state?
- confidence?
```

Una sola codificación del agent state puede alimentar muchas decisiones.

Éste es precisamente donde el diseño puede superar a llamar repetidamente a un LLM.

---

# 105. Ejemplo multi-question de agente

```json
{
  "state": "...",
  "questions": [
    {
      "id": "next_action",
      "type": "choice",
      "options": [
        "call_calendar",
        "call_email",
        "ask_user",
        "finish"
      ]
    },
    {
      "id": "need_user_input",
      "type": "boolean"
    },
    {
      "id": "task_complete",
      "type": "boolean"
    },
    {
      "id": "risk",
      "type": "score",
      "scale": [0,1,2,3,4]
    }
  ]
}
```

Un LLM convencional puede gastar varios generation loops. Nuestro modelo debería producir las cuatro decisiones en un pass compuesto.

---

# 106. Cost model conceptual

LLM autoregresivo:

```text
prefill(state)
+
decode token 1
+
decode token 2
...
+
decode token N
```

Jev-like:

```text
encode(state)
+
parallel question fusion
+
readout
```

Nuestro objetivo adicional:

```text
encode(state) once
+
N cheap branches
```

Ésta es la ventaja a defender en cada decisión de diseño.

---

# 107. Donde puede fracasar la arquitectura split

No ocultar este riesgo.

Un cross-encoder completo deja que:

```text
state tokens ↔ question tokens
```

se influyan en cada capa.

Nuestro diseño:

```text
state is computed without knowing the question
```

Eso puede perder razonamiento query-dependent.

Soluciones, en orden:

1. más capas cross-attention;
2. bidirectional fusion final;
3. query-dependent state token selection;
4. state compressor conditioned on question;
5. small re-encoder sólo de top relevant state spans.

No abandonar state caching demasiado pronto.

---

# 108. Query-dependent state retrieval

Una mejora posible:

```text
question embedding
      ↓
score state token/chunks
      ↓
top relevant chunks
      ↓
shallow cross encoder
```

Para 8K state, puede ser mejor que atender a todo.

State chunks se codifican una vez.

---

# 109. Chunk memory

Long state:

```text
state
→ chunks 128–256 tokens
→ chunk vectors
```

Question:

```text
retrieve top chunks
→ token-level cross-attn only within selected chunks
```

Esto aproxima un RAG **dentro del state**.

Especialmente interesante para agent memory/logs.

---

# 110. Training para state reuse

No basta con arquitectura.

El dataset debe contener grupos:

```text
same state
question A
question B
question C
...
```

Si todos los training examples tienen sólo Q=1, el modelo nunca aprende a explotar bien la estructura multi-Q.

Crear multi-Q synthetic packs.

---

# 111. Training con mutable state

En agentes:

```text
state_t
→ decision
→ action
→ state_t+1
```

Para V2:

- cache state segments;
- incremental state encoder;
- append-only logs;
- invalidate sólo changed chunks.

No MVP.

---

# 112. Distillation desde reasoning models sin copiar CoT

Teacher puede razonar internamente, pero dataset final guarda:

```text
distribution
answer
difficulty
```

No necesitamos almacenar chain-of-thought.

Opcionalmente guardar una **short rationale** sólo para data QA, no como target del student.

---

# 113. Teacher prompt design

Obligar a considerar las opciones como conjunto.

Prompt conceptual:

```text
Given STATE, QUESTION and OPTIONS:
1. estimate probability that each option is the best answer;
2. probabilities must sum to 1;
3. do not assume one option is correct if evidence is insufficient;
4. output JSON only.
```

Para `unknown`:

```text
Include explicit insufficient_evidence probability.
```

Repetir con option permutations para detectar positional bias del teacher.

---

# 114. Teacher quality filter

Descartar o marcar ejemplos cuando:

```text
teacher output invalid
sum(prob) bad
teacher changes answer under permutation
teachers disagree strongly
ground truth contradicted
```

Los ejemplos difíciles no se tiran necesariamente: pasan a adjudication.

---

# 115. Synthetic reasoning generation

No generar “random puzzles” sin control.

Generar a partir de templates verificables:

- boolean logic;
- ordering;
- temporal;
- arithmetic small;
- set membership;
- route/tool decisions;
- contradictions.

Al disponer de generador programático, conocemos el gold exacto.

Esto evita teacher hallucinations.

---

# 116. Programmatic datasets

Crear generadores propios.

### Logic

```text
A before B.
C after B.
Who is earliest?
```

### Policy

```text
rule set + event → allowed/blocked/escalate
```

### Tool routing

```text
capability definitions + request → tool
```

### State machines

```text
current state + event → next state
```

### Numeric thresholds

```text
metrics → choice
```

Son muy buenos para aprender meta-decision sin copyright/licencia.

---

# 117. Domain-specific data para agentes

Más adelante crear corpus propio inspirado en operaciones reales:

```text
tool selection
API error routing
workflow status
task completion
risk gating
memory relevance
notification priority
```

Éste puede ser el mayor valor práctico del modelo.

---

# 118. Español

No depender sólo de traducción.

Combinar:

- MASSIVE Spanish;
- synthetic Spanish;
- native Spanish human examples;
- bilingual pairs;
- code-switching.

Test:

```text
¿entiende "domiciliar un recibo", "empadronamiento", etc.?
```

Si el producto se usa en Europa, español debería entrar antes de release V1.

---

# 119. Código y developer decisions

Si interesa routing en IDE/agentes de código:

State puede incluir:

```text
error
code diff
tool state
test results
```

Questions:

```text
which subsystem?
which test?
safe to commit?
retry or rollback?
```

ModernBERT tiene code en pretraining; podría aventajar a otros encoders aquí. Añadir un mini benchmark de code-routing al bake-off.

---

# 120. Model selection no debe basarse sólo en NLU estándar

Backbone scorecard:

| Dimensión | Peso inicial |
|---|---:|
| shared-state decision quality | alta |
| zero-shot dynamic labels | alta |
| latency Apple | alta |
| latency CUDA | alta |
| calibration | alta |
| Spanish | media |
| 8K context | media |
| permissive license | alta |
| export/runtime ease | media |
| code understanding | media |
| checkpoint size | baja-media |

No convertir pesos en verdad absoluta; revisar después de POC.

---

# 121. Licencia como criterio técnico

Si dos backbones son similares y uno es MIT/Apache mientras otro tiene licencia custom, elegir el permisivo puede ahorrar mucho coste futuro.

Por eso:

- Ettin MIT: muy atractivo.
- NeoBERT MIT: muy atractivo.
- ModernBERT Apache 2.0: muy atractivo.
- mmBERT MIT: muy atractivo.
- LFM2.5: técnicamente muy atractivo, pero revisar LFM Open License.

---

# 122. No distribuir datasets mezclados sin control

Release ideal:

```text
code
model weights
training manifest
converter scripts
dataset source URLs
```

No necesariamente redistribuir raw datasets.

Esto reduce problemas de licencia y trazabilidad.

---

# 123. Training strategy para hardware doméstico

## Primera fase

Fine-tune:

```text
32M–250M
```

No `large` de entrada.

### Techniques

- bf16/fp16;
- gradient accumulation;
- gradient checkpointing sólo si necesario;
- freeze lower layers al principio;
- adapters/LoRA si full FT es demasiado lento;
- sequence bucketing;
- dynamic padding;
- mixed dataset batches.

## Mac

MPS para iteration.

## CUDA

Si VRAM suficiente, training principal.

## M5

Preferirlo como **inference reference machine**, para no contaminar latency measurements con un entorno diferente cada vez.

---

# 124. No intentar distributed training entre los dos Macs al principio

El overhead y complejidad probablemente no compensa.

Mejor:

```text
Mac A → experiment family A
Mac B → experiment family B
CUDA → experiment family C
```

Paralelizar experiments, no gradients.

---

# 125. Experiment scheduler simple

Una cola local:

```text
experiments/*.yaml
results/<run_id>/
```

Cada máquina toma un job compatible.

No montar Kubernetes.

---

# 126. Hyperparameter strategy

No hacer grid enorme.

Primero:

- learning rate;
- fusion depth;
- backbone;
- KD weight;
- Brier weight;
- state memory size.

Usar:

- coarse sweep;
- successive halving;
- después afinado.

La arquitectura probablemente domina más que pequeños cambios de LR.

---

# 127. Ablations obligatorias

Cada claim necesita ablation.

### Shared state

```text
full cross encoder
vs split state
```

### Option mixer

```text
none
vs DeepSets
vs attention
```

### Distillation

```text
hard labels
vs soft teacher
```

### Brier

```text
CE
vs CE+Brier
```

### Permutation loss

on/off.

### Unknown

explicit option vs trust head.

### Compression

raw tokens vs 64/128/256 memories.

### Backbone

top candidates.

---

# 128. Criterios GO / NO-GO tras POC

Continuar arquitectura propia si se cumplen al menos:

1. quality útil en 5 familias;
2. dynamic labels reales;
3. split-state mantiene ~95% o más de la calidad del full cross-encoder como objetivo orientativo;
4. multi-question latency mejora claramente;
5. calibration puede corregirse;
6. unknown funciona;
7. 150–250M es suficiente para tareas no triviales.

Si no:

### Caso A

Split-state pierde demasiada calidad.

→ probar shallow bidirectional fusion/retrieval.

### Caso B

Encoder no generaliza.

→ mayor backbone/distillation.

### Caso C

Latency sigue alta.

→ Ettin smaller / LFM / compression / quantization.

### Caso D

Small decoder first-token supera mucho la calidad.

→ considerar decoder-to-encoder conversion o hybrid.

---

# 129. Milestone M0 — 2–3 días

Resultados:

- hardware inventory;
- repos clonados;
- baseline benchmark harness;
- HuffPost adapter;
- Banking77 adapter.

No escribir todavía arquitectura compleja.

---

# 130. Milestone M1 — primera semana

Resultados:

- backbone bake-off preliminar;
- simple dynamic-option head;
- 3 datasets;
- p50/p95 por máquina;
- primera calibration report.

Decisión:

```text
top-2 backbones
```

---

# 131. Milestone M2 — segunda semana

Resultados:

- shared-state V1;
- true cache;
- multi-Q benchmark;
- BoolQ;
- Civil Comments;
- HelpSteer2;
- option permutation tests.

Éste es el primer punto donde sabremos si la idea central funciona.

---

# 132. Milestone M3 — semanas 3–4

Resultados:

- hard-negative engine;
- unknown/OOD;
- distillation;
- reasoning datasets;
- Spanish;
- benchmark firewall;
- ONNX export inicial.

Decisión:

```text
¿POC demuestra ventaja real?
```

---

# 133. Milestone M4 — semanas 5–6

Resultados:

- state compression;
- INT8;
- runtime optimization;
- MLX experiment si procede;
- gold human pilot;
- eval on held-out task families.

---

# 134. Milestone M5 — research release

Resultados:

- checkpoint;
- inference library;
- API;
- model card;
- benchmark suite;
- dataset manifests;
- calibration report;
- hardware results;
- limitations.

---

# 135. Qué NO hacer en el primer mes

- pretrain foundation model desde cero;
- MoE;
- custom CUDA kernels;
- distributed multi-node;
- million-dollar-style synthetic corpus;
- RL complejo;
- 20 tipos de output;
- browser/WebGPU antes de demostrar calidad;
- quantization 4-bit antes del baseline;
- optimizar benchmark antes de architecture proof.

---

# 136. Riesgo 1 — encoder semantic ceiling

Problema:

Los encoders pequeños pueden clasificar muy bien pero fallar en razonamiento nuevo.

Mitigaciones:

- modern backbone;
- distillation;
- reasoning mix;
- explicit state;
- cross-attention;
- cascade to LLM.

No intentar ocultar este límite.

---

# 137. Riesgo 2 — calibration sólo in-domain

Un ECE bonito sobre random split puede ser engañoso.

Mitigaciones:

- domain holdouts;
- task-family holdouts;
- time split;
- label paraphrases;
- OOD;
- risk-coverage;
- human gold.

---

# 138. Riesgo 3 — dataset contamination

La mayoría de foundation models ya han visto datasets públicos durante pretraining.

No podemos eliminar esa posibilidad.

Pero sí podemos evitar contaminar **nuestro fine-tune** y evaluar transferencia en:

- datasets nuevos;
- synthetic programmatic;
- custom human;
- time-separated data;
- adversarial transforms.

---

# 139. Riesgo 4 — teacher contamination

Un teacher puede haber memorizado benchmarks.

No generar training examples directamente a partir de GPQA/MMLU-Pro/SimpleQA.

Mantener firewall.

---

# 140. Riesgo 5 — teacher confidence falsa

Los números `0.87` de un LLM no son probabilidad ground-truth.

Mitigaciones:

- teacher ensemble;
- permutation consistency;
- ground truth whenever available;
- human calibration set;
- train soft labels como señal, no verdad absoluta.

---

# 141. Riesgo 6 — option dependence

Softmax fuerza suma 1.

Si todas las opciones son malas, puede dar 0.9 a la menos mala.

Por eso `unknown` es imprescindible.

---

# 142. Riesgo 7 — option set manipulation

Añadir/remover options puede cambiar distribuciones.

A veces es correcto.

Necesitamos tests para distinguir:

```text
rational contextual change
```

de:

```text
arbitrary positional/set instability
```

---

# 143. Riesgo 8 — long context illusion

Un modelo que acepta 8K no significa que use bien evidencia a token 7,900.

Necesitamos positional evidence benchmark.

---

# 144. Riesgo 9 — export

Custom masks / attention pueden funcionar en PyTorch y exportar mal a ONNX/CoreML.

Por eso la arquitectura split con módulos simples es preferible a una máscara exótica gigante.

---

# 145. Riesgo 10 — MPS performance

Una operación custom o poco soportada puede hacer que MPS caiga a CPU.

Benchmark por módulo y vigilar device transfers.

---

# 146. Riesgo 11 — licencia

Puede invalidar meses de trabajo.

Mantener desde día 1:

```text
commercial clean checkpoint
```

separado de:

```text
research maximum-quality checkpoint
```

---

# 147. Risk register

| Riesgo | Prob. | Impacto | Mitigación |
|---|---:|---:|---|
| encoder no razona suficiente | alta | alta | distill + cascade |
| split state pierde interacción | media | alta | shallow fusion/retrieval |
| OOD overconfidence | alta | alta | unknown + trust + conformal |
| dataset artifacts | alta | media | transforms + held-out families |
| license contamination | media | alta | manifests/fences |
| ONNX custom op | media | media | simple modules + export tests early |
| Apple speed < expected | media | media | MLX/CoreML/INT8 |
| CUDA VRAM baja | media | media | smaller backbone/LoRA |
| synthetic labels poor | alta | media | ensemble/human QA |
| benchmark leakage | alta | alta | eval firewall |

---

# 148. Decisión sobre HuffPost

**Sí usar.**

Pero su papel debe ser:

```text
architecture smoke test
+
dynamic options
+
hard negatives
+
K scaling
```

No:

```text
core intelligence dataset
```

La primera demo debería poder entrenarse sobre HuffPost porque permitirá verificar rápido que el sistema aprende.

Después, añadir Banking77/BoolQ/CivilComments/HelpSteer2 antes de sacar conclusiones.

---

# 149. Decisión sobre “hacerlo nosotros desde cero”

Hay tres significados de “desde cero”.

## Código de arquitectura propio

**Sí.**

Es razonable.

## Head/fusion/training propio sobre encoder pretrained

**Sí. Recomendado.**

## Foundation encoder pretraining desde pesos aleatorios

**No inicialmente.**

Es el peor uso de nuestro hardware y tiempo.

---

# 150. Decisión sobre reutilizar repos

No forkear un clon entero como base salvo que acelere días de trabajo.

Reutilizar:

- GLiClass: dynamic labels/data/scorers;
- kev: branch/pointer concepts;
- Laya/Verdict: training/calibration ideas;
- ModernBERT/FlexBERT: encoder implementation;
- Ettin: scale experiments;
- MLX examples: Apple runtime;
- ORT: deployment.

Nuestro repo debe ser arquitectónicamente limpio desde el principio.

---

# 151. Primera arquitectura a codificar — pseudocódigo

```python
S = state_encoder(state_tokens)

Q, O = question_encoder(question_tokens, option_tokens)

QO = fusion(Q, O, memory=S)

O = option_mixer(O, question=QO.question)

logits = pointer_head(QO.question, O)

probs = calibrated_softmax(logits)
```

Batch multi-question:

```python
S = state_encoder(state)

branches = question_encoder(all_questions)
branches = cross_attend(branches, S)

return decision_heads(branches)
```

Ésta debe ser la unidad conceptual mínima.

---

# 152. Una optimización importante: option cache

En routing, los labels suelen repetirse:

```text
calendar
email
web
filesystem
slack
...
```

Pre-encode:

```text
option text → option base embedding
```

Cachearlo.

Sólo la capa contextual/cross-attention debe recalcularse.

Esto puede reducir aún más la latencia.

---

# 153. Otra optimización: label hierarchy

Si existen 1,000 tools:

```text
top-level:
  communication
  files
  finance
  travel
  code
```

Primero seleccionar grupo, luego opciones.

GLiClass ya explora hierarchical labels; podemos aprovechar la idea.

---

# 154. Otra mejora: cardinality-aware training

La entropía natural cambia con K.

No comparar directamente:

```text
max_prob=0.7 with K=2
```

contra:

```text
max_prob=0.7 with K=50
```

Entrenar/evaluar calibration por K.

Posiblemente alimentar:

```text
log(K)
```

como feature sólo si demuestra mejora, porque puede introducir shortcuts.

Preferible calibrar externamente por buckets.

---

# 155. Otra mejora: semantic equivalence

Dos opciones pueden significar casi lo mismo.

Detectar:

```text
duplicate/near-duplicate options
```

y:

- fusionarlas;
- reducir confidence;
- o devolver ambiguity flag.

Este problema importa mucho en tool routing generado dinámicamente.

---

# 156. Otra mejora: contradiction detector

Auxiliary head:

```text
Does state contain mutually inconsistent evidence relevant to question?
```

Puede mejorar abstención.

No P0.

---

# 157. Otra mejora: state provenance

El state podría tener segmentos con source IDs.

Ejemplo:

```text
[calendar]
...
[email]
...
[user]
...
```

Añadir source/type embeddings.

Muy útil para agentes.

---

# 158. Otra mejora: freshness/time embeddings

Para decisiones con state temporal:

```text
event timestamp
```

podría entrar como metadata numérica/embedding.

No mezclar esto en V1 general.

---

# 159. Otra mejora: structured state

No convertir todo JSON ciegamente a texto si la estructura es importante.

V2 puede permitir:

```text
text tokens
+
typed fields
+
numeric features
```

fusionados en state memory.

---

# 160. Otra mejora: uncertainty from ensembles

En deployment high-risk:

```text
2 tiny checkpoints
```

pueden ser más útiles que uno mayor.

Disagreement = uncertainty.

Pero duplica latency; usar sólo donde importe.

---

# 161. Otra mejora: snapshot ensembles durante training

Guardar varios checkpoints cercanos y evaluar si combinar logits mejora calibration sin necesidad de modelos distintos.

No MVP.

---

# 162. Otra mejora: stochastic depth / layerdrop

Puede entrenar robustez a early exit y permitir runtime adaptable.

Investigar después de baseline.

---

# 163. Otra mejora: knowledge distillation de capas

No sólo logits.

Teacher encoder/LLM-derived embeddings pueden enseñar:

```text
state semantic representation
option representation
attention alignment
```

Pero logits distillation es mucho más simple.

Prioridad baja hasta probar soft-label KD.

---

# 164. Otra mejora: decoder-to-encoder distillation

Teacher grande:

```text
state + question + options
```

Produce:

- answer distribution;
- relevance of state spans;
- option similarities.

Student aprende las tres señales.

Esto puede compensar la falta de query-conditioned state encoding.

---

# 165. Span relevance auxiliary loss

Si podemos obtener/crear evidence spans:

```text
which state tokens support this answer?
```

Entrenar un token relevance head.

Después usarlo para:

- state pruning;
- explainability;
- query-dependent retrieval.

Muy valioso para long context.

---

# 166. Explicabilidad sin generar texto

Podemos devolver:

```text
top relevant spans
```

No rationale generada.

Esto mantiene latencia baja y facilita debugging.

---

# 167. Benchmark de state reuse que debemos publicar

Caso:

```text
state: 2K tokens
Q: 1, 2, 5, 10, 20, 50
K: 4
```

Graficar:

```text
latency vs Q
```

Comparar:

- Laya-like reencode;
- small decoder;
- nuestro shared state.

Éste puede ser el gráfico principal del proyecto.

---

# 168. Benchmark de Pareto que debemos publicar

Scatter:

```text
x = p50 latency
y = quality
bubble = memory
```

Para:

- Ettin 32M
- Ettin 68M
- Ettin 150M
- ModernBERT
- NeoBERT
- LFM2.5
- Verdict
- Laya
- kev
- Qwen/Gemma baseline

No esconder modelos que ganen.

---

# 169. Benchmark de calibration

Reliability diagram:

```text
predicted confidence
vs
actual accuracy
```

Por:

- dataset;
- K;
- language;
- in-domain/OOD;
- quantized/non-quantized.

---

# 170. Release criteria

No llamar al modelo “calibrated” sólo porque tiene temperature scaling.

Requerir:

- held-out calibration metrics;
- reliability plots;
- OOD results;
- risk-coverage;
- methodology pública.

---

# 171. Naming

Evitar usar `Jev` como nombre del proyecto/modelo.

Usarlo sólo como referencia conceptual.

Nombre interno provisional:

```text
system-one-lab
```

El producto final puede nombrarse después.

---

# 172. Priority backlog condensado

## P0 — ahora

- hardware inventory;
- baseline harness;
- HuffPost;
- Banking77;
- BoolQ;
- Civil Comments;
- HelpSteer2;
- backbone bake-off;
- simple dynamic-option model;
- shared-state V1;
- calibration;
- latency breakdown.

## P1

- hard negatives;
- unknown/OOD;
- distillation;
- MASSIVE/español;
- LogiQA/ReClor;
- ONNX INT8;
- long state;
- gold pilot.

## P2

- latent compression;
- fast-path reranker;
- early exit;
- MLX port;
- conformal risk;
- multilingual expansion.

## P3 research

- MoE;
- from-scratch tiny;
- decoder→encoder conversion;
- continual/online;
- custom kernels.

---

# 173. Mi recomendación concreta para el primer experimento

No empezar con el modelo más sofisticado.

## Run A

```text
Backbone: Ettin-68M
Dataset: HuffPost
Architecture: dynamic option markers + pointer scorer
K: random 2–32
```

Objetivo:

```text
proof that dynamic choice works
```

## Run B

```text
Backbone: ModernBERT-base
same everything
```

## Run C

```text
Backbone: LFM2.5-230M
same everything
```

Medir.

Después sólo top-2 pasan a:

```text
Banking77 + BoolQ + CivilComments + HelpSteer2
```

---

# 174. Segundo experimento decisivo

Comparar:

### Full cross encoder

```text
[state + question + options] × Q
```

contra:

### Shared-state

```text
state encoder once
+ 2 fusion layers/question
```

Si el shared-state conserva calidad y escala claramente mejor, tenemos una arquitectura interesante.

Si no, investigar por qué antes de añadir más datasets.

---

# 175. Tercer experimento decisivo

Hold out un dataset entero.

Ejemplo:

```text
train:
HuffPost
Banking77
BoolQ
Civil
HelpSteer

test zero-shot:
CLINC / new intent set
```

Las labels se pasan como texto.

Si falla completamente, seguimos teniendo un multi-task classifier, no un decision foundation model.

---

# 176. Cuarto experimento decisivo

Teacher distillation.

Mismo checkpoint:

```text
hard-label only
vs
teacher distributions
```

Eval:

- unseen labels;
- unseen task;
- NLL;
- Brier;
- OOD.

Sólo escalar synthetic data si mejora.

---

# 177. Quinto experimento decisivo

State caching.

En una sesión:

```text
encode state once
run 100 different question packs
```

Medir warm-state p50.

Si esto llega a pocos ms por pregunta/pack, ahí aparece una ventaja práctica que un decoder generalista tiene difícil igualar.

---

# 178. Qué considero éxito técnico

No hace falta “batir Jev” para que el proyecto sea útil.

Un éxito sería:

```text
100–250M parameters
local
<20–40 ms common path
true state reuse
dynamic labels
good Spanish/English
strong calibration
OOD abstention
simple API
commercially usable weights
```

y que un LLM sólo sea necesario en los casos difíciles.

Eso ya sería una primitive muy valiosa para sistemas de agentes.

---

# 179. Fuentes técnicas principales

## Jev / System One

- TypeSafe launch:  
  https://typesafe.ai/blog/introducing-system-one-models-and-jev
- TypeSafe:  
  https://typesafe.ai/
- Community alternatives tracker:  
  https://systemonemodels.org/examples/alternatives/

## Jev-like implementations

- Laya:  
  https://github.com/NandhaKishorM/laya
- Laya model:  
  https://huggingface.co/convaiinnovations/laya
- Verdict/OpenJev model:  
  https://huggingface.co/heman10x/rlcd-modernbert-151m
- kev:  
  https://github.com/jaredpalmer/kev
- jevlike:  
  https://github.com/vinnylarouge/jevlike
- open-jev:  
  https://github.com/daseinlabs/open-jev
- NanoJev:  
  https://github.com/TianyuCodings/NanoJev
- openjev-sglang:  
  https://github.com/ekzhang/openjev-sglang

## Encoders

- LFM2.5 Encoders blog:  
  https://www.liquid.ai/blog/lfm2-5-encoders
- LFM2.5-Encoder-230M:  
  https://huggingface.co/LiquidAI/LFM2.5-Encoder-230M
- LFM2.5-Encoder-350M:  
  https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M
- Ettin 150M:  
  https://huggingface.co/jhu-clsp/ettin-encoder-150m
- Ettin family:  
  https://huggingface.co/models?other=ettin
- NeoBERT:  
  https://huggingface.co/chandar-lab/NeoBERT
- NeoBERT paper:  
  https://arxiv.org/abs/2502.19587
- ModernBERT base:  
  https://huggingface.co/answerdotai/ModernBERT-base
- ModernBERT large:  
  https://huggingface.co/answerdotai/ModernBERT-large
- ModernBERT paper:  
  https://aclanthology.org/2025.acl-long.127/
- ModernBERT/FlexBERT code:  
  https://github.com/answerdotai/modernbert
- mmBERT:  
  https://huggingface.co/jhu-clsp/mmBERT-base
- GLiClass:  
  https://github.com/Knowledgator/GLiClass
- GLiClass C/ONNX:  
  https://github.com/Knowledgator/GLiClass.c

## Architectural theory

- Pointer Networks:  
  https://arxiv.org/abs/1506.03134
- Deep Sets:  
  https://arxiv.org/abs/1703.06114
- Set Transformer:  
  https://arxiv.org/abs/1810.00825
- ColBERT:  
  https://github.com/stanford-futuredata/ColBERT
- Hydragen:  
  https://arxiv.org/abs/2402.05099
- Hydragen repo:  
  https://github.com/ScalingIntelligence/hydragen
- DeFT:  
  https://github.com/LINs-lab/DeFT
- DeeBERT / early exit:  
  https://arxiv.org/abs/2004.12993
- Calibration / temperature scaling:  
  https://arxiv.org/abs/1706.04599

## Risk / conformal

- Selective Conformal Risk Control:  
  https://arxiv.org/abs/2512.12844
- Conformal Selective Prediction with General Risk Control:  
  https://arxiv.org/abs/2603.24704
- Conformal Risk Control for Non-Monotonic Losses:  
  https://arxiv.org/abs/2602.20151

## Datasets

- HuffPost News Category:  
  https://www.kaggle.com/datasets/rmisra/news-category-dataset
- HuffPost HF mirror:  
  https://huggingface.co/datasets/khalidalt/HuffPost
- Banking77:  
  https://huggingface.co/datasets/PolyAI/banking77
- CLINC150:  
  https://huggingface.co/datasets/DeepPavlov/clinc150
- MASSIVE:  
  https://huggingface.co/datasets/AmazonScience/massive
- BoolQ:  
  https://huggingface.co/datasets/google/boolq
- ANLI:  
  https://huggingface.co/datasets/facebook/anli
- LogiQA2:  
  https://huggingface.co/datasets/datatune/LogiQA2.0
- ReClor:  
  https://huggingface.co/datasets/sxiong/ReClor
- Civil Comments:  
  https://huggingface.co/datasets/google/civil_comments
- GoEmotions:  
  https://github.com/google-research/google-research/tree/master/goemotions
- SST-5:  
  https://huggingface.co/datasets/SetFit/sst5
- HelpSteer2:  
  https://huggingface.co/datasets/nvidia/HelpSteer2
- FEVER:  
  https://huggingface.co/datasets/fever/fever
- MMLU-Pro:  
  https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
- GPQA:  
  https://huggingface.co/datasets/Idavidrein/gpqa
- SimpleQA:  
  https://huggingface.co/datasets/OpenEvals/SimpleQA
- MuSR:  
  https://huggingface.co/datasets/TAUR-Lab/MuSR
- RewardBench2:  
  https://huggingface.co/datasets/allenai/reward-bench-2
- ARC:  
  https://huggingface.co/datasets/allenai/ai2_arc
- OpenBookQA:  
  https://huggingface.co/datasets/allenai/openbookqa

## Runtime

- PyTorch MPS:  
  https://docs.pytorch.org/docs/stable/notes/mps.html
- Apple MLX:  
  https://github.com/ml-explore/mlx
- MLX examples:  
  https://github.com/ml-explore/mlx-examples
- ONNX Runtime CoreML:  
  https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
- ONNX Runtime CUDA:  
  https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- ONNX Runtime TensorRT:  
  https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
- Optimum ONNX quantization:  
  https://huggingface.co/docs/optimum-onnx/onnxruntime/usage_guides/quantization

## Paid / data generation

- Google Gemini API pricing:  
  https://ai.google.dev/gemini-api/docs/pricing
- OpenAI API pricing:  
  https://developers.openai.com/api/docs/pricing
- Prolific:  
  https://www.prolific.com/pricing

---

# 180. Final architectural recommendation

Si tuviera que iniciar el repo hoy, sin más investigación previa, lo haría así:

```text
1. Benchmark Ettin-68M, Ettin-150M, ModernBERT-base y LFM2.5-230M.

2. Elegir top-2.

3. Construir un baseline full cross-encoder con dynamic option pointer head.

4. Construir inmediatamente después la versión real:
      state_encoder(state) ONCE
      +
      lightweight question encoder
      +
      2-layer Q→state cross attention
      +
      1-layer option mixer
      +
      pointer head
      +
      unknown/trust head

5. Entrenar primero con:
      HuffPost
      Banking77
      BoolQ
      Civil Comments
      HelpSteer2

6. Demostrar state-reuse advantage antes de escalar datos.

7. Añadir:
      hard negatives
      missing-answer/OOD
      MASSIVE Spanish
      reasoning mix
      soft teacher distillation

8. Reservar:
      MMLU-Pro
      GPQA
      SimpleQA
      MuSR
      RewardBench2
   sólo para evaluación.

9. Quantizar INT8 y exportar ONNX únicamente cuando la arquitectura sea estable.

10. Añadir LLM fallback; no intentar que el modelo pequeño resuelva todo.
```

La tesis del proyecto debería permanecer muy clara:

> **No estamos construyendo un LLM que genera más rápido. Estamos construyendo una red especializada en convertir estado + preguntas + opciones dinámicas directamente en decisiones probabilísticas calibradas, reutilizando el estado y evitando por diseño el coste de generación autoregresiva.**

Si esa tesis se mantiene durante los experiments, el proyecto tiene una dirección técnicamente diferenciada. Si terminamos pasando `state+prompt` repetidamente por un Qwen/Gemma y leyendo logits, habremos construido un wrapper rápido, no el sistema que buscamos.
