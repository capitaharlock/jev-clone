---
title: Context — jev-clone
updated: 2026-09-24
owner: architect-master
---

# Context — jev-clone (`system-one-lab`)

Motor local de decisión semántica tipo Jev / System-One: dado un `STATE` +
N preguntas con opciones dinámicas, devuelve una distribución de
probabilidad por pregunta. Sin chat, sin generación token-a-token.

## Goal

Primitive local (100–250M params, <20–40 ms camino común) con state
codificado una vez, labels dinámicas nunca vistas, buena calibración,
abstención OOD y ES+EN. Un LLM solo entra en cascada para casos difíciles.

## Audience

Sistemas de agentes y apps que necesitan decisiones tipadas, baratas y
auditables (intents, booleanos, scores), on-device o vía API.

## Stack (firme)

- Research/entreno: Python + PyTorch. Producto: Rust + Candle.
- Contrato entre ambos: SafeTensors + `config/tokenizer.json`.
- Apple M4 Max/M5 48 GB → Candle Metal; Windows RTX → Candle CUDA;
  CPU fallback. Comparadores: MLX, ONNX/CoreML (no runtimes paralelos).
- Servo: Axum + Tokio, `encode_state` separado de `decide`, state cache,
  batching dinámico. Cloud: Docker + RunPod Serverless, scale-to-zero.
- Backbones candidatos: Ettin (MIT, familia 17M–1B), ModernBERT, NeoBERT
  (MIT), LFM2.5-Encoder (licencia LFM Open v1.0 — revisar antes de uso
  comercial). No casarse con ninguno hasta el bake-off.

## Non-obvious decisions

- No clonar Jev literal ni preentrenar foundation desde cero; head y
  fusión propios sobre encoder preentrenado.
- Una sola máquina ejecuta cada training run. Si hay tres equipos, se
  reparten runs reproducibles desde una cola compartida; no se sincronizan
  gradientes entre MPS y CUDA. DDP queda condicionado a nodos CUDA homogéneos
  y a demostrar ganancia neta frente a paralelizar experimentos.
- RLCD completo no: empezar con supervised + proper scoring + distill;
  RL solo si hay necesidad real.
- `unknown` explícito y trust/OOD head desde V1, no como parche.
- Benchmarks (MMLU-Pro, GPQA, ARC…) en firewall: nunca entrenar con ellos.
- Nombre `Jev` solo como referencia; el producto se renombra al salir.

> assumption: se mantiene el título de trabajo `jev-clone` y el binario
> `jevclone` hasta decidir el nombre comercial antes de la release.

## Model size tracker (actualizar en cada cambio de backbone)

- 2026-09-20 · baseline real entrenado: TF-IDF + regresión logística (~miles de parámetros, no transformer; `artifacts/runs/20260920T154511Z/`). No es el modelo final.
- 2026-09-20 · bake-off real pendiente de pesos: Ettin-68M (68M) y ModernBERT-base (149M) en `pending_weights`; proxies medidos proxy-S/M/L (dims 256/1024/4096, sin pesos reales).
- Objetivo V1 (plan §178 + data-strategy §13): 100–250M; rango full fine-tune según ganador 68M–400M (Ettin 17M→1B, ModernBERT-large 395M, LFM2.5-230M/350M si pasa licencia).
- Regla: cada vez que cambie el backbone o el head, añadir aquí una línea `fecha · backbone · M params` y repetirla en el chat.

## Objetivo operativo actual (2026-09-21, vigente)

Entrenar a máxima capacidad el motor de decisión (NO un LLM): generar
conjuntos de datos de decisión sin parar y entrenar encoders con ellos,
24h, desacoplado del terminal. Piezas: `tools/data_gen_loop.py` genera
filas schema-universal en `artifacts/data-prefetch/synth-loop.jsonl`
(+ reentreno dirigido + forward test por lote); Q-W-E-N
(`qwen3.6:27b-mlx` vía Ollama :11434) solo inventa plantillas frescas,
nunca toca pesos; `tools/training_monitor.py` reentrena el baseline y
sirve el dashboard. Arranque único: `./start-all.sh [PORT]` (por defecto
8794). Estado 2026-09-21: Q-W-E-N aporta 0 plantillas (hook sin éxito,
pendiente revisar) — la generación programática sí avanza.

## Transición a fase 2 (2026-09-24) — leer antes de tocar entreno o eval

La fase 1 validó el sistema end-to-end en un régimen barato: cabeza pointer
sobre backbone **congelado**, pérdida sobre **3–8 opciones muestreadas**
(`data/optset.py: k_min=3, k_max=8`). Cumplió su función —runtime, corpus,
gates y una arquitectura que aprende— y la primera comparación con protocolo
idéntico al del profesor marca el salto: a 77 vías en BANKING77 damos **0,0123
con azar 0,0130**, porque en régimen de pocas opciones el modelo aprende una
preferencia local, no un ranking del espacio.

Fase 2 entrena y mide lo que el producto promete: el **espacio de etiquetas
completo**, con encoder entrenable. El eval se mueve primero
(`#T-eval-cardinality`) y el entreno detrás (`#full-space-training`). Los
veredictos de fase 1 —el NO-GO de `#T-unfreeze-backbone`, el de `#T-mix-5m`, la
curva de escalado— son válidos en su régimen y **no se heredan** al nuevo: se
vuelven a medir. Detalle completo en `.meshkore/docs/fase-2-espacio-completo.md`.

**Reglas vigentes (R1–R6), resumidas:** se entrena la tarea que se mide · toda
cifra se publica con su azar al lado · ninguna comparación externa sin mismo
protocolo · un GO/NO-GO sólo vale dentro del régimen con el que se midió ·
antes de escalar datos, demostrar la pendiente a 62 k · una referencia externa
en la mesa desde el día 1.

## Sources

- `tmp/jev_like_system_one_plan_2026-09-19.md` (plan maestro, 180 §)
- `tmp/JEV_CLONE_TECH_STACK_2026-09-19.md` (stack, 41 §)
- `.meshkore/docs/source-register.md` (fuentes canónicas, licencias,
  transformación al schema y uso train/eval)
