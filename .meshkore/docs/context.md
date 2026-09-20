---
title: Context — jev-clone
updated: 2026-09-19
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

## Sources

- `tmp/jev_like_system_one_plan_2026-09-19.md` (plan maestro, 180 §)
- `tmp/JEV_CLONE_TECH_STACK_2026-09-19.md` (stack, 41 §)
- `.meshkore/docs/source-register.md` (fuentes canónicas, licencias,
  transformación al schema y uso train/eval)
