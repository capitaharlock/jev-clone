---
id: T-distillation
title: Distillation and reasoning mix
status: done
priority: medium
owner: unassigned
category: model
initiative: decision-model
depends_on:
  - T-curriculum
created: 2026-09-19
updated: 2026-09-20
completed_at: 2026-09-20T00:21:58Z
resolved_by: A004
resolved_by_conv: general-09192230
commit_shas: ['7f48fd53c7810cb2a66e8402a983ffe97cf4bf13']
---

# Distillation and reasoning mix

Generación local primero con Qwen (p. ej. Qwen3) en este mismo
terminal: convierte/adapta datasets existentes (HuffPost y otros P0) al
schema choice — p. ej. titular→state, pregunta generada, opciones con
hard negatives — y produce masa sintética de dominio (p. ej. zapatos ×
peso/altura/uso) sin depender de APIs. Teachers externos baratos
después (Gemini Flash-Lite, GPT Luna) solo donde Qwen local no llegue,
con pipeline de coste contabilizado, segundo teacher para desacuerdos,
adjudicación con teacher fuerte solo donde discrepan, KL suave
ponderada por confianza y filtro de calidad. Mix de generalización:
LogiQA/ReClor con license fence; ANLI solo research/eval mientras siga NC,
sin copiar CoT. Todo ejemplo conserva teacher/model/version, prompt hash,
coste y términos de uso aplicables. Presupuesto sintético por fases
(local gratis → €0–20 → €20–100 → escalar solo con señal).

Fuente: plan §§45–47, 111–115.

## Verification gate

- Requiere PASS de `T-curriculum`; Qwen local es el teacher por defecto y las
  APIs externas se sustituyen por fixtures si no hay credenciales/presupuesto.
- Contract tests validan schema de teacher, prompt hash, coste, confianza,
  desacuerdo/adjudicación y rechazo de outputs inválidos o contaminados.
- Ablation hard-label vs soft-KL usa idénticos splits/seeds y reporta NLL,
  Brier, OOD y zero-shot; test permanece sellado hasta el informe final.
- `T-distillation` pasa sólo con mejora medida o con decisión documentada de
  conservar hard-labels; una mejora inexistente no detiene el roadmap.

## Done when

- Teacher distributions superan a hard-labels en labels no vistas ( §176).
- Coste por 1K etiquetas medido y dentro del presupuesto de fase.
- Zero-shot a familia de tareas retenida demuestra transferencia (§175).
- Ningún benchmark `eval-only` ni su paráfrasis entra en prompts o outputs
  aceptados por el filtro de contaminación.

## Resolution

#T-distillation con gate PASS (linaje 5bba14f9b1c4): pipeline local-first con contrato teacher completo, desacuerdo/adjudicación, filtro de contaminación sobre prompts y outputs, ablación hard vs soft-KL en splits idénticos → keep-hard-labels documentado (NLL 0.168 vs 0.273) sin bloquear el roadmap; 11 tests verdes.
