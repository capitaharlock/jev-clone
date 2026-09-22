---
id: T-mix-5m
title: Escala a 5M y distillation soft piloto
status: done
priority: high
owner: unassigned
category: data
initiative: data-training
depends_on:
  - T-mix-1m
created: 2026-09-20
updated: 2026-09-22
completed_at: 2026-09-22T10:11:50.212Z
resolved_by: A003
resolved_by_conv: roadmap-architect-uwgjq
commit_shas: ['ef69953286e9b56ca394c58be96be69c1713505d']
---
# Escala a 5M y distillation soft piloto

> **Resultado: NO-GO (2026-09-22).** 1 M no demostró generalización, así
> que por §§89/136 esta task registra el NO-GO y NO genera 5 M. Gate:
> `artifacts/gates/T-mix-5m/gate.json` (`pass: false`, `verdict: NO-GO`).
> Informe: `artifacts/gates/T-mix-5m/SCALING_NOGO.md`. La regla que impide
> que este gate se vuelva GO mientras los evals de razonamiento estén en el
> azar vive en `tools/mix_5m/nogo.py` y la cubre `data/test_mix_5m.py`.
> En 250 k→1 M la accuracy unseen-label CAYÓ en los dos backbones
> (−0.166556 modernbert-base, −0.189827 ettin-68m, azar 0.165236) y
> ninguno de los 9 cortes de razonamiento (logiqa-mc / logiqa-nli / reclor
> × 3 checkpoints) superó su propio azar por el límite inferior del CI95.
> Hipótesis que la curva sostiene: `backbone.frozen: true` — sólo entrena
> un pointer head del 1.9 % (modernbert-base) / 3.8 % (ettin-68m) de los
> parámetros. Lo que lo volvería GO está en el §5 del informe.

SÓLO si 1 M demuestra generalización (§§87, 136): escalar a 5 M
añadiendo diversidad (más familias, buckets K, idiomas, long context,
OOD, desacuerdos teacher — no duplicar, §87). En paralelo, piloto de
distillation sobre 100 k high-value (§137): soft targets por acuerdo
de N teachers o N stochastic samples — NUNCA un único número
verbalizado como verdad (§52: métodos A/B/C; loss CE(gold) +
λ·KL(teacher‖student)); stop cada 500 k (micro-checkpoint + eval
fija; sin ganancia → cambiar fuente/familia/dificultad, §73);
ablations real-only / +programmatic / +synthetic / full para saber
si el factory aporta (§112); curva 250 k→1 M→5 M→10 M con quality,
NLL, Brier, OOD y latencia (§111). Calibración final con labels
reales held-out, no copiando a Qwen (§53).

Fuentes: data-training §§52–53, 62, 73–75, 87, 111–112, 136–137.

## Verification gate

- Requiere PASS de `T-mix-1m` con ganancia demostrada; si 1 M no
  mejora, esta task registra NO-GO y no genera 5 M (§§89, 136).
- Tests: curva de scaling graficada con los mismos evals fijos;
  ablation muestra el delta de cada capa; soft targets verificados
  por construcción (acuerdo/samples, no verbalización única);
  active generation posterior: el student-in-the-loop propone
  confusiones X↔Y validadas por otro teacher (§§74–75).
- El gate escribe `artifacts/gates/T-mix-5m/gate.json` con `pass: true`,
  curva, deltas de ablation y veredicto de continuar a 10 M; sin ese
  artifact no arranca `#T-data-eval`.

## Done when

- Mix 5 M (si la curva lo justifica) + piloto soft 100 k evaluado.
- Curva 250 k→1 M→5 M y ablations publicadas; decisión 10 M tomada
  con datos, no con intuición.

## Resolution

✓ #data-training #T-mix-5m done · commit `ef69953` en HEAD · tests verdes (el `no-commit` del wake es falso otra vez). Ya estaba `done` en disco — sin reescritura.

A030 hizo lo correcto: **no generó el corpus de 5M**. Registró el NO-GO que el propio gate de la task ordenaba, con los números medidos: de 250k → 1M la accuracy unseen-label **baja** en los dos backbones (−0,167 modernbert-base, −0,190 ettin-68m) y los 9 cortes de razonamiento contienen su propio azar dentro del CI95. Multiplicar filas no arregla un backbone congelado.

═══ Roadmap pass complete ═══
Cola vacía — nada más despachable. #data-training queda con #T-data-eval (recién desbloqueada, era su dep) y #T-mix-10m en backlog; #honest-eval con #T-release-gate esperando tu firma; #oss-release en backlog.

**Falta la task que importa**: el NO-GO dice que el siguiente experimento es un run a **250k con backbone descongelado (o LoRA)**, y no existe ninguna task para eso en el roadmap. No la escribo yo — es trabajo del Architect. Es lo único que puede convertir esto en un modelo que funcione.

<details><summary>Estado detallado de las 4 iniciativas activas</summary>

- **#data-training** (activa) · #T-mix-5m ✓ NO-GO registrado · #T-data-eval `blocked` con la dep ya resuelta (OOD + calibration split + reportes públicos — es cierre de work-stream, no entrenamiento) · #T-mix-10m backlog, sin sentido tras el NO-GO.
- **#honest-eval** (activa) · #T-release-gate `blocked: espera la firma del operador en .meshkore/docs/release-criteria.md (signed_by)`. Acción manual tuya, 1 línea.
- **#oss-release** (activa) · 3 tasks en backlog (#T-repo-clean, #T-candle-infer, #T-readme-card) — paradas a propósito por tu instrucción de no empaquetar release.
- **#train-scaleout** (backlog) · 3 tasks, ninguna cubre el descongelado: #T-dist-train es aceleración multi-dispositivo, no arquitectura.
</details>

<details><summary>deferred-ops + integridad</summary>

- **deferred-ops**: firmar `.meshkore/docs/release-criteria.md` (`signed_by`) para desbloquear #T-release-gate.
- **integridad**: #data-training carga 14 tasks (flag `initiative_too_dense`). Partirla es autoría de roadmap → Architect, no yo.
- **jobs**: `mix1m-curve` terminó (exit 0, 01:41 UTC). Solo queda `training-monitor` vivo.
</details>

— T-mix-5m · el NO-GO al corpus de 5M queda registrado y protegido por test, en vez de quemar días generándolo

2.6M tokens
