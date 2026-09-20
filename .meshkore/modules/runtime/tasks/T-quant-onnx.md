---
id: T-quant-onnx
title: Quantization and export backends
status: done
priority: medium
owner: general-09192230
category: runtime
initiative: local-runtime
depends_on:
  - T-state-cache
created: 2026-09-19
updated: 2026-09-20
completed_at: 2026-09-20T02:16:53.950Z
resolved_by: A004
resolved_by_conv: general-09192230
commit_shas: ['1e6d24fa67d66f98af24439999a72d3ee7f1246c']
---
# Quantization and export backends

Orden estricto: FP16/BF16 referencia → INT8 dinámico → INT8 estático
(con recalibración) → INT4 experimental. ONNX como derivado (shapes
dinámicas K/Q o buckets), EPs CUDA/CoreML, TensorRT si el benchmark lo
justifica, comparador MLX en Apple. Kernels custom solo tras profiling.

Fuente: stack §§10–12, 14, 40; plan §§59, 88–90.

## Verification gate

- Requiere PASS de `T-state-cache`; FP16/BF16 es la referencia congelada antes
  de generar cualquier derivado.
- Golden vectors y stress suite comparan cada derivado con Candle canónico;
  INT8 repite calibración y ONNX cubre buckets K/Q, shapes inválidas y export
  round-trip.
- Cada backend se conserva sólo si mejora latencia/memoria en el hardware donde
  existe y respeta umbrales de accuracy, NLL, Brier y ECE predefinidos.
- `T-quant-onnx` puede pasar seleccionando FP16 y rechazando INT8/ONNX si no
  aportan; la decisión y los benchmarks quedan en el gate, sin bloquear cloud.

## Done when

- INT8 sin degradación material de calidad ni calibración.
- Benchmark Candle vs ONNX vs TensorRT por plataforma publicado.
- Ningún kernel custom sin perfil que lo justifique.
- Cada derivado referencia el checkpoint canónico y repite paridad,
  calibración y stress suite antes de ser releaseable.

## Resolution

8/17: `T-option-mixer` PASS. Sigue.

— Sin tareas completadas en esta tanda

**Commit** `1e6d24fa6` · 60 files · 16.5M tokens

**Files changed (60):**
- `.DS_Store`
- `.clinerules`
- `.cursor/rules/meshkore.mdc`
- `.gitignore`
- `.meshkore/STANDARD_VERSION`
- `.meshkore/docs/INDEX.md`
- `.meshkore/docs/context.md`
- `.meshkore/docs/governance.md`
- `.meshkore/docs/source-register.md`
- `.meshkore/modules/cloud/tasks/T-cloud-api.md`
- `.meshkore/modules/cloud/tasks/T-release.md`
- `.meshkore/modules/data/tasks/T-data-p0.md`
- `.meshkore/modules/eval/tasks/T-calib.md`
- `.meshkore/modules/eval/tasks/T-firewall.md`
- `.meshkore/modules/eval/tasks/T-recon.md`
- `.meshkore/modules/general/README.md`
- `.meshkore/modules/model/tasks/T-dist-train.md`
- `.meshkore/modules/project/README.md`
- `.meshkore/modules/runtime/tasks/T-local-infer.md`
- `.meshkore/modules/runtime/tasks/T-quant-onnx.md`
- `.meshkore/modules/runtime/tasks/T-rust-skel.md`
- `.meshkore/modules/runtime/tasks/T-state-cache.md`
- `.meshkore/public/AGENT_INSTRUCTIONS.md`
- `.meshkore/public/README.md`
- `.meshkore/public/cluster.yaml`
- `.meshkore/roadmap/initiatives/cloud-release.md`
- `.meshkore/roadmap/initiatives/decision-model.md`
- `.meshkore/roadmap/initiatives/eval-trust.md`
- `.meshkore/roadmap/initiatives/local-runtime.md`
- `.meshkore/team/api-developer.md`
- `.meshkore/team/architect-master.md`
- `.meshkore/team/commit-pr-reviewer.md`
- `.meshkore/team/consultant.md`
- `.meshkore/team/deployer.md`
- `.meshkore/team/developer.md`
- `.meshkore/team/roadmap-orchestrator.md`
- `.meshkore/team/tester.md`
- `.meshkore/team/ui-developer.md`
- `.meshkore/team/ui-reviewer.md`
- `AGENTS.md`
- `CLAUDE.md`
- `Cargo.lock`
- `Cargo.toml`
- `GEMINI.md`
- `artifacts/gates/T-bakeoff/gate.json`
- `artifacts/gates/T-bakeoff/report.json`
- `artifacts/gates/T-calib/gate.json`
- `artifacts/gates/T-curriculum/gate.json`
- `artifacts/gates/T-data-p0/gate.json`
- `artifacts/gates/T-data-schema/gate.json`
- `artifacts/gates/T-distillation/gate.json`
- `artifacts/gates/T-firewall/gate.json`
- `artifacts/gates/T-gold/gate.json`
- `artifacts/gates/T-hardneg/gate.json`
- `artifacts/gates/T-local-infer/gate.json`
- `artifacts/gates/T-option-mixer/gate.json`
- `artifacts/gates/T-option-mixer/report.json`
- `artifacts/gates/T-quant-onnx/gate.json`
- `artifacts/gates/T-recon/gate.json`
- `artifacts/gates/T-rust-skel/gate.json`
