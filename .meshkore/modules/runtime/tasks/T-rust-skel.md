---
id: T-rust-skel
title: Workspace and progressive verification harness
status: done
priority: high
owner: unassigned
category: runtime
initiative: local-runtime
created: 2026-09-19
updated: 2026-09-19
completed_at: 2026-09-19T22:38:34.931Z
resolved_by: A004
resolved_by_conv: general-09192230
commit_shas: ['1f7c66d7c30dacfe8f6356f86ee63dd41c471ac6']
---
# Workspace and progressive verification harness

Esqueleto `crates/{jev-core,jev-model,jev-runtime,jev-server,jev-cli,
jev-bench,jev-format}` con contrato de artefacto (`config.json`,
`tokenizer.json`, `model.safetensors`, `calibration.json`, `manifest.json`,
`LICENSE`) y harness de paridad PyTorch↔Candle (states cortos/largos, 2–32
opciones, 1–N preguntas, choice/boolean × FP32/FP16; INT8 cuando exista).
También crea desde el primer commit el gate runner
`python training/python/tools/gate.py --task <task-id>`. Cada gate ejecuta la
regresión acumulada, guarda `artifacts/gates/<task-id>/gate.json` con commit,
configuración, hashes y métricas, y falla si el artefacto upstream no es PASS
para el mismo linaje. CI: fmt, clippy, tests Python/Rust CPU + paridad; GPU
separado y no bloqueante cuando el dispositivo no existe.

Fuente: stack §§3, 29–30, 36–37.

> assumption: Candle primero; Burn solo si Candle bloquea de forma
> material (decisión ya tomada en el stack §4).

## Verification gate

- `T-rust-skel` es el único gate sin predecesor: crea el runner, fixtures
  deterministas y el formato versionado de `gate.json`.
- El self-test demuestra que un test rojo, un hash alterado o un gate upstream
  ausente producen salida distinta de cero.
- `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`,
  `cargo test --workspace` y los tests Python corren en CPU sobre un artefacto
  diminuto PyTorch↔Candle.
- Salida obligatoria: `artifacts/gates/T-rust-skel/gate.json` con `pass: true`.

## Done when

- Workspace compila con CI verde en CPU.
- Paridad dentro de tolerancia en toda la matriz del stack §30.
- Ninguna release candidata sale sin pasar el harness.
- El manifest enlaza hashes de pesos, tokenizer, config, calibrador y dataset
  manifest; un mismatch impide cargar el artefacto.
- El runner de gates rechaza continuar cuando el gate anterior falta, falla o
  pertenece a otro commit/linaje de artefactos.

## Resolution

No. #cloud-release (`next`, 0/2: T-cloud-api y T-release en `next`) y #data-foundation (`active`, 0/4: T-data-p0, T-data-schema, T-gold, T-hardneg en `next`). La única completada del cluster es #T-rust-skel.

— Sin tareas completadas en esta tanda

2.3M tokens
