---
id: T-halt-contam
title: Parar la hemorragia — congelar synth-loop y firewall de benchmarks
status: done
priority: high
owner: unassigned
category: data
initiative: decision-rebuild
depends_on: []
created: 2026-09-21
updated: 2026-09-21
commit: 88ad4e6
completed_at: 2026-09-21T13:05:00.000Z
resolved_by: A003
resolved_by_conv: roadmap-architect-uwgjq
---

# Parar la hemorragia — congelar synth-loop y firewall de benchmarks

Fase 0 de la auditoría (hallazgos C y G). Todo lo que sigue corriendo ahora
mismo añade coste sin añadir información y mantiene un verde falso en el
dashboard. Esta task es de horas, no de semanas, pero bloquea al resto: nada
de lo demás se mide bien mientras `synth-loop` esté en la mezcla.

Estado de partida: `artifacts/data-prefetch/synth-loop.jsonl` tiene 258 700
filas generadas desde **190 esqueletos** (≈70 plantillas × 4 estados); el
esqueleto más frecuente aparece 10 776 veces; crece a ≈358 filas/min y ya
ocupa ~140 MB sin aportar nada desde la fila ~2 000.

Trabajo:

1. Parar el generador: `tools/data_gen_loop.py` (arrancado por
   `start-all.sh`, pid en `artifacts/logs/genloop/gen.pid`). `start-all.sh`
   deja de arrancarlo hasta que `#T-gen-schemas` lo sustituya. Si el loop
   debe seguir vivo en alguna forma, que sea como **job del daemon**, no
   como nohup de una terminal.
2. Congelar `synth-loop.jsonl`: moverlo a
   `artifacts/data-prefetch/quarantine/synth-loop-20260921.jsonl` con un
   `README` al lado que diga por qué está en cuarentena (190 esqueletos,
   split `i % 10`, acc 1,0000 por fuga). No se borra: es la evidencia.
3. Sacar `synth-loop` de `JOBS` en `data/train_baseline.py:26`.
4. Sacar `logiqa` y `reclor` de `JOBS` y **registrarlos en el firewall**
   como eval-only: son benchmarks de razonamiento y el plan los quiere en
   firewall. `data/firewall.py` y `data/leakage.py` ya saben hacerlo; lo
   que falta es el registro, no el detector.
5. Dashboard honesto (`tools/training_monitor.py`, `:8794`): la fila
   synth-loop se marca como contaminada, deja de contar en el titular, y
   ninguna métrica se pinta verde sin un split limpio detrás. El titular
   "1.000 → 1.000 → 1.000" desaparece.

## Verification gate

- `pgrep -f data_gen_loop.py` vacío; `synth-loop.jsonl` ya no está en
  `artifacts/data-prefetch/`.
- Un run de `train_baseline.py` no produce ninguna entrada `synth-loop`,
  `logiqa` ni `reclor` en `metrics.json`.
- `data/test_firewall.py` cubre que una fila de logiqa/reclor en train es
  rechazada como error, no como warning.
- El gate escribe `artifacts/gates/T-halt-contam/gate.json` con `pass: true`,
  el hash del fichero en cuarentena y la lista de jobs resultante.

## Done when

- El generador está parado y `synth-loop.jsonl` en cuarentena con su motivo
  escrito al lado.
- `JOBS` no contiene synth-loop, logiqa ni reclor; logiqa/reclor figuran como
  eval-only en el registro del firewall, con test que lo prueba.
- El dashboard `:8794` no muestra ningún verde sostenido por synth-loop.
