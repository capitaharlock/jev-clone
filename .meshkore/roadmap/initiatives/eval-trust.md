---
reactivated: 2026-09-20
id: eval-trust
title: Evaluation, baselines and trust
status: done
owner: architect-master
modules:
  - eval
created: 2026-09-19
updated: 2026-09-20
completed_at: 2026-09-20T00:40:09.447Z
commit_sha: 8865912dc172347c00edc17bc2471ad5801aeafb
---
# Evaluation, baselines and trust

La ventaja frente a un classifier rápido es calibración real:
reconocimiento de baselines (I0), harness de métricas + latencia,
temperature scaling, abstención/OOD con curvas risk-coverage, firewall
anti-contaminación y stress tests propios (shuffle, siblings, OOD…).

Segundo tramo de la cadena V1. Se selecciona junto a las cinco iniciativas;
sus gates reciben los manifests de datos y fijan los baselines antes de que
el modelo pueda elegir una arquitectura.

Cubre plan §§25–26, 42, 60–66, 71, 120, 128, 167–170.

## Done when

- Report de calibración (ECE, Brier, reliability plots) in-domain y OOD.
- Curva risk-coverage con abstención claramente mejor que azar.
- Firewall de contaminación activo y benchmarks versionados.
- GO/NO-GO del POC decidido con los criterios del §128.
- Un dataset completo retenido prueba transferencia semántica; si falla, el
  sistema se clasifica honestamente como multi-task classifier.
