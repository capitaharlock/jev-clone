---
id: honest-eval
title: Evaluación honesta y criterio de release
status: active
owner: architect-master
modules:
  - eval
created: 2026-09-21
updated: 2026-09-21
---

# Evaluación honesta y criterio de release

Hallazgos C, F y G de `.meshkore/docs/audit-2026-09-21.md`: la evaluación no
mide la propiedad que define el producto. No existe ninguna métrica sobre
etiquetas **no vistas en entrenamiento**; `eval/calib.py` y el gate de release
calibran un scorer coseno char-3gram sin parámetros, no el modelo; y
`artifacts/gates/T-release/release.json` publica `cohen_kappa: 0.0` junto a
`exact_agree_rate: 0.852` — acuerdo con el profesor indistinguible del azar,
enmascarado por desbalanceo de clases. El split por `i % 10` pone la misma
plantilla en train y en test. LogiQA y ReClor, benchmarks de razonamiento que
el plan quiere en firewall, están entrando en `JOBS` y entrenándose.

Esta iniciativa hace que un número verde signifique algo: split por
plantilla/dominio, métrica primaria sobre etiquetas no vistas, y un criterio
de release explícito **escrito antes de medir**.

`data/firewall.py` y `data/leakage.py` están bien hechos y se reutilizan tal
cual; lo que falta es el registro y el uso, no el detector.

## Done when

- Ningún split del proyecto se genera por índice de fila.
- La métrica primaria publicada es accuracy + ECE sobre etiquetas no vistas
  en entrenamiento, con su intervalo de confianza.
- El criterio de "ya es suficientemente capaz" está escrito, versionado y
  fechado antes de la medición que lo evalúa.
- Ningún gate publica un verde que conviva con `cohen_kappa: 0.0`.
