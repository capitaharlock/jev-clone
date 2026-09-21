---
id: oss-release
title: Release pública ejecutable (repo limpio + inferencia local)
status: next
owner: architect-master
modules:
  - project
  - runtime
created: 2026-09-21
updated: 2026-09-21
---

# Release pública ejecutable (repo limpio + inferencia local)

Hallazgo H de `.meshkore/docs/audit-2026-09-21.md`: el repositorio no es
publicable hoy. 9 145 ficheros de `target/` (build debug de Rust) versionados
en git, datasets y `.pkl` por run (~250 MB), sin `README.md`, sin quickstart,
sin pesos, sin model card ejecutable. Y sin backend de inferencia neuronal en
Rust —no hay Candle en `Cargo.toml`— "ejecutar en Metal/CUDA" no existe aún
ni como camino.

Va en `next`, no en `active`: publicar antes de que `#I-decision-rebuild`
produzca un modelo real sería publicar un baseline de bolsa de palabras.

`crates/jev-runtime` (cache de estado con clave compuesta, LRU, batching
dinámico 0-2 ms, backpressure, cancelación) es sólido y se reutiliza tal cual;
lo que falta es el backend que ejecuta los pesos debajo.

## Done when

- `git clone` del repo público pesa < 50 MB y no contiene `target/`,
  `artifacts/runs/` ni datasets.
- Tres comandos del README llevan de clone a una decisión correcta.
- El mismo prompt da el mismo argmax en Metal y en CUDA, con paridad numérica
  verificada contra el runner de referencia.
