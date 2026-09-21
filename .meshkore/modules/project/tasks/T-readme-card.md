---
id: T-readme-card
title: README, quickstart de 3 comandos y model card ejecutable
status: backlog
priority: medium
owner: unassigned
category: project
initiative: oss-release
depends_on:
  - T-repo-clean
  - T-release-gate
created: 2026-09-21
updated: 2026-09-21
---

# README, quickstart de 3 comandos y model card ejecutable

Hallazgo H: no hay `README.md`, ni quickstart, ni pesos publicados, ni model
card ejecutable. `.meshkore/docs/model-card.md` existe pero describe un
modelo que no es el que se entrena.

Trabajo:

1. `README.md` con: qué es (clon del modelo de decisión JEV — System One, no
   autoregresivo, decisión tipada + probabilidad calibrada en 70-500 ms), qué
   **no** es, y el estado real del proyecto sin adornos.
2. **Quickstart de 3 comandos**: clonar → descargar pesos → decidir sobre un
   ejemplo. Los tres se ejecutan en CI en una máquina limpia; si alguno
   necesita un cuarto, el quickstart está mal.
3. `model-card.md` reescrito sobre el checkpoint real de `#T-train-real`:
   arquitectura (pointer head sobre texto de opción), datos con sus
   licencias, métrica primaria **seen/unseen** de `#T-unseen-labels`,
   calibración, limitaciones conocidas, y qué datasets están en firewall
   (LogiQA/ReClor eval-only) y por qué.
4. La auditoría `.meshkore/docs/audit-2026-09-21.md` y el criterio de release
   se enlazan desde el README: el proyecto publica también lo que encontró
   mal en sí mismo.

Va después de `#T-release-gate`: publicar un model card antes de tener
criterio de release es publicar una opinión.

## Verification gate

- CI ejecuta los 3 comandos del quickstart en un runner limpio y verifica la
  decisión de salida.
- Test: el model card contiene el `model_version` y el sha de los pesos que
  publica; un card sin linaje falla.
- El gate escribe `artifacts/gates/T-readme-card/gate.json` con `pass: true`
  y la salida del quickstart.

## Done when

- Tres comandos llevan de `git clone` a una decisión correcta, verificado en
  CI.
- El model card describe el modelo que realmente existe, con métricas
  seen/unseen y licencias.
- README enlaza auditoría y criterio de release.
