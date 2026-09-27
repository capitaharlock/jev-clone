---
id: daily-learning-loop
title: El bucle diario — que mañana el modelo sea mejor que hoy, medido
status: next
owner: architect-master
modules:
  - model
  - eval
  - data
created: 2026-09-27
updated: 2026-09-27
---

# El bucle diario — que mañana el modelo sea mejor que hoy, medido

Decisión del operador (2026-09-27): el objetivo no es un checkpoint, es **un
sistema que mejora cada día**. Esta iniciativa monta ese sistema encima del
piloto de `#cross-encoder-pilot`, y por eso arranca **cuando `#T-ce-finetune`
demuestre que la técnica aprende** (mejora en desarrollo sobre el mismo
checkpoint sin ajustar). Montar un bucle diario alrededor de un entreno que no
aprende es automatizar el gasto.

```
 noche N:  datagen → verify → splits(+train) → trainer(continúa) → evalgate(dev)
           → scoreboard(fila N) → promote si mejora, si no conserva N-1
 el test sellado NO participa: sólo #T-ce-confirm lo abre, una vez
```

## Lo que lo hace honesto

- **Continuación, no reentreno:** el trainer parte de
  `artifacts/checkpoints/ce/current` y registra qué mezcla consumió
  (`#T-loop-trainer`). Cada día se puede auditar qué filas vio el modelo.
- **Una fila por día, append-only:** `artifacts/scoreboard/history.jsonl` con
  las cuatro cifras de la batería de desarrollo (forzada, con abstención, macro
  por familia, contrafactual conjunto), por idioma, con IC95 %, y los cortes
  externos (typed-decisions test contra Jev 0,727 y Laya 0,766; BANKING77
  K=77). El dashboard (`tools/training_monitor.py`, job `dashboard`) pinta la
  curva (`#T-loop-scoreboard`).
- **Regla de promoción escrita antes de correr:** se promueve sólo si la
  diferencia en desarrollo tiene IC95 % que no cruza 0 **y** ninguna familia ni
  idioma cae bajo el checkpoint anterior. Si no, se conserva el anterior y el
  día queda registrado como «sin mejora» con su causa candidata
  (`#T-loop-nightly`).
- **Dev no se agota:** elegir cada día con la misma batería de desarrollo la
  gasta. Se rota: `#honest-eval` prepara cortes de desarrollo nuevos con el
  mismo protocolo, y el sellado sólo se abre en decisiones finales.
- **Dirigido a errores** (después): las familias y cruces donde el modelo
  falla en dev alimentan la generación del día siguiente
  (`#T-loop-error-mining`); y cuando haya profesor con credencial, sus
  probabilidades como recompensa (`#T-loop-rl-jev`, estilo RLCD de Laya).

## Done when

- Existe un único comando encadenado (job canónico) que ejecuta las siete
  etapas y deja rastro por etapa; un fallo en una etapa no promueve nada.
- `history.jsonl` tiene ≥ 7 filas consecutivas reales y el dashboard las pinta.
- La regla de promoción está implementada con test: un día que empeora una
  familia no promueve aunque suba la media.
- El consumo de datos por día es auditable: qué fuentes, cuántas filas, con
  qué pesos, y el embudo generado → verificado → consumido.
