---
id: T-ce-finetune
title: Aprender de verdad — ajustar con 5.000 decisiones verificadas, ampliar a 20.000 si mejora
status: next
priority: high
owner: unassigned
category: model
initiative: cross-encoder-pilot
depends_on:
  - T-ce-mechanics
  - T-preflight-refs
  - T-episode-splits
created: 2026-09-25
updated: 2026-09-27
---

# Aprender de verdad — ajustar con 5.000 decisiones verificadas, ampliar a 20.000 si mejora

> **Reescrita el 2026-09-27** como guía ejecutable (objetivo único, ver
> `docs/guia-un-solo-objetivo.md`). Paso 4 del piloto. **Es el experimento que
> decide si la técnica aprende.** Todo lo que viene después (`#daily-learning-loop`)
> se monta sobre su GO.

## Contexto — lo que ya está y lo que no

- Está: el scorer compartido (`model/ce_scorer.py`: `Decision`, `Candidate`,
  `render_pairs`, `encode_pairs`, `PairScorer`), el bucle de sobreajuste que lo
  entrena (`training/python/ce_overfit.py`: 60 épocas, MPS, 0,958 en 48 casos)
  y el formato de hipótesis **congelado** (`format_fingerprint()`): el mismo
  en entreno y en evaluación, o el gate no vale.
- Está: la batería de desarrollo (`data/battery_dev.jsonl`, 400 filas, 5
  familias, ES/EN, K=2/3/8, 140 grupos contrafactuales) y las referencias
  medidas sobre ella (`artifacts/gates/T-preflight-refs/`): el listón es
  **nli-nograd 0,6225** forzada, conjunto contrafactual 0,343; pointer = azar.
- No está: **el trainer**. `ce_overfit.py` sólo sabe sobreajustar su conjunto
  fijo. Hay que escribir `training/python/ce_finetune.py`.
- No está (aún): el split de train de episodios (`#T-episode-splits`). La
  cifra oficial lo espera; **el trainer y el smoke no**.

## Qué hacer, en este orden

### A. El trainer (sin GPU, con tests) — se puede hacer HOY

`training/python/ce_finetune.py`, subcomandos `train`, `eval`, `gate`:

- `train --episodes <dir o manifest> --weights minilmv2-l6-mnli-xnli
  --init <checkpoint|none> --out artifacts/checkpoints/ce/<run> --device mps
  --seed S --epochs E --lr-encoder 2e-5 --lr-head 1e-4 --decisions-per-batch 8
  --eval-every <n decisiones> --budget <n decisiones>`.
  - Lee episodios `episode-v1` (usa `data/episode_contract.py` para validar;
    rechaza lo inválido con motivo). Construye `Decision` con **todos** los
    candidatos del episodio; pérdida = CE listwise sobre los K logits del
    scorer (la misma de `ce_overfit.py`). Sin neuronas por etiqueta.
  - **Ajusta el encoder** (dos LR: encoder y cabeza, como Laya) y guarda
    checkpoint + `train_manifest.json`: sha de los episodios consumidos por
    familia/idioma, semilla, hiperparámetros, `format_fingerprint`, pasos.
  - Evaluación intermedia en presupuestos predeclarados (`--eval-every`): a
    cada corte corre `eval` sobre dev y escribe una fila en
    `<out>/curve.jsonl`. Nunca elige el checkpoint con dev y luego lo reporta
    como si fuera ciego: la elección queda escrita como «elegido por dev».
- `eval --checkpoint <dir|none> --battery data/battery_dev.jsonl` publica las
  **cuatro** cifras: forzada, con abstención (umbral fijo, declarado), macro
  por familia, éxito conjunto en pares contrafactuales; por idioma y por K;
  n, azar e IC95 % (Wilson) en cada una; `rows_sha256` de la batería. Con
  `none` reproduce la columna `nli-nograd` de preflight (**test: tiene que
  dar 0,6225 ± 0 en las mismas filas**; si no, el formato ha cambiado).
- `gate --run <dir>` compara el checkpoint contra `none` sobre las mismas
  filas (sha) y escribe `artifacts/gates/T-ce-finetune/gate.json` con la regla
  de abajo. Ninguna cifra a mano.
- Tests (`training/python/test_ce_finetune.py`): un episodio inválido se
  rechaza; el manifest de consumo cuadra con lo leído; `eval none` reproduce
  el listón; el gate con un dev peor en una familia da NO-GO aunque la media
  suba; con dos checkpoints iguales da «sin diferencia».

### B. El smoke de 20 minutos — antes de gastar más

Con lo que haya de episodios del piloto (`artifacts/episodes-qwen/pilot-2k/`,
≥ 1 394) y **un split provisional por `variant_group`** (80/20, semilla fija,
sólo para este smoke; nada de dev ni sellado dentro), un `train` de
≈ 1 000 decisiones en MPS (`--budget 1000`, tiene que caber en ≈ 20 min; mide
y escribe el tiempo). Predicción escrita ANTES: «dev forzada > 0,6225 con IC
que no cruza 0, y contrafactual conjunto > 0,343».

- **Si mejora:** sigue a C. Es la señal de que la técnica aprende.
- **Si no mejora:** NO subas volumen. Revisa, en este orden, y registra cada
  intento con su cifra: (1) el formato de hipótesis coincide en train y eval
  (`format_fingerprint`); (2) el gold está donde el trainer cree (imprime 5
  ejemplos renderizados); (3) LR del encoder (prueba 1e-5 y 5e-5); (4)
  truncado: ¿cabe el estado entero? (`length_report`); (5) mezcla: ¿hay
  familias con 0 filas? Si tras eso no mejora, gate NO-GO con causa candidata
  y para: el operador decide. No se «ajusta hasta que salga».

### C. La cifra oficial: 5 000, luego 20 000

Cuando `#T-episode-splits` esté `done`: `train` sobre su split de train con
`--budget 5000`, evaluación intermedia a 1 000 / 2 500 / 5 000, mismo
checkpoint sin ajustar como control. Regla de continuación (escrita aquí,
antes de medir):

- **GO a 20 000** si en dev la forzada y el contrafactual conjunto mejoran
  con IC95 % que no cruza 0 **y** ninguna familia ni idioma cae bajo su valor
  sin ajustar.
- Si mejora la media pero cae una familia o un idioma: **no se escala**; se
  registra como limitación y se revisa la mezcla de esa familia.
- Si no mejora: NO-GO, causa candidata, volver al checkpoint base.

La ampliación a 20 000 mezcla episodios (`#T-episode-scale`) y, si ya están,
typed-decisions train y públicos (`#data-flywheel`) con pesos declarados en el
manifest; la regla es la misma.

### D. Ejecución

Por el job canónico `trainer` (edita su `command`, un solo entreno a la vez).
Tests de `training/python` y `model` en verde. `## Resolution` con comandos
exactos, tiempos medidos y las cuatro cifras contra el control.

## Verification gate

- Test: la mezcla consumida por el entreno es auditable contra el manifest de
  `#T-episode-splits` (familias, idiomas, grupos de variantes).
- Test: la evaluación intermedia corre en los presupuestos predeclarados y
  escribe las cuatro cifras, no sólo la media.
- El gate compara siempre contra el mismo modelo sin ajustar, mismas filas
  (sha), protocolo de información equivalente.

## Done when

- El trainer existe con tests y `eval none` reproduce el listón de preflight.
- El smoke de ≈ 20 min está medido y escrito (tiempo, cifras, predicción
  previa, decisión).
- El ajuste a 5 000 está medido en desarrollo contra el checkpoint sin
  ajustar, con IC95 %, y la decisión sobre 20 000 está escrita con la cifra
  que la justifica.
- Ninguna familia ni idioma cae por debajo de su nivel sin ajustar sin quedar
  registrado como limitación; si es NO-GO, el gate lo dice con su causa.
