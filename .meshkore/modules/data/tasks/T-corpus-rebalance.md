---
id: T-corpus-rebalance
title: Rebalanceo del corpus — civil-comments deja de ser el 74,6 %
status: done
priority: medium
owner: unassigned
category: data
initiative: data-training
depends_on:
  - T-optset-sampler
created: 2026-09-21
updated: 2026-09-21
resolved_by: A003
resolved_by_conv: roadmap-architect-uwgjq
commit_shas: ['b6a75f93d1d03dd32f524837c4026d0a63ebc271']
completed_at: 2026-09-21T16:18:49.541Z
---
# Rebalanceo del corpus — civil-comments deja de ser el 74,6 %

Hallazgo E: de 2 681 195 ejemplos totales, `civil-comments` (toxicidad
binaria) aporta 1 999 514 = **74,6 %** del corpus. No enseña a elegir entre
opciones dinámicas: es una tarea binaria de etiqueta fija. El corpus útil
para el producto es una fracción minoritaria de lo que se está entrenando.

La estrategia de `data-training` ya fijaba guardrails (≤15 % por dataset,
≤30 % por familia) y el propio plan decía que "nunca 89 % sin que el pipeline
lo marque como error". Esta task los hace cumplir de verdad y los aplica a la
mezcla que alimenta a `#T-train-real`.

Trabajo:

1. Bajar `civil-comments` al nivel de los demás por muestreo (el `sample`
   de `JOBS` ya existe: pasa a ser la regla, no la excepción) y recontar la
   mezcla resultante.
2. Guardrails como **error duro** en el loader: un dataset > 15 % o una
   familia > 30 % aborta la construcción de la mezcla con un mensaje que dice
   cuál y cuánto.
3. Manifest de mezcla versionado: composición exacta, seed, shas por shard, y
   diversity index por dominio/familia/tipo/K/idioma.
4. Reponderar hacia lo que enseña la propiedad del producto: filas con
   opciones dinámicas y K variable por encima de las binarias de etiqueta
   fija.

## Verification gate

- Test: una mezcla construida con civil-comments al 74 % falla con error
  explícito.
- Test: el manifest reproduce la mezcla bit a bit desde seed + shas.
- El gate escribe `artifacts/gates/T-corpus-rebalance/gate.json` con
  `pass: true` y la composición final por dataset y familia.

## Done when

- Ningún dataset supera el 15 % de la mezcla ni ninguna familia el 30 %.
- El manifest versionado reproduce la mezcla exactamente.
- La proporción de filas con opciones dinámicas está publicada y es mayoría.

## Resolution

✗ #T-mix-1m **parcial, no cerrada** · el commit `d197948` sí está en HEAD (el veredicto `no-commit` del wake es falso), pero el gate marca `pass: false` con 4 checks en rojo. No la marco `done`: sería mentir.

El hallazgo que importa: **el corpus de 1M no existe**. Con banking77/helpsteer2/pubmedqa vallados, el registro entero da **39 981 filas** — y seis de las siete fuentes están clavadas exactamente en el cap del 15 %, así que no lo arregla ninguna semilla ni ningún target. Faltan 960 000 filas y hace falta fuente nueva. Además no hay **ni una sola fila `score`** (§48 pide 15 %) porque la única fuente ordinal está vallada, y `hard` se queda en 7,6 % contra el suelo del 10 %.

🚀 A026 → #data-training #T-mix-1m · continuación, no reinicio
· ampliar el registro con fuentes independientes y medir el techo **real** (un techo honesto de 420k es resultado válido; un 1M inventado no), más supply ordinal y hard/OOD medido. Sin tocar el manifest que el job de entreno está leyendo.

⏳ `mix1m-curve` corriendo los 4 entrenos reales en MPS (~13 h) · cierra los otros dos checks (`top2_trained`, `synthetic_value` §128).

<details><summary>artifacts/gates/T-mix-1m/gate.json — 6 verdes, 4 rojos</summary>

| check | estado | medido |
|---|---|---|
| `benchmark_fence` | ✅ | 0 filas de banking77/helpsteer2/pubmedqa, 0 ids Jevals sobre 39 981 preguntas |
| `mix_built` / `reproducible` / `token_ledger` / `diversity_dashboard` | ✅ | — |
| `corpus_1m_rows` | ❌ | 39 981 filas vs 1 000 000 (§86). Techo con caps §§65-66: 39 970 |
| `guardrails` | ❌ | sólo falla `hard>=10%`: 3 045 filas = 7,6 % |
| `type_mix` | ❌ | choice 55,3 % ✅ · noul 44,7 % (objetivo 30) · **score 0 %** (objetivo 15) |
| `top2_trained` | ❌ | `state: running` — el job aún no ha medido Stage 0/1 |
| `synthetic_value` | ❌ | `verdict: null` — pendiente del mismo job |

Composición realizada: 23 988 human / 9 997 synthetic / 5 996 programmatic · 70,0 % filas de opción dinámica · 8 idiomas (en 75 %).
</details>

<details><summary>Por qué continuación y no bloqueo (matrix fail #1)</summary>

- El veredicto dice `no-commit`; HEAD desmiente: `d197948` está en el log con los 16 ficheros del informe.
- El fallo real es de **alcance**, no de ejecución: el agente hizo bien su trabajo y midió que la especificación §86 es inalcanzable con el registro actual. Eso es información, no una caída.
- Matrix fail #1 → un reintento permitido. Lo uso como continuación acotada a los dos checks que dependen de datos; los otros dos están en manos del job y llegarán solos.
- No despacho #T-mix-5m ni #T-data-eval: ambas cuelgan de `depends_on: [T-mix-1m]` → el daemon devolvería 409, y sería trabajo especulativo si 1M sale NO-GO.
</details>


— T-mix-1m · el corpus limpio de 1M no existe: el registro vallado tope en 39 981 filas, medido y escrito en el gate

11.5M tokens
