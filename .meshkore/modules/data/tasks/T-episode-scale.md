---
id: T-episode-scale
title: De 2 000 a 20 000 episodios con Qwen — cuota diaria, familias nuevas y dedup
status: next
priority: high
owner: unassigned
category: data
initiative: episodic-data
depends_on:
  - T-episode-splits
created: 2026-09-27
updated: 2026-09-27
---

# De 2 000 a 20 000 episodios con Qwen — cuota diaria, familias nuevas y dedup

## Contexto

El piloto de 2 100 (`#T-episode-gen`) cuesta ≈ 13 s por episodio con
`qwen3.6:27b-mlx` en lote de 10 y concurrencia 2 (≈ 6 h). `#T-ce-finetune`
pide 5 000 decisiones verificadas y, si mejora, 20 000. Esta task convierte el
piloto en **producción diaria**: N episodios cada noche, con semilla nueva,
sin duplicar lo anterior, y con las familias que el modelo peor lleve.

## Qué hacer, paso a paso

1. **Semilla por día** (`--seed YYYYMMDD`) y salida
   `artifacts/episodes-qwen/day-YYYYMMDD/`. `--resume` ya existe: un run
   muerto se relanza igual. Mide el ritmo real y escribe en la task cuántos
   episodios caben en 8 h de noche con esta máquina (cifra medida, no
   promesa).
2. **Dedup entre días:** hash normalizado de (estado, pregunta, candidatos)
   contra todos los `day-*` anteriores y contra dev/sellado; un duplicado se
   rechaza con motivo `duplicate-of:<id>`.
3. **Cuota por familia e idioma** configurable (`--mix familia/idioma=peso`),
   por defecto la del piloto; `#T-loop-error-mining` la moverá después.
4. **Familias nuevas** (sólo cuando las cinco estén estables en dev): las que
   typed-decisions muestra y nosotros no tenemos — triage con criterios
   explícitos, «¿necesita revisión humana?», severidad ordinal. Cada familia
   nueva entra con su regla de gold o su verificador, nunca con «Qwen dice».
5. **Embudo publicado por día:** generado → aceptado → verificado
   (`#T-episode-verify`) → con contrafactual (`#T-counterfactuals`) → en el
   split de train (`#T-episode-splits`) → consumido por un entreno
   (`#T-loop-trainer` lo cierra). Un JSON por día en
   `artifacts/episodes-qwen/day-YYYYMMDD/funnel.json`.
6. Gate `artifacts/gates/T-episode-scale/gate.json`: días corridos, episodios
   por día, ritmo, duplicados rechazados, total acumulado en train.

## Done when

- Tres noches consecutivas reales con su `funnel.json` y ≥ 20 000 episodios
  acumulados válidos en train, sin duplicados entre días (test).
- La cuota por familia/idioma se aplica y se ve en el manifest de cada día.
- El ritmo por episodio y el coste por noche están escritos con cifra medida.
