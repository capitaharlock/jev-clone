---
id: T-episode-verify
title: Verificador separado y revisión humana estratificada del gold
status: active
priority: high
owner: unassigned
category: data
initiative: episodic-data
depends_on:
  - T-episode-gen
created: 2026-09-25
updated: 2026-09-27
---

# Verificador separado y revisión humana estratificada del gold

Un segundo pase del mismo modelo reduce algunos errores pero **no es una verdad
independiente**. El gold semántico necesita tres filtros distintos, y el piloto
necesita ser conservador: los desacuerdos y las ambigüedades se **descartan**, no
se resuelven a ojo para no perder volumen.

1. **Regla, donde exista.** Comparaciones numéricas, restricciones y desempates
   se verifican con el mismo determinismo que las generó.
2. **Verificador separado.** Un modelo o prompt distinto del generador recibe el
   episodio sin ver el gold y elige entre los IDs. Desacuerdo → cuarentena.
3. **Muestra humana estratificada.** Por familia × idioma × tipo de variante, con
   tamaño de muestra escrito antes de revisar, y el error observado publicado con
   su intervalo — no «revisado, parece bien».

Lo que esta task tiene que dejar medido es el embudo, porque hoy nadie tiene la
cifra: **generado → aceptado → publicado → consumido por un entreno**. El caso de
`artifacts/labelspace-qwen/spaces.jsonl` es el aviso: tener 294 entradas en un
fichero no prueba que estén en la mezcla que un entreno consumió.

## Verification gate

- Test: un episodio con gold manipulado a mano cae en cuarentena por el
  verificador separado, no pasa.
- Test: el embudo se reconstruye desde los manifests — para cualquier run se
  puede decir cuántos episodios de cada familia entraron de verdad en el entreno.
- La tasa de acuerdo generador↔verificador y el error de la muestra humana se
  publican con n e intervalo.

## Done when

- Toda decisión semántica publicada ha pasado verificador separado; la
  cuarentena es un artefacto, no un borrado.
- La muestra humana está revisada con tamaños fijados de antemano y su error
  publicado por familia e idioma.
- El embudo generado/aceptado/publicado/consumido es consultable por run.

## Resolution parcial (2026-09-27) — queda `active`: falta la muestra humana

- Código: `data/episode_verify.py` + `data/test_episode_verify.py` (commit `41199ef`, 19 tests).
- Ejecutado sobre el piloto (`verify.log`, 2 564 s con Qwen local): **1 840 verificados, 252 en cuarentena** (118 fuga `KEEP:`, 71 abstención del verificador, 64 desacuerdo). Artefactos en `artifacts/episodes-qwen/pilot-2k/`: `verified.jsonl`, `quarantine.jsonl`, `funnel.json`, `human_sample.jsonl`, `verify_manifest.json`.
- Acuerdo generador↔verificador (gate `artifacts/gates/T-episode-verify/gate.json`): **0,9355** IC95 % [0,924 · 0,945] sobre 2 092; predicción escrita antes ≥ 0,90 → cumplida. Peor celda: prioridad/en 0,800 [0,741 · 0,849]; comparación de atributos/en 0,891.
- Embudo: planificado 2 100 → generado 2 100 → aceptado 2 092 → publicado 2 092 → verificado 1 840 → consumido 0 (los smokes de `#T-ce-finetune` aún no escriben `consumed_ids`).
- Gold manipulado a mano: 1 840/1 840 a cuarentena.
- **Pendiente del operador:** revisar `human_sample.jsonl` (100 filas, 5 por celda familia × idioma × variante, fijadas antes) y anotar el error. Hasta entonces `human_error: null` y la task no se cierra.
