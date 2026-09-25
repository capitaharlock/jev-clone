---
id: T-episode-verify
title: Verificador separado y revisión humana estratificada del gold
status: next
priority: high
owner: unassigned
category: data
initiative: episodic-data
depends_on:
  - T-episode-gen
created: 2026-09-25
updated: 2026-09-25
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
