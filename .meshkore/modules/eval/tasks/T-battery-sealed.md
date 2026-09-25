---
id: T-battery-sealed
title: Test sellado — 600 casos finales y el diagnóstico separado a K=20/77
status: next
priority: high
owner: unassigned
category: eval
initiative: honest-eval
depends_on:
  - T-battery-dev
created: 2026-09-25
updated: 2026-09-25
---

# Test sellado — 600 casos finales y el diagnóstico separado a K=20/77

La otra mitad del paso 1, y la que da valor a cualquier afirmación pública. 600
casos **sellados** con las mismas cinco familias, ES y EN, y K=2/3/8; más un
**conjunto diagnóstico separado** a K=20/77 que no entra en la meta del 70 % pero
sí informa de cómo se degrada el modelo al crecer el denominador.

Reglas del sellado:

- Se abre **una sola vez**, por `#T-ce-confirm`. Su sha se marca como consumido y
  el runner rechaza una segunda apertura.
- Después de usarse pasa a **evidencia histórica** y esta iniciativa prepara otro
  para la siguiente decisión final. No se reutiliza para elegir el próximo brazo.
- Ningún `variant_group` se comparte con desarrollo.
- Es un corte **privado**: no lo ha producido el profesor ni se le ha mostrado
  durante la generación. No se declara ausencia de contaminación del
  preentrenamiento, que no es verificable.

BANKING77 a 77 vías se conserva —como transferencia difícil y como diagnóstico—
pero deja de ser la definición de éxito. Y `ModernBERT-base-zeroshot-v2.0` no
puede compararse ahí como «transferencia limpia»: su mezcla publicada incluye
BANKING77.

## Verification gate

- Test: el runner rechaza una segunda apertura del test sellado.
- Test: intersección de `variant_group` con desarrollo = 0; intersección de
  entidades y espacios declarada y por debajo del umbral escrito.
- Test: el diagnóstico K=20/77 se reporta en su propio bloque y no se promedia con
  la meta del 70 %.

## Done when

- Los 600 casos existen sellados, con sha registrado y estado «no consumido».
- El conjunto diagnóstico K=20/77 existe y está separado del corte de meta.
- La regla de apertura única está implementada, con test, no sólo escrita.
- El protocolo para preparar el **siguiente** test sellado está documentado antes
  de abrir este.
