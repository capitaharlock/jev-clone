---
id: T-laya-baseline
title: Laya sobre nuestra batería — la cota superior externa, sin entrenar nada
status: next
priority: high
owner: unassigned
category: eval
initiative: laya-teardown
created: 2026-09-27
updated: 2026-09-27
---

# Laya sobre nuestra batería — la cota superior externa, sin entrenar nada

Todo lo que hemos medido hasta ahora se compara contra el azar o contra nosotros
mismos. Laya es un checkpoint público que responde exactamente nuestro formato de
pregunta; pasarlo por nuestra batería convierte «no aprende» en un número con
referencia.

Alcance: `pip install laya` en un entorno aparte, `convaiinnovations/laya` y
`laya-multilingual`, **sin ajustar**. No se importa código de Laya al repo.

Lo que hay que medir, en este orden:

1. **Los pares contrafactuales del reanálisis del operador** (verde/rojo; cuál
   cuesta menos vs cuál dura más). Es la prueba barata: si Laya también elige
   igual en los dos, el fallo es del planteamiento del problema, no nuestro.
2. **La batería privada de `#honest-eval`**, misma suite, mismo sellado, una sola
   apertura, con IC95 %.
3. **BANKING77 a cardinalidad completa** en NUESTRO arnés (`eval/fullspace.py`),
   para saber si su 0,425 reproduce fuera de su harness o no.

Advertencia que va en el gate: sus checkpoints se entrenaron con mezclas que
incluyen tareas públicas; spam y phishing (0,993) están declarados «in training».
Cualquier dataset que esté en su mezcla NO es transferencia limpia y así hay que
publicarlo.

## Done when

- Hay un gate en `artifacts/gates/T-laya-baseline/` con las tres medidas, su n,
  su K, su IC95 % y la versión exacta del paquete y del checkpoint.
- Está declarado, por dataset, si estaba o no en la mezcla publicada de Laya.
- La comparación contra nuestro checkpoint corre sobre las MISMAS filas, y eso
  está comprobado por hash, no afirmado.
- Está escrito el veredicto en una línea: dónde nos gana, dónde empata y dónde
  falla igual que nosotros.
