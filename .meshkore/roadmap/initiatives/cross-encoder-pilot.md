---
id: cross-encoder-pilot
title: Scorer semántico preentrenado — leer estado, pregunta y opción juntos
status: active
owner: architect-master
modules:
  - model
created: 2026-09-25
updated: 2026-09-25
---

# Scorer semántico preentrenado — leer estado, pregunta y opción juntos

Work-stream abierto por el reanálisis del operador del 2026-09-24
(`.meshkore/docs/plan-recuperacion-2026-09-24.md`). Sustituye a
`#full-space-training` y a `#generalization-fix`, que quedan archivadas.

**Lo que está medido.** Los checkpoints actuales fallan: `#T-fullspace-objective`
cierra en 0/1000 con 87,7 % de abstención y 9/1000 forzando elección, contra un
azar de 12,99/1000 a K=77; `#T-bigk-optsets` da 9/1000 en brazo y en control.
Ampliar el denominador, rellenar más taxonomías con las mismas plantillas y
descongelar capas no mueven la aguja.

**Lo que NO está demostrado** y por tanto no se hereda como hecho: que la
arquitectura sea irrecuperable, que el modelo puntúe por posición, y que el
encoder sea la causa aislada. La prueba de mecanismo del operador (§3) mide lo
contrario en dos checkpoints: al permutar opciones las probabilidades realineadas
se mueven <5 × 10⁻⁷, y al intercambiar textos conservando IDs las puntuaciones
**siguen al texto**. El fallo funcional concreto es otro: el modelo elige «green»
tanto si el estado dice verde como si dice rojo, y elige el mismo producto al
preguntar cuál cuesta menos y cuál dura más (69,709 % → 69,747 %). No es una
cabeza de clases fijas mal conectada: es un modelo insensible al cambio decisivo.

## La hipótesis, escrita antes de medir

Partir de representaciones ya entrenadas para **relacionar textos** y aprender la
decisión como interacción estado–pregunta–respuesta, en lugar de entrenar otra
cabeza aleatoria sobre un backbone congelado:

```text
z_i = scorer_compartido(ESTADO, PREGUNTA, RESPUESTA_i)
p   = softmax(z_1 … z_K)
L   = -log p_correcta
```

Un único scorer, una única salida escalar, ninguna neurona por etiqueta del
catálogo. La interacción entre los tokens del estado, la pregunta y la opción
ocurre dentro del encoder preentrenado. Es una hipótesis comprobable, **no una
promesa del 70 %**: el criterio de éxito es batir al mismo checkpoint sin ajustar
sobre la batería privada de `#honest-eval`, con las familias contrafactuales
intactas y sin destruir un idioma.

Punto de partida: un checkpoint NLI multilingüe barato
(`MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli`) medido primero **sin
entrenar** por entailment, y `ModernBERT-base-zeroshot-v2.0` como alternativa en
inglés — con la advertencia de que su mezcla publicada **incluye BANKING77**, así
que no vale como transferencia limpia a ese benchmark. Los checkpoints pointer
actuales se conservan como control, no se tiran.

## Coste: lo que este work-stream NO promete

El cross-encoder relee el estado por opción. **Los 20–40 ms y la caché de estado
compartida no se dan por cumplidos aquí** — es la referencia de calidad que
descubre si la tarea es aprendible. La recuperación de latencia va a
`#shared-state-distill`, y sólo después de una mejora confirmada.

## Done when

- Existe un scorer compartido que puntúa opciones arbitrarias leyendo estado +
  pregunta + texto de la opción, con su contrato de contexto comparativo escrito.
- El checkpoint preentrenado **sin ajustar** está medido sobre la batería de
  desarrollo antes de cualquier entreno, y la cifra está publicada.
- La mecánica está validada por sobreajuste (>95 % en 32–64 ejemplos inequívocos)
  antes de gastar presupuesto en aprendizaje.
- El ajuste con decisiones verificadas mejora sobre el mismo modelo sin ajustar
  en desarrollo, sin romper pares contrafactuales ni un idioma — o queda
  registrado el NO-GO con su causa.
- La decisión final se toma con desarrollo y se confirma con una segunda semilla
  y **una única apertura** del test sellado, publicando IC95 %, K e idiomas.
