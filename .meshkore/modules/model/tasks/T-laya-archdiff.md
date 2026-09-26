---
id: T-laya-archdiff
title: Laya vs nuestro pointer head — las diferencias, y cuáles son medibles
status: next
priority: high
owner: unassigned
category: model
initiative: laya-teardown
created: 2026-09-27
updated: 2026-09-27
---

# Laya vs nuestro pointer head — las diferencias, y cuáles son medibles

Laya resuelve el mismo problema con otra forma. Esta task las escribe una al lado
de otra y separa las que son *estilo* de las que pueden explicar los 0,425 contra
0,0123 de BANKING77 a K=77.

Las cinco diferencias que ya están identificadas en la lectura del código
(`laya/common.py::build_sequence` y `DecisionModel`, copia en
`TMP/laya/`, commit del clon anotado en el informe):

1. **Las opciones viven dentro de la secuencia.** Formato
   `[CLS] <tipo> instrucciones [SEP] [MASK] opt0 [MASK] opt1 … [SEP] estado [SEP]`.
   El logit de una opción es el hidden state de su `[MASK]` pasado por un scorer
   escalar. No hay memoria de estado compartida ni cross-attention externa: la
   interacción estado↔opción la hace el propio backbone, en self-attention.
2. **El backbone se entrena.** Su fine-tune usa dos LR (encoder 2,5e-5, cabeza
   1e-4) y gradient checkpointing. Nosotros lo congelamos por defecto.
3. **El tipo de pregunta es una señal aprendida** — `type_emb` (3 vectores) se
   suma a TODOS los hidden states, y además va como texto en el prompt.
4. **Cabeza de abstención con features de calibración.** `act_head` lee el
   pooled + `[top1, top1-top2, entropía normalizada, K/255]`. Nuestro `unknown`
   es un logit sobre un resumen invariante a permutación.
5. **Presupuesto duro de tokens por opción:** 48 tokens por opción, 192 de cabeza.
   Laya declara que ESE es su techo en BANKING77, no la capacidad.

## Done when

- Existe un documento en `.meshkore/docs/` con la tabla de diferencias y, por
  cada una, si es medible en nuestro arnés y cuánto costaría medirla.
- Está elegida y justificada **una sola** diferencia para probar primero, con la
  predicción escrita ANTES de medir.
- Está comprobado en su código (no supuesto) si el presupuesto de 48/192 tokens
  explica su propio techo de BANKING77, y qué valor tendría en el nuestro.
- Queda registrado qué se puede adoptar bajo Apache-2.0 y con qué atribución.
