---
id: T-laya-archdiff
title: Laya vs nuestro pointer head — las diferencias, y cuáles son medibles
status: done
priority: high
owner: developer
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

## Resolution

**done · 2026-09-27 · developer (claude-fable-5-1)** — sólo lectura y CPU; nada entrenado.

- Entregable: `.meshkore/docs/laya-archdiff.md` (enlazado desde `docs/INDEX.md`). Once
  diferencias (las cinco de arriba + objetivo RLCD, temperatura por (tipo, K), orden de
  opciones, estado compartido, backbone/tamaño, datos de entreno), cada una con qué hace cada
  sistema, estilo vs candidata, y coste de medirla aquí.
- Elegida UNA (D1, opciones dentro de la secuencia) con predicción escrita antes de medir:
  `nli-nograd` sin ajustar sobre el corte dev de BANKING77 a K=77, GO si IC95 % inferior ≥ 0,05.
  Se mide en `#T-laya-baseline` sobre las mismas filas (hash), no aquí.
- Comprobado en su código (`TMP/laya/laya/common.py:119-132`, clon `4066d5d`, laya 0.3.20) y
  medido con su propio `build_sequence` + tokenizador ModernBERT: a K=77 cada etiqueta conserva
  3 tokens de texto; 73/77 siguen siendo distinguibles; techo de formato **0,948**, no 0,425.
  El presupuesto NO explica su techo por sí solo. En nuestro scorer no existe tal presupuesto
  (par más largo 62/512, 0 recortes; 63 pares/s en CPU).
- Licencia: Apache-2.0, sin NOTICE; regla y texto de atribución en el doc §6.
- Comando reproducible del artefacto: `PYTHONPATH=.:TMP/laya .venv-train/bin/python
  artifacts/gates/T-laya-archdiff/token_budget.py` → `token_budget.json` (25 s, CPU).
- Tests: sin código de producto tocado; `ruff check` limpio sobre el script. Snapshot del daemon
  rechazado (`401`), anotado en el diario.
- NO medido (dicho como tal): Laya en nuestro arnés, mmBERT, ablación 192/512, cualquier cifra
  de entreno.

