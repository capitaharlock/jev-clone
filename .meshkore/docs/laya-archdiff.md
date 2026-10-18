---
title: Laya vs nuestro pointer head vs nuestro cross-encoder — las diferencias, y cuál se prueba primero
updated: 2026-09-27
owner: developer
status: stable
task: T-laya-archdiff
initiative: laya-teardown
---

# Laya vs pointer head vs cross-encoder — las diferencias, y cuál se prueba primero

> Sólo lectura y CPU. Ninguna cifra de aquí sale de un entreno. Lo medido tiene
> su comando en `artifacts/gates/T-laya-archdiff/token_budget.py`; lo que no se
> ha medido está dicho como no medido (§7).

**En seis líneas.** Laya y nosotros resolvemos la misma pregunta con tres formas
distintas: Laya mete estado, pregunta y las K opciones en **una** secuencia y lee
un `[MASK]` por opción; nuestro pointer head congela el backbone, hace *mean-pool*
del texto de cada opción y lo cruza con el estado en una cabeza aparte; nuestro
cross-encoder lee estado+pregunta+opción **juntos** pero en K pasadas. La
diferencia que elegimos probar primero es la primera (§4). Y el «techo por
presupuesto de tokens» con el que Laya explica su 0,425 en BANKING77 **no
aguanta la cuenta** hecha con su propio código: a K=77 sus 77 etiquetas siguen
siendo distinguibles en 73 casos y el techo de formato es 0,948, no 0,425 (§5).

## 1. Qué se ha leído

| Fuente | Qué es |
|---|---|
| `TMP/laya/` — clon de `github.com/NandhaKishorM/laya`, commit `4066d5d` (2026-09-25), `laya` 0.3.20, Apache-2.0 | `laya/common.py` (`build_sequence`, `DecisionModel`, `proper_reward`, temperaturas), `laya/agent.py` (predicción, `_encode_state`, `_decode_answers`), `laya/router.py`, `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`, `BENCHMARKS.md`, `README.md`, `research/scripts/bench_apps.py`, `research/results/app_benchmark_results.json` |
| Nuestro | `model/ce_scorer.py`, `model/decision_head.py`, `model/option_mixer.py` (superseded), `model/shared_state.py` (contrato de juguete), `training/python/ce_overfit.py`, `eval/calib.py`, `eval/fullspace.py`, `eval/cuts.py`, gates `T-fullspace-objective`, `T-teacher-probe/fullspace.json`, `T-preflight-refs` |

Los tres sistemas, en una línea cada uno:

- **Laya** (`common.py:94-146, 149-217`): `[CLS] <tipo> question: instrucciones [SEP] [MASK] opt0 [MASK] opt1 … [SEP] estado [SEP]` por un encoder bidireccional (ModernBERT-large 421 M en `laya`; mmBERT-base en `laya-multilingual`), + `type_emb` sumado a todos los hidden states, + 2 capas `TransformerEncoder` propias sobre TODA la secuencia, + scorer escalar sobre el hidden de cada `[MASK]`, + `act_head`. Una pasada por pregunta; el estado se re-lee por pregunta (`agent.py:661-677`).
- **Pointer head** (`decision_head.py:125-197, 271-313`): backbone (`ettin-68m`) **congelado**, estado codificado una vez; pregunta y opciones como *mean-pool* del backbone (`:315-323`); 2 bloques cross-attention (opciones → estado, opciones ↔ opciones sin posición) y logit = producto escalar entre contexto y la clave del **embedding pooled** de la opción; `unknown` como logit aprendido. K variable, sin matriz por etiqueta.
- **Cross-encoder** (`ce_scorer.py:264-297, 423-486, 519-566`): un checkpoint NLI (`minilmv2-l6-mnli-xnli`, 107 M) lee `premisa = Estado+Pregunta[+bloque de candidatos]`, `hipótesis = "The answer to this question is: {opción}."`; un logit (`entailment`) por par, softmax sobre los K pares. Todo el modelo se entrena (`ce_overfit.py`, LR 2e-5). Formato congelado por huella `b215e3003cc60c0f`.

## 2. Las cifras que se comparan (todas ya publicadas, ninguna nueva)

| Sistema | Corte | Cifra | Fuente |
|---|---|---|---|
| Laya `laya` (sin ajustar) | BANKING77 test, 400 filas, K=77, seed 13, CPU | **0,425** (macro-F1 0,112, ECE 0,540, confianza media 0,962, 234,8 ms/caso) | `research/results/app_benchmark_results.json` → `suites/jev.banking77_full/english` |
| Laya `laya-multilingual` | ídem | **0,425** (macro-F1 0,113, ECE 0,404) | ídem `/multilingual` |
| Laya `laya-typed-decisions` | ídem | 0,4925 | ídem `/typed-decisions` |
| Jev (publicado, no medido por Laya) | BANKING77, 100 filas, **72** etiquetas | 0,870 | `bench_apps.py:47-48` |
| Pointer head `leverstack-…-1M` | BANKING77 **test reservado**, 3 080 filas, K=77 | **0,012338** (38/3 080, IC95 [0,0090, 0,0169], azar 0,012987) | `artifacts/gates/T-teacher-probe/fullspace.json` (2026-09-24) |
| Pointer head `bigk-fullspace` (control) / `fullspace-sampled` (brazo) | BANKING77 dev, 1 000 filas, K=77 | 0,010 / 0,000 (abstiene 87,7 %) | `artifacts/gates/T-fullspace-objective/gate.json` → `primary` |
| Cross-encoder `nli-nograd` (sin ajustar) | batería de desarrollo, K 2-8, forzada | 0,6225 (azar 0,3375) | `artifacts/gates/T-preflight-refs/refs-nli-nograd.json` |
| Cross-encoder | BANKING77 K=77 | **no medido** — es la predicción de §4 | — |

## 3. La tabla de diferencias

Leyenda de la última columna: **estilo** = no puede mover 0,425 vs 0,0123;
**candidata** = puede explicarlo; **parcial** = contribuye pero no basta.
«Coste» es lo que costaría medir ESA diferencia en nuestro arnés, con lo que
hay hoy; horas = desarrollo, min/h GPU = cómputo (MPS o CPU, esta máquina).

| # | Diferencia | Laya | Pointer head (nuestro) | Cross-encoder (nuestro) | ¿Estilo o explica 0,425 vs 0,0123? | ¿Medible aquí? Coste |
|---|---|---|---|---|---|---|
| **D1** | **Dónde ocurre la interacción estado ↔ opción** | Dentro del encoder: cada opción entra como tokens **antes** del estado, con un `[MASK]` delante; el logit es el hidden de ese `[MASK]` tras el encoder + 2 capas propias (`common.py:126, 166-167, 185-198`). El encoder ve el texto literal de las 77 opciones y del estado a la vez. | Fuera del encoder: la opción es UN vector *mean-pool* de un backbone congelado (`decision_head.py:315-323`), cruzado con el estado en una cabeza de 10 M params. El detalle léxico de la opción se pierde en el pooling antes de cruzarse con nada. | Dentro del encoder, pero **una opción por pasada** (`ce_scorer.py:264-272`); la comparación entre opciones la hace el softmax, o el bloque de candidatos en la premisa cuando la familia lo pide (`:126-132`). | **Candidata (la más fuerte).** A 1 M filas el pointer sigue en azar a K=77 y `verdict.md` de `T-fullspace-objective` ya señaló «representación» como sospechoso. Laya y el cross-encoder comparten lo que el pointer no tiene: el encoder lee la opción entera junto al estado. | **Sí, sin entrenar:** el cross-encoder sin ajustar a K=77 (§4). ≈ 2 h de adaptador (`full_samples` → `CE.Decision`, como `token_budget.py`) + ≈ 20 min CPU el corte dev (63 pares/s medidos). Un brazo «Laya-like» (todas las opciones en una secuencia) no existe en el repo: 1-2 días + GPU; no se propone ahora. |
| **D2** | **El backbone se entrena** | Sí, entero: dos LR (encoder 2,5e-5, cabeza 1e-4, `train_ddp.py` en el cuaderno, cell 8, constantes `LR_ENCODER`/`LR_HEAD`), *gradient checkpointing* en encoder y cabeza (`common.py:189-192`), AdamW wd 0,01, coseno, 4 épocas, batch efectivo 64 en 2×T4, ≈ 4-6 min. | Congelado por defecto (`cost.json: frozen_backbone true`); existe la vía con gradiente (`decision_head.py:325-338`, `#T-unfreeze-backbone`) pero el brazo fullspace no la usó. | Sí, entero (`ce_overfit.py`, LR 2e-5 único). | **Candidata.** Un pooling congelado no puede aprender a distinguir 77 intenciones bancarias que el preentreno no separa. | Ya en curso: es `#T-ce-finetune` (smoke ≈ 20 min MPS). Para el pointer con backbone libre: `cost.json` da 3,6 min/62 500 filas a K=41 congelado; descongelado **no medido**. |
| D3 | El tipo de pregunta como señal | `type_emb` (3 vectores) sumado a TODOS los hidden states (`common.py:168, 185`) y además en texto `"%s question: %s"` (`:113`). | No hay tipo: sólo `choice` (`boolean` = choice de 2). | Igual: sólo `choice`; el tipo va implícito en la plantilla. | **Estilo** para nosotros: BANKING77 es todo `choice`, y nuestra batería también. Importa sólo si añadimos `score`/`noul`. | Trivial (una palabra en la plantilla, invalida la huella del formato). No lo propongo. |
| D4 | Abstención / confianza | `act_head` sobre `[CLS] + [top1, top1−top2, entropía/log K, K/255]` (`common.py:170, 200-216`). Laya declara que **no lleva señal útil** (`README.md:1044`: `act_probability` ≈ 1,0 siempre, AUROC 0,30) y recomienda `answer_confidence` = max p (AUROC 0,77). | `unknown`: logit aprendido sobre un resumen invariante a permutación (`decision_head.py:159-162, 191-197`). En el brazo fullspace abstiene el 87,7 % a K=77; forzado sigue en 0,009. | Sin abstención; `eval/calib.py:70` ya compara `unknown / maxprob / margin / energy`. | **Estilo** para la cifra: el 0,425 de Laya y nuestro 0,0123 son ambos **forzados** (argmax sobre opciones). Sí importa para `#T-battery-calib`. | Ya medible gratis con `eval/calib.py` (las cuatro estrategias). Nada nuevo que construir. |
| **D5** | **Presupuesto duro de tokens por opción** | 48 tokens/opción en el tokenizador (`common.py:119-125`) y `head_max_len` compartido (192 en `laya`, 256 en los otros dos; `agent.py:655-656`, `README.md:983-984`); si no cabe, cada opción queda en `max(4, (head_max_len−16)//K)` tokens **incluido** su `[MASK]` (`:127-131`); la instrucción se recorta a ≥ 8 (`:132`); el estado se lleva el resto (`:139-146`). | No hay presupuesto por opción: cada texto se tokeniza aparte hasta 512 (`decision_head.py:317-319`). | No hay presupuesto por opción: la hipótesis nunca se recorta (`only_first`, `ce_scorer.py:157, 343-345`) y `length_report` cuenta cualquier recorte del estado. | **Parcial, y menor de lo que Laya afirma.** Medido en §5: a K=77 quedan 3 tokens de texto por etiqueta, 73/77 siguen distinguibles, techo de formato 0,948. Lo que sí hace el presupuesto: 289 de 512 tokens son opciones, 30 etiquetas van recortadas, y la instrucción queda en 9 tokens. | Medido (tokenizadores, 25 s CPU). La ablación 192 → 512 sobre **su** checkpoint es de `#T-laya-baseline`: 400 filas ≈ 1,6 min CPU por ajuste a 235 ms/caso (su propio ritmo). |
| D6 | El objetivo de entreno | RLCD: recompensa por regla estrictamente propia (`proper_reward`, `common.py:278-304`: log + 0,5·esférica, + RPS para `score`) con G=4 muestras de ruido gaussiano de media cero sobre los logits, ventaja normalizada, σ 0,4→0,1; **más** CE suave con peso 1,0 contra probabilidades del profesor (cuaderno cell 8, bucle de entreno, l.157-173 de la celda). En la práctica el término CE suave hace casi todo el trabajo; el RL es una perturbación alrededor. | CE dura listwise con `unknown` (`decision_head.py:535-537`); brazo fullspace con negativos in-batch corregidos. | CE dura listwise sobre los K pares (`ce_overfit.py`). | **Parcial.** Puede mejorar calibración y aprovechar targets blandos (Jev/Qwen como profesor); no explica azar vs 0,425 por sí solo: nuestro pointer con CE exacto sobre el espacio entero también quedó en azar. | `#T-laya-objective` (backlog): reimplementar `proper_reward` (≈ 30 líneas, cita) + un entreno corto emparejado ≈ 2 × el smoke de `#T-ce-finetune`. |
| D7 | Temperatura por (tipo, K) | Cubos `tipo:{2, 3-5, 6-10, 11+}` (`common.py:367-369`), ajustados por LBFGS sobre un 10 % retirado antes de entrenar (cuaderno cell 8, `fit_one_temp` y `CALIB_MAX`), recortados a [0,5, 5] porque el cubo `choice:11+` publicado era 0,1006 y **afilaba** (`common.py:372-388`). Aplicada en `agent.py:768-774`. | Una temperatura global (`eval/calib.py:169-191`, `group` opcional). | Igual (mismo `calib.py`). | **Estilo** para el acierto (el argmax no cambia con T > 0). Importa para ECE y acc@cobertura. El dato relevante: su cubo `11+` afilaba ×10, y a K=77 reportaban confianza 0,96 con 0,425 de acierto (ECE 0,54). | Gratis: `fit_temperature(entries, group=…)` ya admite grupo. Va en `#T-laya-objective`. |
| D8 | Sensibilidad al orden de las opciones | Las opciones ocupan posiciones en la secuencia → el modelo **puede** depender del orden (Laya lo mide como «option-order robustness», `BENCHMARKS.md` T4 Colab). `bench_apps.py:111` sirve las 77 ordenadas alfabéticamente, siempre igual. | Invariante por construcción (sin posición en el eje de opciones, `decision_head.py:70-75`; test `check_order_invariance`). | Invariante por construcción (bloque de candidatos ordenado por texto, `ce_scorer.py:251-261`). | **Estilo** para la cifra (cada sistema se mide con su orden fijo), pero es una **ventaja nuestra** que Laya no tiene. | Sobre su checkpoint: permutar las 77 y repetir (`#T-laya-baseline`, minutos CPU). |
| D9 | Estado compartido entre preguntas | El estado se tokeniza una vez pero se **re-lee** por pregunta: una secuencia completa por pregunta (`agent.py:661-677`); coste ∝ preguntas × 512. | Estado codificado **una** vez por fila, reutilizado por todas las preguntas (`decision_head.py:295-313`). | K pasadas por pregunta, cada una con el estado dentro. | **Estilo** (coste, no acierto). Nada aquí promete latencia (guía §1). | No aplica. |
| D10 | Tamaño y naturaleza del backbone | ModernBERT-large 421 M (512 ctx) / mmBERT-base 322 M (1 024-8 192). Preentreno MLM; `[MASK]` es un token con significado en el preentreno. | `ettin-68m` congelado (`cost.json`). | `minilmv2-l6-mnli-xnli` 107 M con cabeza NLI (`nograd.json: params 106 995 075`); alternativa `modernbert-zeroshot-v2` — **contaminada con BANKING77** (`ce_overfit.py:83-84`), no vale para medir esta pregunta. | **Parcial.** 68 M congelados vs 421 M entrenados es una brecha de capacidad, pero Laya con 421 M sigue en 0,425 a K=77; la capacidad sola no lo explica. | Cambiar de checkpoint en el cross-encoder es un flag (`--weights`); coste = el del entreno que se compare. Sin propuesta ahora. |
| D11 | Con qué datos aprendió | Base `laya`: mezcla sintética + públicos (AG News, BoolQ «in training mix», `README.md:945-946`); BANKING77 declarado `in_training: false` (`bench_apps.py:120`); fine-tune typed-decisions con 6 000 decisiones (K pequeño: 2-6 opciones) en 4 épocas. Nada indica que haya visto K≫20. | 1 M filas de `decision-mix` con K ≤ 8 en el sampler y espacio entero en el brazo fullspace; sigue en azar. | 48 casos sobreajustados (mecánica); el entreno real espera al piloto. | **Candidata en conjunto con D1/D2:** «no aprende» a 1 M filas con backbone congelado y pooling vs 0,425 con 6 000 decisiones y backbone libre. Y a la vez explica el techo de Laya mejor que el presupuesto: un modelo que sólo vio K ≤ ~20 no ha aprendido a elegir entre 77 marcadores. | Es exactamente lo que `#T-ce-finetune` + `#T-laya-baseline` separan: mismas filas, un sistema sin ajustar vs otro entrenado. |

Dos notas sobre la tabla:

- Las cinco diferencias de la task son D1-D5. D6-D11 salen de leer `agent.py`, el cuaderno y `bench_apps.py`.
- Nuestro `option_mixer.py` y `shared_state.py` son artefactos de juguete (proyecciones aleatorias, stdlib) ya *superseded* por `decision_head.py`; no entran en la comparación más que como historia de por qué el pointer tiene set-attention sin posición.

## 4. La diferencia que se prueba primero: D1, con la predicción escrita

**Elegida: D1 — la opción entra entera en el encoder junto al estado.** Es la
única diferencia que (a) separa a la vez Laya y el cross-encoder del pointer,
(b) se puede medir **sin entrenar nada** con lo que ya está en el repo, y (c)
contesta la pregunta 1 de la iniciativa («¿arquitectura o entreno?») en la
dirección más barata: si un cross-encoder **sin ajustar** ya supera claramente
a un pointer **entrenado con 1 M de filas** en el mismo corte y la misma K, el
problema del pointer es de representación, y no hay vuelta atrás a él.

**Medición.** Cross-encoder `CrossEncoderScorer(NliPairScorer("minilmv2-l6-mnli-xnli"))`,
modo `entail_logit`, formato `b215e3003cc60c0f`, familia `description_classification`
(sin bloque comparativo), sobre el **corte de desarrollo** de BANKING77
(`eval/cuts.py::DEV`: split train, 1 000 filas, seed 20260924, sellado por sha;
el test reservado **no se abre**), K=77 (todas las etiquetas, orden alfabético
fijo como en `full_samples`), acierto **forzado** (argmax sobre las 77, sin
`unknown`), IC95 % de Wilson. Mismas filas que el control pointer del gate
`T-fullspace-objective` (`primary.control`: 0,010, IC [0,0054, 0,0183]),
comprobado por hash del corte. `modernbert-zeroshot-v2` queda **excluido**:
lleva BANKING77 en su mezcla.

**Predicción (escrita 2026-09-27, antes de medir):** acierto entre **0,15 y 0,45**.
No es una cifra medida ni un cálculo: es el rango que apuesto por un NLI
multilingüe pequeño en zero-shot sobre 77 intenciones bancarias con nombres
cortos; lo escribo para que el resultado se lea contra algo.

**Regla GO / NO-GO (fijada antes de medir):**

- **GO** si el límite inferior del IC95 % ≥ **0,05** (≈ 4× el azar 0,013 y ≈ 3× el
  límite superior del pointer, 0,018). Lectura: la interacción léxica dentro del
  encoder es una condición necesaria; el pointer con pooling congelado queda
  descartado como representación y `#cross-encoder-pilot` sigue siendo la única vía.
- **NO-GO** si el límite inferior < 0,05. Lectura: la interacción sola no basta
  sin entreno (D2/D6/D11); nada se decide hasta `#T-ce-finetune`, y la
  reproducción de Laya (`#T-laya-baseline`) pasa a ser la referencia que manda.
- Sea cual sea el resultado, **no cambia el plan de entreno** (guía §3): esto
  ordena hipótesis, no compra GPU.

**Coste, con ritmo medido:** 77 000 pares; 63,0 pares/s en CPU sobre 308 pares
reales (`token_budget.json → throughput_cpu`) → ≈ 20 min CPU, más ≈ 2 h de
adaptador (`eval/fullspace.py::full_samples` → `CE.Decision`, el mismo camino que
usa `token_budget.py`). Se ejecuta **dentro de `#T-laya-baseline`**, cuya Done when
ya exige «la comparación contra nuestro checkpoint sobre las MISMAS filas,
comprobado por hash»; no hace falta una task nueva. Fecha: cuando se despache
`#T-laya-baseline` (siguiente en la fila P1 de la guía §3).

**Predicción secundaria, gratis dentro del mismo run (D5, sobre su checkpoint):**
`laya` a `head_max_len` 192 vs 512 sobre las mismas 400 filas de su benchmark.
Apuesto a que a 512 sube **menos de +0,15** (es decir, queda por debajo de 0,58):
si el presupuesto fuera la causa, al quitarlo debería acercarse al techo de
formato (1,0 a 512, §5); si es D11 (nunca vio K≫20), subirá poco. No es la
diferencia elegida; es una comprobación de la afirmación de Laya que cuesta
1,6 min de CPU por ajuste.

## 5. El presupuesto 48/192, comprobado en su código

### 5.1 Lo que hace el código (citas)

- `common.py:119-125` — cada opción se tokeniza con `truncation=True, max_length=48`: el 48 es un tope **por opción**, en el tokenizador.
- `common.py:126` — se antepone `[MASK]`: la opción ocupa `1 + len` tokens.
- `common.py:127-131` — `opt_budget = head_max_len − Σ(1+len)`. Si `opt_budget < 16`: `per = max(4, (head_max_len − 16) // K)` y **cada opción se recorta a `per` tokens incluido su `[MASK]`**. A K=77: `(192−16)//77 = 2 → max(4, 2) = 4` → **3 tokens de texto por etiqueta**. Con 256 (`laya-multilingual`, `laya-typed-decisions`): `(256−16)//77 = 3 → 4` → idéntico. El README (`:985`) escribe «`(256 − 16) // 77` ≈ 3-4 tokens por etiqueta» y olvida que uno de ellos es el `[MASK]`.
- `common.py:132` — la instrucción se recorta a `max(8, opt_budget)`: a K=77 quedan 9 tokens de `"choice question: Which banking intent does `message` express?"`.
- `common.py:139-146` — el estado se lleva `max_len − len(ids) − 1`: 211 tokens a 512/192, 723 a 1 024/256.
- `agent.py:655-656` — los valores por defecto son 512/192; `README.md:983-984` los de cada checkpoint; `README.md:986` propone subir a 512/1 024 en runtime; `agent.py:675-676` rechaza con error si un marcador se cae de la ventana.
- `bench_apps.py:110-120` — cómo se midió el 0,425: `mteb/banking77` test, primeras 400 filas (seed 13 no afecta al orden aquí), etiquetas `sorted(set(label_text))` con `_` → espacio y **sin descripción**, instrucción «Which banking intent does `message` express?», estado `{"message": texto}`.
- `BENCHMARKS.md:147-149` — la afirmación: «77 etiquetas reciben ~4 tokens cada una y dejan de ser distinguibles. Los dos checkpoints puntúan **exactamente 0,425**, lo que se espera de un techo de presupuesto y no de una brecha de capacidad».

### 5.2 Lo que sale al ejecutar SU `build_sequence` sobre las 77 etiquetas (medido)

Comando: `PYTHONPATH=.:TMP/laya .venv-train/bin/python artifacts/gates/T-laya-archdiff/token_budget.py`
(importa `build_sequence` y `render_options` del clon; tokenizador ModernBERT-base
verificado en `artifacts/weights/modernbert-base`, vocabulario 50 280 — el de
ModernBERT-large, encoder de `laya`, es el mismo `tokenizer.json`, pero eso **no se
ha verificado contra el checkpoint de Laya**, que no está descargado; mmBERT
**no medido**). Espacio de etiquetas leído del split de train (corte dev); el
test reservado no se ha abierto.

| | `laya` 512/192 | `laya-typed-decisions` 1 024/256 | Remedio del README 1 024/512 |
|---|---|---|---|
| Tokens por etiqueta sin recortar | min 2 · mediana 3 · máx 9 · Σ 265 (+77 `[MASK]` = 342) | ídem | ídem |
| ¿Se dispara la regla `< 16`? | sí (192 − 342 < 16) | sí | no (512 − 342 = 170) |
| Tokens por opción tras el recorte (incl. `[MASK]`) | **4** → 2-3 de texto | **4** | sin recorte (hasta 10) |
| Etiquetas intactas | **47 / 77** | 47 / 77 | 77 / 77 |
| Etiquetas distinguibles tras el recorte | **73 / 77** | 73 / 77 | 77 / 77 |
| Grupos de colisión (etiquetas idénticas tras recortar) | **3** (7 etiquetas): `balance not updated` ×2, `lost or stolen` ×2, `top up by` ×3 | 3 | 0 |
| **Techo de acierto que impone el formato** (uniforme / frecuencia de train) | **0,948 / 0,952** | 0,948 / 0,952 | 1,0 / 1,0 |
| Tokens de instrucción que sobreviven | 9 | 9 | 13 |
| Hueco para el estado | 211 | 723 | 666 |

**Conclusión sobre el punto 3 de la task.** El presupuesto **sí** recorta (30 de
77 etiquetas pierden tokens y las opciones ocupan 289 de los 512), pero **no
explica el techo de 0,425**: el techo que impone el formato es 0,95. Tres
observaciones más en la misma dirección, todas de su propio json:

1. macro-F1 0,112 con acierto 0,425 → las predicciones se concentran en pocas
   etiquetas; una colisión de formato daría macro-F1 cercano al acierto.
2. `laya` y `laya-multilingual` usan tokenizadores distintos (50 k vs 256 k de
   vocabulario): la estructura de colisiones no puede ser la misma, así que
   «exactamente 0,425» en ambos no es la firma de un techo de formato. A n=400 es
   170/400; puede ser coincidencia o un efecto del arnés. `#T-laya-baseline` lo
   resuelve reproduciéndolo aquí. **No medido.**
3. Confianza media 0,96 con 0,425 de acierto (ECE 0,54) y un cubo de temperatura
   `choice:11+` que afilaba ×10 (`common.py:372-376`): el checkpoint no ha
   aprendido a ser incierto a K grande, que es lo que se espera de D11 (nunca
   vio K≫20), no de D5.

Lo más probable es que el 0,425 sea **D11 + D1 en su forma extrema**: 77
marcadores en una secuencia que el modelo nunca vio con más de ~20, con la
instrucción reducida a 9 tokens. La prueba directa (192 → 512 en su checkpoint)
está en §4 como predicción secundaria.

### 5.3 Qué valor tendría ese presupuesto en nuestro scorer (medido con `length_report`)

En el cross-encoder no hay presupuesto por opción: la opción va entera en su
propio par y `TRUNCATION_STRATEGY = only_first` prohíbe tocarla. Con las 77
etiquetas de BANKING77 y 200 filas del corte dev:

| | `minilmv2-l6-mnli-xnli` | `modernbert-zeroshot-v2` |
|---|---|---|
| Tokens de la opción | 2-9 (mediana 4) | 2-9 (mediana 3) |
| Tokens de la hipótesis completa | 10-17 | 10-17 |
| Par más largo sobre 200 filas × 77 | **62 / 512** | 60 / 512 |
| Pares recortados | **0** | 0 |
| Con bloque comparativo (las 77 en la premisa) | 437 / 512, 0 recortes | 463 / 512, 0 recortes |

Es decir: el equivalente de «3 tokens por etiqueta» en nuestro scorer es «la
etiqueta entera, siempre», y ni siquiera el peor caso (todas las opciones en la
premisa) llega a la ventana. El precio es el otro lado de la misma moneda: Laya
lee **una** secuencia de 512 (289 de opciones, 211 de estado) por pregunta;
nosotros leemos **77 pares** de ≈ 45 tokens → 237 160 pares para el test
(≈ 63 min CPU a los 63 pares/s medidos) y 77 000 para el corte dev (≈ 20 min).
Si algún día se construye un brazo «Laya-like» con las opciones en una sola
secuencia, nuestra ventana de 512 chocaría con la misma regla: 342 tokens de
opciones antes de escribir el primer token de estado.

## 6. Licencia — qué se puede adoptar y cómo

- **Laya es Apache-2.0** (`TMP/laya/LICENSE`, texto íntegro; no hay fichero
  `NOTICE` en el clon). Titular según README y cuaderno: Convai Innovations
  (`convaiinnovations/laya*` en HF, mismos términos para los pesos, cuaderno cell 16
  «license: apache-2.0»). Commit citado: `4066d5d`, `laya` 0.3.20.
- **Ideas, formatos y fórmulas** (el formato `[MASK]`-por-opción, la regla de
  presupuesto, `proper_reward`, los cubos de temperatura, las features de
  `act_head`) no están cubiertos por el copyright: se pueden **reimplementar**
  citando la fuente («inspirado en Laya, `laya/common.py::proper_reward`, commit
  `4066d5d`, Apache-2.0»). Es la vía por defecto de esta iniciativa: **medir y
  aprender**.
- **Copiar código** (p. ej. `build_sequence`, `DecisionModel`, `proper_reward`
  tal cual) lo permite Apache-2.0 con cuatro condiciones (§4 de la licencia):
  incluir una copia de la licencia, conservar los avisos de copyright, marcar los
  ficheros modificados, y propagar `NOTICE` si existiera (no existe). Cabecera
  propuesta si el operador lo decide:

  ```
  # Portions derived from Laya (https://github.com/NandhaKishorM/laya),
  # Copyright (c) Convai Innovations. Licensed under the Apache License 2.0;
  # see THIRD_PARTY_LICENSES/laya-APACHE-2.0. Modified by jev-clone (fecha, qué).
  ```

  Y **la regla**: no se copia código al repo **sin decisión explícita del
  operador** (guía §5.8, iniciativa «Restricciones»). Esta task no copia nada:
  `token_budget.py` importa del clon en `TMP/` (symlink ignorado, `.gitignore:92`)
  y sólo commitea el json y el script propio.
- **Pesos**: descargar y medir `convaiinnovations/laya*` está permitido; antes de
  publicar una cifra con ellos hay que darlos de alta en `source-register.md`
  con licencia y uso `eval-only`, y declarar por dataset si estaba en su mezcla
  (`README.md:945-946`: AG News y BoolQ «in training mix»; BANKING77 declarado
  fuera). Eso es de `#T-laya-baseline`.
- Nuestro repo **no tiene `LICENSE` en la raíz**: lo anoto; no lo resuelvo aquí.

## 7. Lo que NO se ha medido (y quién lo mide)

- Ninguna cifra de Laya en nuestro arnés (checkpoint no descargado) → `#T-laya-baseline`.
- La ablación 192 → 512 sobre su checkpoint, y el tokenizador de mmBERT → `#T-laya-baseline`.
- El cross-encoder sin ajustar a K=77 (la predicción de §4) → `#T-laya-baseline`, mismas filas.
- El coste total de la pasada K=77 del cross-encoder es una **extrapolación
  lineal** de 308 pares; el ritmo sí está medido.
- Todo lo que sea entreno (D2, D6, D10, D11) → `#T-ce-finetune`, `#T-laya-objective`.

## 8. Artefactos

- `artifacts/gates/T-laya-archdiff/token_budget.py` — el comando; 25 s en CPU.
- `artifacts/gates/T-laya-archdiff/token_budget.json` — la salida citada en §5.
