# Veredicto pre-registrado — #T-bigk-optsets

> Escrito y commiteado **antes** de que exista el checkpoint del brazo de
> espacio completo, y antes de mirar ninguna cifra suya. El commit que trae
> este fichero precede al arranque del job; eso es la prueba de que la regla
> de decisión no se eligió después de ver el número.
>
> Fecha: 2026-09-24 · autor: agente `developer-copy` (A037)

## Qué se compara

Par emparejado, una sola variable — la cardinalidad del objetivo:

| | control | brazo |
|---|---|---|
| run | `leverstack-d512-prior-ettin-68m-s20260922` | `bigk-fullspace-d512-prior-ettin-68m-s20260922` |
| K de entreno | 3-8 muestreadas | **el espacio entero de cada fila** |
| corpus | `decision-mix-clean-1m`, `--fence-clean` | idéntico |
| seed / mix-seed | 20260922 / 20260922 | idénticos |
| objetivo | `cross_entropy` sobre [K+1] + prior 1.0 | idéntico |
| arquitectura | ettin-68m congelado, d512, 2 capas, 8 cabezas, B=64 | idéntica |
| presupuesto leído | etapas 62 500 / 125 000 / 250 000 | las mismas |

Mismo `mix-seed` y mismo orden round-robin ⇒ las etapas comparadas ven
**las mismas filas**; lo único que cambia es cuántas opciones se le
entregan a cada una.

## La métrica que decide

**Primaria:** accuracy a cardinalidad completa sobre BANKING77 (77
etiquetas, azar 0,012987) en el **corte de desarrollo congelado**
(`eval.cuts` DEV, n=1 000). `beats_chance` = límite inferior Wilson 95 % >
azar (`eval/gate_rules.py`, regla R2). El corte reservado (n=3 080) se lee
**una sola vez**, sólo si la primaria de desarrollo pasa, y con su entrada
en `test-queries.json` (regla R7).

Diagnósticos, publicados y nunca en el titular: `eval.unseen` y la stage
eval del trainer, ambas en el régimen K≤8 de fase 1 — son las que hacen
comparable este run con los brazos anteriores (divergencia entreno/eval
declarada en `cardinality_regime`, regla R1).

## La regla, escrita antes

1. **GO** si el límite inferior Wilson 95 % de la primaria de desarrollo del
   brazo supera 0,012987 **y** su estimación puntual supera la del control
   medido en el mismo corte y presupuesto. Entonces —y sólo entonces— se lee
   el corte reservado una vez y se publica junto a la cifra vieja (0,0123 con
   azar 0,0130).
2. **NO-GO** si el intervalo de la primaria contiene el azar. Lectura
   obligatoria: **la cardinalidad del objetivo queda descartada como causa**
   del fracaso de transferencia. El siguiente sospechoso, por orden, es la
   **representación**: primero `#T-encoder-finetune` (el backbone congelado
   no recibía presión con K≤8; con el espacio entero sí, así que la pregunta
   se rehace), y si eso tampoco mueve nada, la **capacidad** del backbone
   (68 M → 149 M+). La vía muestreada (`#T-fullspace-objective`, negativos
   in-batch + log-Q) deja de ser la continuación natural de esta hipótesis y
   pasa a ser lo que hay que hacer sólo para los espacios no enumerables.
3. **R9, sin excepción:** un NO-GO a este presupuesto **limita gasto y no
   establece causa**. No se escribirá «la cardinalidad no es la causa» como
   hecho establecido: se escribirá «descartada como causa *a 250 k filas,
   una seed, backbone congelado*», y cualquier afirmación causal exige
   repetir el brazo ganador en otra seed y a mayor presupuesto.
4. **Trampa de abstención:** si la tasa de abstención a K=|espacio| se va a
   ~1,0, la primaria no es interpretable (una cabeza que se abstiene siempre
   publica accuracy 0 junto a un azar de 0,013) y eso se dice en el titular
   en vez de publicar el 0 como resultado. La abstención se publica en los
   dos regímenes, K pequeño y K=|espacio|, en este mismo artefacto.
5. **Diagnóstico que se hunde:** si la primaria sigue en azar **y** además
   `eval.unseen` (K≤8) del brazo cae por debajo del control, la lectura es
   que el objetivo nuevo no compró nada y costó el diagnóstico viejo — sigue
   siendo NO-GO, y así se escribe.

## Riesgo declarado antes de medir

El coste ya está medido (`cost.json`): a B=64 en MPS el paso completo cae de
421 filas/s a 245 filas/s entre K=8 y K=77, y la rebanada de la cabeza —la
atención entre opciones, ≈K²— se multiplica por 3,43. La caché de textos de
etiqueta no amortiza nada de eso. Es asumible a 250 k filas y no lo es a
escala sin más evidencia, que es exactamente por lo que R5 manda demostrar
la pendiente en pequeño antes de escalar.
