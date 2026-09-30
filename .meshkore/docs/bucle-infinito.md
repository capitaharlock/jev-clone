---
title: El bucle infinito — entrenar, probar, promover, escalar
updated: 2026-09-30
owner: architect-master
status: stable
---

# El bucle infinito — entrenar, probar, promover, escalar

> **Decisión del operador (2026-09-27):** el entreno funciona, se escala a la
> máxima potencia de esta máquina y no se para. Este documento es el manual de
> operación. Lo lee cualquier agente que tome el control. Si algo de aquí choca
> con una task, manda este documento, salvo que la task sea posterior a esta fecha.
> Contexto y reglas generales: `guia-un-solo-objetivo.md`.

## 0. Lo que sabemos al empezar (medido, no supuesto)

| Hecho | Cifra | Fuente |
|---|---|---|
| El control sin ajustar (MiniLMv2-L6 NLI) en dev | forzada 0,6225, contrafactual 0,343 | `artifacts/gates/T-ce-finetune/eval-none.json` |
| El ajuste aprende | contrafactual +0,121 [0,043 · 0,200], forzada +0,055 [0,015 · 0,098] con 1 000 decisiones | `smoke.json` |
| Coste del ajuste | ≈ 40 s por 1 000 decisiones en MPS con L6 | `artifacts/checkpoints/ce/smoke-1k.log` |
| Lo que NO aprende | comparación de atributos 0,574 → 0,574 **en el holdout del propio piloto**; prioridad sube en distribución (0,62 → 0,78) pero no transfiere a dev | T-ce-finetune `## Resolution parcial` |
| Profesor local | `qwen3.8:27b-mlx` (desde 2026-09-27; el piloto usó 3.6) | `data/episode_gen.py::MODEL` |
| Coste de Qwen | ≈ 13 s por episodio (lote 10, concurrencia 2): **≈ 2 700 episodios por 10 h** | T-episode-gen |
| Metas | Jev 0,727 y Laya 0,766 en typed-decisions test; ≥ 0,70 macro con IC inferior ≥ 0,70 en la batería | guía §1 |
| **Jev en NUESTRA batería (2026-09-28)** | **1,000** forzada (400/400) y **1,000** contrafactual (140/140); las cinco familias 80/80; $0,0071 | `artifacts/gates/T-teacher-probe/battery.json` |

**Consecuencia de diseño.** Qwen es el cuello de botella de volumen, y el
backbone pequeño es el techo de la comparación numérica. El bucle ataca las dos
cosas a la vez. Escala el volumen con fuentes baratas: episodios numéricos
generados por regla (instantáneos), typed-decisions (6 000) y datasets
públicos. Y sube por una **escalera de brazos** (datos dirigidos, backbone
mayor, formato listwise) cuando el volumen deja de mover una familia.

## 0.1 Conclusiones tras medir a Jev (2026-09-28) — léelas antes de planificar

1. **La distancia real es grande.** Nuestro mejor ajuste queda a **0,32** de Jev
   en forzada (0,6775 frente a 1,000) y a **0,54** en contrafactuales (0,464 frente
   a 1,000). Jev resuelve perfectas las dos familias que nosotros y Laya no
   aprendemos: comparación de atributos y prioridad.
2. **Lo que hay basta para subir, no para llegar.** Primero se exprime lo que
   tenemos: el bucle con MiniLM, datos por regla, typed-decisions y el productor.
   Pero el smoke indica que el límite de la comparación numérica es la
   **capacidad del modelo**, no el volumen ni las horas: la familia no se mueve ni
   en su propia distribución. No se espera cerrar la distancia con MiniLM y más
   datos. El salto lo tienen que dar la **escalera de backbones** y el **formato
   listwise** (§4). Por eso `#T-backbone-ladder` sube a P0 en cuanto el bucle
   lleve 3 ciclos seguidos sin mover atributos o prioridad, sin esperar a la
   racha completa.
3. **Datos y cómputo no son el cuello, hoy.** Los datos por regla son
   ilimitados y 1 000 decisiones cuestan unos 40 s en MPS con L6 (unos 4 min con
   un backbone base). Si un peldaño grande (≥ 400 M) deja de caber en tiempo o en
   memoria, eso sí se escala al operador como petición de cómputo, con la cifra
   medida.
4. **La batería actual se satura.** Si Jev saca 1,0, la batería no distingue
   nada por encima de unos 0,95: sirve para subir desde 0,68, no para comparar
   con Jev en la parte alta. `#T-dev-rotation` pasa a P0 y el dev nuevo tiene que
   ser **más difícil**: más opciones (K hasta 20), distractores cercanos, cadenas
   de 3 criterios, negaciones dobles, estados largos con ruido. Se valida
   midiendo a Jev encima: **si Jev vuelve a sacar 1,0, el dev no vale.** Cuesta
   céntimos (`eval/jev_battery.py`, con la key en `.meshkore/credentials/`).
5. **Jev es también la referencia de cada hito.** Cada fila del marcador
   publica la distancia a Jev en el dev vigente. H3 y H4 (§5) se leen junto a
   esa distancia, no sólo contra el 0,70.

## 0.2 Corrección de la auditoría (2026-09-30) — el punto 2 de §0.1 es prematuro

El «límite de capacidad» salió de un smoke de 625 pasos, 1 época, LR constante, con la
pérdida de entreno en 1,07 → 1,02: el modelo no ajustó ni su propio entreno. Una sonda
de sobreajuste interrumpida en la época 8 (`#T-capacity-probe`) ya lleva el fit de las
familias numéricas de 0,34 a **0,76** y subiendo; unseen 0,31 → 0,42. **Antes de subir
de backbone**, `#T-capacity-probe` (terminar la sonda + entreno largo con los 50 000 por
regla y warmup/decay). Si el fit llega a ~1 y unseen no sigue, el siguiente brazo es
`#T-listwise-format` (el formato pairwise no deja comparar opciones), y después
`#T-backbone-ladder`. Con tres familias ya en 0,76–0,99, llevar las dos numéricas a
~0,75 pone la macro por encima de 0,80.

## 1. Las piezas (dos procesos que no se paran)

```
 PROCESO A — productor de datos (job `datagen`, sin fin)
   cada tanda: Qwen 3.8 genera N episodios (semilla = nº de tanda)
               → verificador separado (cuarentena)
               → dedup contra todo lo anterior y contra dev/sellado
               → publica en artifacts/episodes-qwen/stream/batch-XXXX/
   en paralelo, instantáneo: generador numérico por regla (atributos, prioridad)
               → artifacts/episodes-rule/batch-XXXX/

 PROCESO B — el bucle (job `trainer`, sin fin): ciclo k
   1 MIX       manifest con TODO lo publicado hasta ahora + pesos del brazo
   2 TRAIN     continúa desde `current` con presupuesto B(k) decisiones
   3 EVAL      dev (4 cifras + familia + idioma) · holdout en distribución
               por familia · typed-decisions test · BANKING77 cada 5 ciclos
   4 PROMOTE   regla §3: current ← candidato, o se conserva
   5 SCORE     fila en artifacts/scoreboard/history.jsonl
   6 DECIDE    §4: sube B(k) si promovió; si no, siguiente peldaño de brazos
   7 MILESTONE §5: si se cruza una meta, confirmación con sellado
   → ciclo k+1 (sin esperar: si no hay datos nuevos, entrena con los que hay
     a más épocas hasta el tope de repetición §2)
```

Estado del bucle en `artifacts/loop/state.json` (ciclo, `current`, presupuesto,
brazo activo, racha sin promover, datos consumidos por fuente). Cada ciclo deja
`artifacts/loop/cycle-XXXX/` con `mix.json`, `train_manifest.json`,
`eval-dev.json`, `gate.json` y `decision.json`. **Todo es reanudable:** si el
proceso muere, se relanza el mismo comando y retoma en el ciclo y la etapa
donde estaba.

## 2. Presupuesto por ciclo — la escalera de volumen

| Escalón | Decisiones por ciclo B(k) | Tiempo aprox. L6 (MPS) | Tiempo aprox. base (×6) |
|---|---|---|---|
| 1 | 2 000 | 1,5 min | 9 min |
| 2 | 5 000 | 3,5 min | 20 min |
| 3 | 10 000 | 7 min | 40 min |
| 4 | 20 000 | 14 min | 1,5 h |
| 5 | 50 000 | 35 min | 3,5 h |
| 6 | 100 000 | 70 min | 7 h |
| 7+ | ×2 cada escalón | — | — |

- **Se sube un escalón cada vez que un ciclo promueve.** Si no promueve, se
  queda en el mismo escalón y se cambia de brazo (§4).
- **Tope de repetición:** ningún episodio se ve más de 4 veces en total. Si el
  escalón pide más de lo que hay, el ciclo usa lo disponible y lo escribe
  (`decision.json#starved`). La escalera no se inventa datos.
- Tiempos medidos a partir de los 40 s por 1 000 del smoke. El agente sustituye
  la tabla por los tiempos reales tras el primer ciclo de cada escalón.

## 3. Regla de promoción (escrita el 2026-09-27 en `#T-loop-nightly`, antes del primer ciclo)

El candidato del ciclo k sustituye a `current` si y sólo si, en dev, con
bootstrap pareado por fila (2 000 réplicas, semilla fija):

1. la forzada **y** el contrafactual conjunto mejoran, cada uno con IC95 % de
   la diferencia pareada que no cruza 0; **y**
2. **ninguna familia ni idioma cae de forma significativa**: diferencia pareada
   con IC95 % superior < 0 = caída = no promueve.

Esta regla es la de `#T-loop-nightly`, escrita antes del smoke. La regla del
smoke, «ninguna familia por debajo en punto», era la del piloto y se queda en
`#T-ce-finetune`. No se elige una regla mirando un resultado. (Por esta regla
el smoke-1k habría promovido: forzada [0,015 · 0,098] y contrafactual
[0,043 · 0,200] despejan 0, y ninguna caída de familia es significativa con
n = 80. Se anota como hecho, no como motivo de la regla.) Si alguien
quiere cambiar esta, lo decide el operador y la versión nueva se fecha.

`current` arranca en el **control sin ajustar**. El primer ciclo compara contra
él, y desde ahí siempre contra el `current` anterior. **Además**, cada fila del
marcador publica la diferencia contra el control original, para que la deriva
acumulada sea visible.

## 4. La escalera de brazos — qué cambiar cuando un ciclo no promueve

Racha sin promover `r` (se reinicia a 0 al promover):

| r | Brazo siguiente | Qué cambia | Task |
|---|---|---|---|
| 1 | **datos dirigidos** | la familia que más cae recibe ×2 de peso y lote extra del generador por regla | `#T-numeric-gen`, `#T-loop-error-mining` |
| 2 | **LR** | encoder 2e-5 → 5e-5 → 1e-5 (el siguiente que no se haya probado en este escalón) | `#T-loop-trainer` |
| 3 | **backbone siguiente** de la escalera (abajo); reinicia `current` al control de ese backbone y baja al escalón 1 | `#T-backbone-ladder` |
| 4 | **formato listwise** (todas las opciones en una secuencia, D1 de Laya) | `#T-listwise-format` |
| 5 | **alerta al operador**: diario + `decision.json#needs_operator`. El productor de datos **sigue**, y el bucle sigue en el último brazo que promovió, sin subir escalón | — |

**Escalera de backbones:** MiniLMv2-L6-mnli-xnli, 107 M (hoy). Después
mDeBERTa-v3-base-xnli-multilingual, 280 M. Después XLM-R-large-xnli o
ModernBERT-large-NLI, alrededor de 400 M. Cada peldaño entra con su **control
sin ajustar medido** en dev. Si ese control ya es peor que el `current` del
peldaño anterior, se anota y se sube igual: se entrena antes de descartarlo.
Pesos verificados por `model/weights.py` y fila en `source-register.md`.

## 5. Hitos — cuándo se para a confirmar (y el operador decide publicar)

| Hito | Condición en dev / corte externo | Acción |
|---|---|---|
| H1 | typed-decisions test ≥ 0,727 (Jev) con IC inferior ≥ 0,70 | aviso al operador; se sigue |
| H2 | typed-decisions test ≥ 0,766 (Laya ajustado) | aviso; se sigue |
| H3 | batería dev: macro por familia ≥ 0,70 **y** cada familia ≥ su azar + 0,2 | **`#T-ce-confirm`**: segunda semilla + **una** apertura del sellado. Resultado al operador, que decide si se publica. El bucle sigue con un dev rotado (`#T-dev-rotation`) |
| H4 | dev ≥ 0,85 macro | igual que H3, con un sellado nuevo |

**El sellado sólo se abre en un hito H3/H4**, nunca para elegir brazo.
**No se hace push** a `origin` hasta que el operador lo diga (repo conectado
el 2026-09-27, sin publicar).

## 6. Protección contra agotar dev

Elegir en cada ciclo con las mismas 400 filas termina sobreajustando a dev.
Contramedidas, en orden:

- `current` se promueve por dev, pero el marcador publica también el holdout
  en distribución y typed-decisions test. Si dev sube y typed-decisions baja
  durante 3 ciclos promovidos seguidos, se escribe alerta de sobreajuste a dev.
- Cada 10 ciclos promovidos, o en cada hito, `#T-dev-rotation` produce un dev
  nuevo (mismas cinco familias, ES/EN, K=2/3/8, gold por regla o verificado +
  muestra humana). El dev anterior pasa a **regresión**: se sigue midiendo, ya
  no decide.

## 7. Operación — comandos (los implementan las tasks; nombres fijados aquí)

```bash
# Proceso A — productor (job `datagen`, command fijo, sin fin)
PYTHONPATH=. .venv-train/bin/python -u -m data.stream run --forever \
   --qwen-batch 500 --rule-batch 5000 --out-qwen artifacts/episodes-qwen/stream \
   --out-rule artifacts/episodes-rule

# Proceso B — el bucle (job `trainer`, command fijo, sin fin)
env PYTHONPATH=. TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1 \
  .venv-train/bin/python -u -m training.python.loop run --forever --state artifacts/loop/state.json

# estado y marcador
PYTHONPATH=. .venv-train/bin/python -m training.python.loop status
tail -5 artifacts/scoreboard/history.jsonl
```

- Los dos procesos corren a la vez: Qwen en MLX con `nice -n 10` y el entreno
  en MPS. Si el entreno se ralentiza más de ×2 por la contención, el productor
  hace pausa durante la etapa TRAIN (el bucle escribe `artifacts/loop/TRAINING`
  y el productor espera mientras exista).
- Se lanzan con `nohup` y log a `artifacts/loop/*.log` si el daemon no acepta
  llamadas. `jobs.yaml` lleva siempre el command real.
- **Vigilancia** (cualquier agente, en cada turno): ¿viven los dos procesos?
  ¿`state.json` ha avanzado en la última hora? ¿hay `needs_operator`? Si un
  proceso ha muerto, se relanza con el mismo comando; los dos son reanudables.

## 8. Orden de construcción para el agente que toma el control

Todo P0 salvo lo marcado. Lo que no depende de otra cosa, en paralelo.

1. `#T-qwen38-ref`: Qwen 3.8 sobre dev (400 filas, 15 min). Confirma que el
   profesor nuevo no es peor que el 3.6 (0,965) antes de que genere datos.
2. `#T-numeric-gen`: generador por regla de atributos y prioridad con volumen
   ilimitado. Es el primer brazo dirigido.
3. `#T-loop-trainer`: mezcla multi-fuente con manifest + continuación desde
   `current` + tope de repetición.
4. `#T-loop-scoreboard`: marcador y dashboard.
5. `#T-episode-scale` → `data.stream`: el productor sin fin.
6. `#T-loop-nightly` → `training.python.loop`: el orquestador. Implementa §1–§5
   literalmente, con tests de la regla de promoción y de la escalera.
7. Arrancar los dos procesos. Primer ciclo: escalón 1, brazo base, mezcla =
   piloto verificado + typed-decisions train + primer lote por regla.
8. En paralelo, cuando haga falta por la escalera: `#T-backbone-ladder`,
   `#T-listwise-format`, `#T-dev-rotation`. P1 sin bloquear:
   `#T-ingest-public` (más volumen), `#T-counterfactuals`, `#T-episode-splits`.
