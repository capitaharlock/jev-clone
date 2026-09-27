---
title: Guía — un solo objetivo: un modelo que aprende y mejora cada día
updated: 2026-09-27
owner: architect-master
status: stable
---

# Un solo objetivo: un modelo que aprende y mejora cada día

> Léelo entero antes de tocar nada. Son diez minutos y evita el 90 % de los
> errores que han costado semanas en este repo. Si algo de aquí contradice a
> una task, manda la task (es más nueva o más concreta); si contradice al
> operador, manda el operador.

## 1. El objetivo, y lo que NO es el objetivo

**Objetivo (decisión del operador, 2026-09-27):** un modelo de decisión que
**aprende de verdad** y que **mejora día a día** con más datos y mejor entreno,
hasta competir con TypeSafe Jev. Tres palancas, las tres a la vez:
tecnología (arquitectura y objetivo de entreno), aprendizaje (el bucle que
entrena, mide y decide) y **volumen de datos** (todo lo que enseñe a decidir:
Laya, datasets públicos, Qwen generando y verificando, Jev como profesor).

**No es el objetivo ahora** — está archivado en `.meshkore/roadmap/initiatives/log/`
y nadie lo despacha: release pública, deploy, servidor, Candle/ONNX, latencia,
destilación a estado compartido, entreno en tres máquinas. Se reactivan
sólo si el operador lo pide, y ninguna task viva depende de ellas.

**Las cifras a batir** (todas públicas, todas reproducibles en nuestro arnés):

| Referencia | Corte | Cifra | Dónde se mide en nuestro repo |
|---|---|---|---|
| Jev 1.13.0 | typed-decisions test (400 estados, 2 000 decisiones) | 0,727 | `#T-ingest-laya` lo convierte; `#T-loop-scoreboard` lo publica |
| Laya fine-tuned | typed-decisions test | 0,766 | ídem |
| Jev | BANKING77 K=77 | 0,870 (publicado) | `eval/fullspace.py` (nuestro: 0,0123 = azar) |
| Laya | BANKING77 K=77 | 0,425 (techo por tokens) | ídem |
| Qwen local (profesor) | nuestra batería de desarrollo, forzada | 0,965 | `artifacts/gates/T-preflight-refs/` |
| nli-nograd (punto de partida, SIN ajustar) | batería de desarrollo, forzada | 0,6225 | ídem — **es el listón: todo entreno se compara contra esto** |
| pointer heads actuales (control) | batería de desarrollo | 0,33 (= azar 0,3375) | ídem |
| Meta del operador | batería sellada, macro por familia, preguntas respondibles | **≥ 0,70 con IC95 % inferior ≥ 0,70** | `#T-ce-confirm` |

## 2. Dónde estamos (2026-09-27, 19:30)

- **El entreno aprende.** Trainer `training/python/ce_finetune.py` y smoke de
  1 000 decisiones en ≈ 40 s en MPS: contrafactual +0,121 [0,043 · 0,200],
  forzada +0,055 [0,015 · 0,098] sobre el control (0,6225). **No aprende la
  comparación numérica**: la comparación de atributos queda en 0,574 incluso
  en su propia distribución (`#T-ce-finetune`, done).
- **Datos disponibles:** piloto de 2 092 episodios (1 840 verificados),
  typed-decisions train con 6 000 decisiones, y typed-decisions test con 2 000
  (solo evaluación).
- **Laya sin ajustar** empata con nuestro control (0,595) y falla en las mismas
  familias. En BANKING77 con K=77 da 0,379; su 0,425 publicado no se reproduce.
- **Profesor local:** `qwen3.8:27b-mlx` desde hoy, sin medir (`#T-qwen38-ref`).
- **Remoto** `origin` conectado, **sin push** hasta que el acierto lo justifique.

## 3. El orden de ejecución

**Decisión del operador (2026-09-27): se escala sin parar.** El orden y el
funcionamiento están en **`bucle-infinito.md` §8**. Resumen:

| Prioridad | Qué | Tasks |
|---|---|---|
| P0 | Profesor nuevo medido | `#T-qwen38-ref` |
| P0 | Volumen dirigido a lo que no aprendemos | `#T-numeric-gen` |
| P0 | Trainer de mezcla + marcador | `#T-loop-trainer`, `#T-loop-scoreboard` |
| P0 | Los dos procesos sin fin | `#T-episode-scale` (productor `data.stream`), `#T-loop-nightly` (bucle `training.python.loop`) |
| P1 | Escalera de brazos | `#T-backbone-ladder`, `#T-listwise-format`, `#T-loop-error-mining` |
| P1 | Más volumen y calidad de datos | `#T-ingest-public` (a medias), `#T-counterfactuals`, `#T-episode-splits`, `#T-dev-rotation` |
| P2 | Jev como vara y profesor | `#T-teacher-auth` (bloqueada: key), `#T-teacher-probe`, `#T-jev-soft-targets`, `#T-loop-rl-jev` |
| Hito | Confirmación con sellado | `#T-ce-confirm`, disparada por H3/H4 del bucle |
| Operador | Muestra humana del piloto | `#T-episode-verify` (active) |

## 4. El bucle — cómo funciona

Está en **`bucle-infinito.md`**: dos procesos sin fin, un productor de datos y
un bucle de entreno/prueba/promoción. Una escalera de volumen: el presupuesto
sube un escalón por cada promoción. Una regla de promoción escrita antes de
medir. Una escalera de brazos para cuando no se promueve: datos dirigidos, LR,
backbone mayor, formato listwise y, al final, alerta al operador. Hitos que
abren el sellado y protección contra agotar dev. **Cualquier agente que tome
el control lee ese documento entero antes de tocar nada.**

## 5. Reglas de oro (violarlas invalida el trabajo)

1. **Ninguna cifra se escribe a mano.** Todo número de un gate sale de un
   comando reproducible que está escrito en el propio artefacto. Si no se ha
   medido, el campo es `null` y `status: awaiting-compute`; no se firma.
2. **Todo entreno se compara contra el MISMO checkpoint sin ajustar**, sobre las
   mismas filas (comprobado por sha), con protocolo de información equivalente.
   «Mejora» significa: la diferencia en desarrollo tiene IC95 % que no cruza 0,
   y **ninguna familia ni idioma** cae por debajo de su valor sin ajustar.
3. **Si no mejora, no se escala.** Se revisan datos y objetivo y se vuelve al
   checkpoint base. Doblar el dato con un entreno que no aprende es gastar GPU.
4. **El test sellado no se abre** salvo en `#T-ce-confirm`, una vez, y queda
   consumido (`data/battery_sealed.py` lo impone). Dev decide; sellado confirma.
5. **Nada de dev ni de sellado entra en train.** Los splits van por familia,
   entidad, espacio y `variant_group` (`#T-episode-splits`); un dataset externo
   con corte de test publicado (typed-decisions, BANKING77) mantiene ese corte
   fuera del train, siempre.
6. **Jobs canónicos, no jobs nuevos:** `trainer`, `datagen`, `evalgate`,
   `dashboard`, `ollama` en `.meshkore/public/jobs.yaml`. Para otro experimento
   se EDITA el `command` del job; no se crea otro. Un solo entreno a la vez en
   esta máquina.
7. **Intérprete:** `.venv-train/bin/python` con `PYTHONPATH=.`. `pytest` sólo
   existe ahí. Tests por directorio: `data/`, `eval/`, `model/`,
   `training/python/`. Los dos rojos preexistentes y ajenos son
   `eval/test_release_gate.py::…t_unseen_labels_published` y
   `training/python/test_cloud_api.py::…docker…`; cualquier otro rojo es tuyo.
8. **Licencias y procedencia:** todo dataset entra por `source-register.md` con
   licencia y uso declarados; lo que no se puede declarar es `eval-only`.
   Laya es Apache-2.0: se mide, se aprende y se adapta citando; **no se copia
   código al repo sin decisión explícita del operador**.
9. **Presupuesto de profesor externo:** cada llamada a Jev va cacheada por hash
   y cada gate publica llamadas y coste. Sin tope declarado, no se llama.
10. **Registro:** cada task cierra con `## Resolution` (qué se midió, comandos,
    conteo de tests verbatim), entrada en `.meshkore/log/<fecha>.md`, sección en
    `coverage.md`, y commit con los tres trailers `Agent:`/`Model:`/`MeshKore:`.
    Snapshot (§20) antes de editar un fichero existente cuando el daemon lo
    admita; si el daemon rechaza la llamada, se anota y se sigue.

## 6. Cómo ejecutar una task (checklist para cualquier agente)

1. Lee la task entera y la iniciativa a la que pertenece. Lee los `depends_on`:
   si uno no está `done`, para y dilo (no lo «asumas»).
2. Escribe la predicción ANTES de medir cuando la task lo pida (qué esperas y
   qué umbral decide GO/NO-GO). Va en el gate, con fecha.
3. Construye lo mínimo que la task pide, con test de unidad, sin GPU.
4. Corre la medición por el job canónico (edita el `command`, arranca, espera,
   lee el log). Nunca en primer plano de tu sesión si dura más de 10 min.
5. Escribe el gate en `artifacts/gates/<task>/gate.json`: n, K, azar, IC95 %,
   sha de las filas, versión del checkpoint, comando exacto.
6. Suite del directorio en verde (salvo los dos rojos ajenos). `ruff check`.
7. `## Resolution` en la task, `status: done`, coverage, diario, commit. Si no
   has terminado, la task queda `active` con lo que falta escrito, nunca `done`.
8. Línea final de tu informe: `— <task> · <qué era>`.

## 7. Comandos que vas a necesitar

```bash
# tests por directorio
PYTHONPATH=. .venv-train/bin/python -m pytest data -q
PYTHONPATH=. .venv-train/bin/python -m pytest eval -q
PYTHONPATH=. .venv-train/bin/python -m pytest model -q
PYTHONPATH=. .venv-train/bin/python -m pytest training/python -q

# piloto de episodios (retomable) y su gate
env EPISODE_GEN_BULK=1 PYTHONPATH=. .venv-train/bin/python -u -m data.episode_gen run \
  --n 2100 --seed 20260926 --prose qwen --teacher qwen --concurrency 2 --batch 10 \
  --out artifacts/episodes-qwen/pilot-2k --resume
PYTHONPATH=. .venv-train/bin/python -m data.episode_gen gate --dir artifacts/episodes-qwen/pilot-2k --job-id datagen

# referencias previas al entreno (ya medidas; sólo rehacer tabla)
PYTHONPATH=. .venv-train/bin/python -m eval.preflight_refs table

# mecánica (ya firmada; reproducible)
env PYTHONPATH=. TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1 \
  .venv-train/bin/python -m training.python.ce_overfit overfit --device mps --seed 20260926 \
  --epochs 60 --lr 2e-05 --decisions-per-batch 8 --eval-every 5 --weights minilmv2-l6-mnli-xnli

# batería de desarrollo / sellada (auditoría, sin abrir)
PYTHONPATH=. .venv-train/bin/python -m data.battery_dev --gate
PYTHONPATH=. .venv-train/bin/python -m data.battery_sealed --gate

# Ollama (profesor local)
OLLAMA_NUM_PARALLEL=4 ollama serve      # modelo: qwen3.6:27b-mlx
curl -s localhost:11434/api/tags
```

## 8. Qué hacer si…

- **…el piloto de datos está muerto** (no hay proceso `data.episode_gen run`,
  `episodes.jsonl` sin cambios en >30 min): comprueba ollama
  (`curl localhost:11434/api/tags`); relanza ollama y el job `datagen` (lleva
  `--resume`). No borres `episodes.jsonl`.
- **…el entreno corto no mejora sobre nli-nograd:** NO subas el volumen. Abre
  `#T-ce-finetune` § «si no mejora» y sigue la lista (formato de hipótesis,
  LR del encoder, longitud, mezcla). Registra cada intento con su cifra.
- **…un test ajeno está rojo:** si es uno de los dos preexistentes, sigue; si
  no, es tuyo o de tu rama, arréglalo o para.
- **…la task pide GPU y el operador ha embargado cómputo:** deja el gate en
  `awaiting-compute`, status `blocked` con `blocked_reason`, y dilo.
- **…una cifra parece demasiado buena:** revisa fuga (variantes del mismo grupo
  en train y dev, dataset externo con su test dentro del train, dev usado para
  elegir hiperparámetros y luego reportado). `data/leakage.py` y
  `data/firewall.py` existen para esto.
