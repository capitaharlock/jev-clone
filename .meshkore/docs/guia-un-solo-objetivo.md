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

## 2. Dónde estamos (2026-09-27)

- **Mecánica validada:** el scorer compartido (`model/ce_scorer.py`) sobreajusta
  48 casos a 0,958 en 117 s en MPS (`#T-ce-mechanics`). La tubería aprende.
- **Puerta previa abierta:** Qwen resuelve la batería (0,965), el punto de
  partida es `nli-nograd` (0,6225), los pointer actuales son el control
  (`#T-preflight-refs`, GO).
- **Datos:** el piloto de 2 100 episodios con Qwen está EN VUELO (job `datagen`,
  `artifacts/episodes-qwen/pilot-2k/`, con `--resume`). Murió una vez a 1 394;
  se retomó el 27 a las 12:40.
- **Entreno real:** NO ha empezado. `#T-ce-finetune` espera los splits
  (`#T-episode-splits`), que esperan verificador y contrafactuales, que esperan
  el piloto. **El trainer se puede construir ya** con la fixture y el piloto
  parcial; sólo la cifra que cuenta espera a los splits.
- **Laya:** clon leído en `TMP/laya/`; ninguna de sus tres tasks ejecutada.
- **Jev como profesor:** cliente hecho (`eval/teacher.py`), credencial
  rechazada (401). Necesita key válida del operador (`#T-teacher-auth`).

## 3. El orden de ejecución — qué se despacha primero

Cuando haya que elegir, este es el orden. Dentro de cada fila, las tasks van
en el orden en que se listan; entre filas, todo lo que no dependa de otra cosa
va **en paralelo**.

| Prioridad | Iniciativa | Tasks (en orden) | Por qué ahora |
|---|---|---|---|
| P0 | `#episodic-data` | `T-episode-gen` (vigilar el piloto) → `T-episode-verify` + `T-counterfactuals` → `T-episode-splits` → `T-episode-scale` | Sin datos con splits no hay entreno medido. Bloquea todo. |
| P0 | `#cross-encoder-pilot` | `T-ce-finetune` (construir el trainer YA; correr el smoke de 20 min con el piloto parcial; la cifra oficial con los splits) → `T-ce-confirm` | Es el experimento que decide si la técnica aprende. |
| P1 | `#laya-teardown` | `T-laya-archdiff` → `T-laya-baseline` → `T-laya-objective` | Cota externa sin entrenar; enseña qué copiar. Sólo CPU/inferencia. |
| P1 | `#data-flywheel` | `T-ingest-laya` → `T-ingest-public` → (`T-jev-soft-targets` cuando haya key) | Volumen: de 2 000 episodios a cientos de miles de decisiones. Sólo CPU. |
| P2 | `#teacher-distill` | `T-teacher-auth` (necesita al operador) → `T-teacher-probe` | Jev como vara de medir y como profesor. |
| P2 | `#daily-learning-loop` | `T-loop-trainer` → `T-loop-scoreboard` → `T-loop-nightly` → (`T-loop-error-mining`, `T-loop-rl-jev`) | El sistema que hace que mañana sea mejor que hoy. Arranca cuando `T-ce-finetune` dé GO. |
| P3 | `#honest-eval` | `T-battery-calib` (tras `T-ce-finetune`) | Calibración y abstención, medidas aparte. |

## 4. El bucle diario — a dónde vamos

```
 cada noche (job canónico, un solo comando encadenado, #T-loop-nightly)
 ┌──────────────────────────────────────────────────────────────────────┐
 │ 1. datagen   Qwen genera N episodios nuevos (semilla = fecha)         │
 │ 2. verify    verificador separado + contrafactuales + cuarentena     │
 │ 3. splits    se AÑADE al train; dev y sellado NO se tocan            │
 │ 4. trainer   continúa desde artifacts/checkpoints/ce/current         │
 │              mezcla declarada: episodios + Laya + públicos (+ Jev)   │
 │ 5. evalgate  batería de desarrollo: forzada, con abstención,         │
 │              macro por familia, contrafactual conjunto, por idioma,  │
 │              IC95 %; typed-decisions test contra Jev/Laya            │
 │ 6. scoreboard fila del día en artifacts/scoreboard/history.jsonl     │
 │ 7. promote   sólo si dev mejora y ninguna familia/idioma cae:        │
 │              current → candidato; si no, se conserva el anterior    │
 └──────────────────────────────────────────────────────────────────────┘
 el test sellado se abre UNA vez por decisión final (#T-ce-confirm), nunca aquí
```

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
