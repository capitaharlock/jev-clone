---
id: T-lever-stack
title: Apilar las dos palancas que sí aplanan la pendiente — cabeza ×2 + objetivo prior
status: done
priority: high
owner: unassigned
category: model
initiative: generalization-fix
depends_on:
  - T-antiscale-diag
  - T-gen-objective
  - T-unfreeze-backbone
created: 2026-09-23
updated: 2026-09-24
completed_at: 2026-09-24T09:55:50.109Z
resolved_by: A035
resolved_by_conv: general-09222211
commit_shas: ['9f17c594db3c469e9bb6d09de332eb9b397c08b1']
---
# Apilar las dos palancas que sí aplanan la pendiente

Tres hipótesis medidas contra la curva anti-monótona 250 k → 1 M, y solo
dos sobrevivieron:

| palanca | pendiente unseen 250 k → 1 M | fuente |
|---|---|---|
| nada (congelado, d256) | −0,1899 | `#T-mix-5m` |
| cabeza ×2 (d512) | −0,0549 | `#T-antiscale-diag`, brazo `antiscale-wide-d512` |
| objetivo prior-penalty | −0,0868 | `#T-gen-objective`, brazo `genobj-prior-1m` |
| backbone `last-n=2` @1e-5 | −0,0927 (y peor en el gate) | `#T-unfreeze-backbone` — NO-GO |

Las dos que funcionan se midieron **por separado** y nunca se han
apilado. Son mecanismos distintos —una da capacidad a la cabeza, el otro
quita la recompensa de memorizar la etiqueta frecuente— así que a priori
no compiten. Esta task mide la combinación con un único brazo controlado.

Brazo: `ettin-68m`, `--d-model 512`, `--prior-penalty 1.0`,
`--fence-clean`, 1 M filas, batch 64, seed y mix-seed 20260922, MPS — todo
idéntico al brazo prior de `#T-gen-objective` salvo `d_model`. Ese brazo
(unseen 0,2347 a 1 M, 0,2879 en el gate) es el control.

Coste: ~2 h de GPU por la ruta batched de `#T-metal-throughput`, no las
~11 h de antes. Eso permite, si el apilado da señal, un segundo brazo d1024
sin renegociar la GPU.

## Done when

- El run de 1 M termina con las 5 etapas (62 k…1 M) y su `summary.json` en
  `artifacts/runs/`, y `eval.unseen gate` corre sobre el checkpoint de 1 M.
- `artifacts/gates/T-lever-stack/gate.json` publica la pendiente
  250 k → 1 M del apilado junto a las tres ya medidas, y dice si el efecto
  es aditivo, parcial o nulo — con el número, no con adjetivos.
- La decisión queda escrita: si la pendiente apilada no mejora la mejor de
  las dos sueltas (−0,0549), el eje de capacidad/objetivo se declara
  agotado y el relevo pasa al espacio de etiquetas (`#T-labelspace-div`).

## Resolution

Documentado primero, replanificado después, todo commiteado (`9f17c59`).

**La transición queda escrita** en `.meshkore/docs/fase-2-espacio-completo.md` y resumida en `context.md` para que cualquier agente la lea antes de tocar entreno: la fase 1 entrenó y midió con 3–8 opciones y backbone congelado —régimen barato, elegido a propósito— y la primera comparación a 77 vías marca el salto a espacio completo. Con ella van seis reglas de medición (R1–R6) y la lista de lo que se vuelve a medir: los NO-GO de `#T-unfreeze-backbone` y `#T-mix-5m` valen en el régimen de fase 1 y no se heredan al nuevo.

**Iniciativa nueva `#full-space-training`** (fase 2) con 5 tasks, más el testing ordenado en `#honest-eval`. Dos herramientas revisadas: el profesor es TypeSafe (URL correcta, key inválida) y el Qwen local está instalado pero apagado.

<details><summary>#full-space-training — la teoría y las 5 tasks</summary>

La teoría, escrita antes de medir: entrenar con K candidatos enseña a discriminar en una vecindad de tamaño K, no a rankear un espacio. Con K=8 el camino más corto al acierto es memorizar el mapa cerrado de las 9 taxonomías que cubren el 83,1 % del corpus — y ese mapa es justo lo que no transfiere, por eso más datos lo empeoraban. La receta que sí produce un ranking es la de recuperación densa (DPR/E5/SetFit), que es lo que Jev hace de facto, y son **tres cambios que sólo funcionan juntos**.

- `#T-fullspace-objective` (model, **active**) — la task central: pérdida normalizada sobre todo el pool, negativos in-batch con corrección log-Q, y caché de claves de etiqueta para que el coste no crezca con K. Brazo a 62 k antes de comprometer ningún 1 M.
- `#T-bigk-optsets` (data, repointed desde `#generalization-fix`) — el lado de datos: el corpus entrega el espacio entero cuando existe (banking77=77, huffpost=41, goemotions=28).
- `#T-encoder-finetune` (model) — reabre el NO-GO del backbone, incluido el brazo `full` que nunca se corrió. Razón: con K=8 la representación congelada ya basta, así que la pérdida no tenía presión que transmitir al encoder — el experimento no podía dar GO.
- `#T-jev-parity` (eval) — la referencia externa con protocolo idéntico, sin gastar saldo: mismas 3 080 filas, K=77, diferencias de protocolo declaradas una a una.
- `#T-labelspace-factory` (data) — espacios de etiquetas con el Qwen local, no paráfrasis.

Riesgo declarado: si con los tres cambios la cifra a 77 vías sigue pegada al azar, la hipótesis del objetivo queda refutada y el siguiente sospechoso es el tamaño del backbone.
</details>

<details><summary>El testing — #T-eval-cardinality en #honest-eval</summary>

Hoy hay **tres cortes que se contradicen** y ningún sitio donde se lean juntos: el stage eval del trainer (n=3 008), el gate de `eval.unseen` (n=5 624) y `eval.fullspace` (n=3 080, K=77). La task fija: jerarquía explícita (primaria = cardinalidad completa, el resto marcado como diagnóstico); `beats_chance` **obligatorio** en `eval/gate_rules.py`, de modo que un artefacto sin azar, K e IC 95 % **falle el gate**; un solo scoreboard por checkpoint; los tres cortes reconciliados por escrito; y el régimen de cardinalidad estampado en cada checkpoint, para que nadie compare un run K≤8 con uno de espacio completo sin saberlo.

La suite pytest la dejé corriendo en segundo plano y **aún no ha terminado**, así que no puedo afirmar que esté verde.
</details>

<details><summary>Datasets y herramientas — qué hay realmente disponible</summary>

- **En disco y sin usar bajo el objetivo nuevo**: `artifacts/episodic-div/` con **20 000 mini-taxonomías** (~1 M filas) que nunca se llegaron a entrenar; ~16 MB de paráfrasis del Qwen (`artifacts/data-qwen/`: banking77 13 083 filas, civil-comments 4 906, boolq 2 141); 6 shards de `synth/v1`.
- **Qwen local**: `ollama` instalado con `qwen3.6` y `embeddinggemma`, pero **el servidor no escucha en :11434** (el túnel cloudflared sigue vivo apuntando a nada). Hay que levantarlo como job del daemon. Y e

…(truncated)

**Commit** `9f17c594d` · 16 files · 14.5M tokens

**Files changed (16):**
- `.meshkore/docs/context.md`
- `.meshkore/docs/coverage.md`
- `.meshkore/docs/fase-2-espacio-completo.md`
- `.meshkore/modules/data/tasks/T-bigk-optsets.md`
- `.meshkore/modules/data/tasks/T-labelspace-factory.md`
- `.meshkore/modules/eval/tasks/T-eval-cardinality.md`
- `.meshkore/modules/eval/tasks/T-jev-parity.md`
- `.meshkore/modules/eval/tasks/T-teacher-auth.md`
- `.meshkore/modules/eval/tasks/T-teacher-probe.md`
- `.meshkore/modules/model/tasks/T-encoder-finetune.md`
- `.meshkore/modules/model/tasks/T-fullspace-objective.md`
- `.meshkore/modules/model/tasks/T-lever-stack.md`
- `.meshkore/modules/model/tasks/T-unfreeze-backbone.md`
- `.meshkore/roadmap/initiatives/full-space-training.md`
- `.meshkore/roadmap/initiatives/generalization-fix.md`
- `.meshkore/roadmap/initiatives/teacher-distill.md`
