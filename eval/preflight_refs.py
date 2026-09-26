"""Las tres referencias ANTES de entrenar (#T-preflight-refs).

Paso 2 del piloto y una PUERTA: si ni Qwen responde bien a la batería de
desarrollo, el problema está en la tarea, los datos o el formato, y
entrenar sería gastar la GPU en un enunciado roto. Medir antes de
entrenar es lo que la fase 1 no hizo.

Tres columnas sobre las MISMAS 400 filas de `#T-battery-dev`:

1. **Qwen local** — la referencia de capacidad, por **elección
   estructurada entre ids válidos** (`data.episode_gen.teacher_structured`),
   nunca texto libre. Es el techo práctico que hay en casa.
2. **El scorer NLI sin ajustar** de `#T-ce-scorer` — entailment con las
   hipótesis explícitas de `model.ce_scorer`, sin tocar un peso.
3. **Los checkpoints pointer** (leverstack 1 M y fullspace 62 528) como
   **CONTROL**: aquí el fallo ya medido en
   `.meshkore/docs/evidence/probe-mechanism-2026-09-24.json` tiene que
   reproducirse sobre la batería nueva. Si no se reproduce, la batería no
   es representativa y hay que revisarla antes de creerse nada.

Lo que este módulo garantiza en vez de suponerlo
------------------------------------------------

* **Mismo corte, mismas filas.** `same_rows()` compara las columnas fila a
  fila —id, familia, grupo, K y slot del gold— y LEVANTA
  `ProtocolMismatch` si no coinciden. El runner no escribe nada antes de
  pasar por ahí: dos columnas con n distinto no son una comparación, son
  dos mediciones que se parecen.
* **Protocolo de información equivalente.** Un único formato por modelo,
  fijado antes de medir y derivado del contrato comparativo de
  `episode-v1`: las familias con `COMPARATIVE_CONTEXT True` llevan el
  bloque de candidatos en la premisa y las demás no. La huella del
  formato (`CE.HYPOTHESIS_FORMAT_ID`) viaja en cada artefacto.
* **Cero aritmética a mano.** Toda cifra sale de
  `eval.metrics_suite.report()` y pasa por `require()`. Este módulo
  calcula CONTROLES (seguimiento del hecho decisivo, invariancia de la
  permutación), que son propiedades contadas, no accuracies: las
  accuracies las publica la suite o no se publican.
* **El test sellado no se toca.** Sólo se lee `data/battery_dev.jsonl` y
  el manifiesto de desarrollo; `assert_not_sealed()` rechaza cualquier
  ruta del corte sellado y `load_cut()` rechaza un corte cuyos grupos no
  lleven el prefijo `dev-`.

Lo que NO hace: entrenar, ajustar un umbral o elegir el formato después de
ver una cifra. El ajuste es `#T-ce-finetune`; la calibración de producto,
`#T-battery-calib`.

CPU/GPU: la referencia Qwen y la NLI y el control pointer necesitan
cómputo real y se lanzan por JOB (`preflight-refs`), nunca dentro de un
turno de agente. Todo lo demás de este módulo —protocolo, coincidencia de
filas, controles, lectura de la evidencia, gate— es stdlib y corre en
milisegundos sin cargar un solo peso.

CLI (stdlib, cualquier python3, desde la raíz del repo):

    PYTHONPATH=. python3 -m eval.preflight_refs protocol   # el formato fijado
    PYTHONPATH=. python3 -m eval.preflight_refs gate       # gate sin cómputo
    PYTHONPATH=. .venv-train/bin/python -m eval.preflight_refs refs   # JOB
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import battery_dev as BD  # noqa: E402
from eval import calib as C  # noqa: E402
from eval import metrics_suite as MS  # noqa: E402
from model import ce_scorer as CE  # noqa: E402

TASK = "T-preflight-refs"
GATE_DIR = ROOT / "artifacts" / "gates" / TASK
GATE_PATH = GATE_DIR / "gate.json"

#: El corte: desarrollo, 400 casos, NO reservado. El sellado es otro.
CUT_NAME = "battery-dev"
CUT_SPLIT = "development"
BATTERY = Path(BD.BATTERY)
BATTERY_MANIFEST = Path(BD.MANIFEST)

#: La evidencia del fallo de mecanismo que el control tiene que reproducir.
EVIDENCE = (ROOT / ".meshkore" / "docs" / "evidence"
            / "probe-mechanism-2026-09-24.json")

#: Las ÚNICAS rutas que las referencias leen. Una referencia que necesite
#: otra cosa se declara aquí y en el gate, o no se mide.
SOURCES = (
    "data/battery_dev.jsonl",
    "data/battery_dev_manifest.json",
    ".meshkore/docs/evidence/probe-mechanism-2026-09-24.json",
)

#: Lo que delata una ruta del corte sellado. `#T-battery-sealed` aún no
#: existe en disco: el guardia se escribe AHORA, no cuando el fichero
#: aparezca y alguien lo lea sin darse cuenta.
SEALED_MARKERS = ("battery_sealed", "battery-sealed", "T-battery-sealed",
                  "battery_diag", "battery-diag")
SEALED_GROUP = re.compile(r"^sealed-")
DEV_GROUP = re.compile(r"^dev-")

# -- las tres referencias --------------------------------------------------

REF_QWEN = "qwen-local"
REF_NLI = "nli-nograd"
REF_POINTER = "pointer-control"

#: Los dos checkpoints pointer del control, con el `model_version` que la
#: evidencia del 2026-09-24 ya les puso: la comparación es contra ESAS
#: cifras, así que el nombre tiene que cuadrar.
POINTER_CHECKPOINTS = {
    "leverstack-1m": {
        "dir": "artifacts/checkpoints/decision/"
               "leverstack-d512-prior-ettin-68m-s20260922/stage-001000000",
        "model_version": "jev-dec-ettin-68m-d512l2h8-s20260922-"
                         "n1000000-be9f0d454360",
    },
    "fullspace-62528": {
        "dir": "artifacts/checkpoints/decision/"
               "fullspace-sampled-d512-prior-ettin-68m-s20260922/"
               "stage-000062500",
        "model_version": "jev-dec-ettin-68m-d512l2h8-s20260922-"
                         "n62528-62ee2f57d7e6",
    },
}

REFERENCES = {
    REF_QWEN: {
        "id": REF_QWEN,
        "role": "capacity reference — the practical ceiling available here",
        "protocol": "structured choice among the valid candidate ids "
                    "(one JSON object, `choice` + `evidence` copied from "
                    "the state); free text is never parsed for an answer",
        "weights": "INDICATOR, not a distribution: a structured choice "
                   "yields a chosen id and nothing else, so the row's "
                   "weights are 1 on the pick and 0 elsewhere. NLL, Brier "
                   "and ECE of this column read that indicator and are "
                   "published only because the suite is whole-or-nothing",
        "abstains": "yes — a reply that is not one of the offered ids, or "
                    "whose evidence is not a span of the state, is "
                    "recorded as `unknown`, never as a guess",
        "needs_compute": True,
        "signed_by": "operator (job `preflight-refs`)",
    },
    REF_NLI: {
        "id": REF_NLI,
        "role": "untuned starting point — kept for #T-ce-finetune if it "
                "answers reasonably; a NLI head is not assumed to solve "
                "arithmetic or complex preferences without adaptation",
        "protocol": "entailment with the explicit hypotheses of "
                    "`model.ce_scorer` (HYPOTHESIS), one pair per "
                    "candidate, reduced to a scalar by "
                    f"`{CE.DEFAULT_SCORE_MODE}`",
        "weights": "softmax over the K candidate scores — relative to the "
                   "offered candidates, as the suite stamps",
        "abstains": "no — the untuned scorer carries no threshold, so its "
                    "K weights have no `unknown` column and its "
                    "abstention figure equals its forced figure by "
                    "construction",
        "needs_compute": True,
        "signed_by": "operator (job `preflight-refs`)",
    },
    REF_POINTER: {
        "id": REF_POINTER,
        "role": "CONTROL — the documented mechanism failure has to "
                "reproduce on this battery, or the battery is not "
                "representative",
        "protocol": "the trained pointer head over `[K + 1]`: the state "
                    "encoded once, the question and the K option texts "
                    "embedded, `unknown` as the extra column",
        "weights": "softmax over [K + 1] logits, `unknown` included",
        "abstains": "yes — `unknown` is a column of the head itself",
        "needs_compute": True,
        "signed_by": "operator (job `preflight-refs`)",
        "checkpoints": sorted(POINTER_CHECKPOINTS),
    },
}

#: El profesor externo no bloquea: `#teacher-distill` está en backlog y su
#: papel de referencia lo cubre Qwen (task body). Cuando haya acceso
#: válido y protocolo comparable se añade como CUARTA columna aquí.
TEACHER_COLUMN = {
    "id": "typesafe-teacher",
    "present": False,
    "why": "#teacher-distill is in backlog and its reference role is "
           "covered by Qwen; the 2026-09-23 key answers 401 against "
           "https://api.typesafe.ai. It is added as a fourth column when "
           "there is valid access AND a comparable protocol, never as a "
           "number quoted from another protocol",
}

# -- reglas escritas ANTES de medir ---------------------------------------

#: La propiedad que el control tiene que reproducir, en una frase: el
#: modelo elige el MISMO texto de opción aunque el hecho decisivo del
#: estado se mueva. La batería la mide por grupo contrafactual: el grupo
#: comparte pregunta y textos de candidato, y sólo cambia el estado.
TRACKING_WHAT = MS.TRACKING_WHAT

#: Umbrales FIJADOS aquí, antes de que exista una sola medición sobre la
#: batería. Mover uno de estos después de ver una cifra invalida la
#: lectura, y el gate publica el valor con el que se decidió.
REPRO_RULE = {
    "documented_invariance_min": 0.5,
    "battery_invariance_min": 0.5,
    "joint_at_or_below_chance": True,
    "why": "the documented failure is an INVARIANCE: the forced pick does "
           "not move when the decisive fact moves. It reproduces on the "
           "battery when (a) the evidence file itself shows that "
           "invariance in at least half of its paired probes, (b) at "
           "least half of the battery's counterfactual groups get the "
           "same slot picked in both halves, and (c) the joint "
           "counterfactual interval published by the suite sits at or "
           "below the joint chance rate — one half of a pair is what a "
           "fixed preference for a text already gets for free",
}

#: La puerta de Qwen. El piloto sigue si la referencia de capacidad
#: responde por encima del azar de forma no ambigua sobre la batería: el
#: extremo inferior del IC95 % de la elección forzada por encima del azar
#: del corte, en la macro por familia y en las cinco familias. Escrito
#: aquí, antes de medir, para que la puerta no se mueva al ver la cifra.
QWEN_GATE_RULE = {
    "decided_on": "the ranking (forced choice) figure of "
                  f"`{MS.__name__}.report`, plus the per-family breakdown",
    "opens_if": "ranking accuracy_ci95[0] > chance on the whole cut AND "
                "the macro-by-family figure clears its chance the same "
                "way AND no family's interval sits at or below its own "
                "chance rate",
    "closes_if": "any of the three fails — then the task, the data or the "
                 "format is what gets revised, and the GPU is not spent",
    "why_not_a_single_number": "a global mean passes with one family at "
                              "zero, which is the failure the battery was "
                              "written to make visible",
}

#: De qué depende ELEGIR el checkpoint de partida del ajuste. No es una
#: opinión: son las cifras de las columnas 2 y 3 sobre el mismo corte.
STARTING_POINT_RULE = {
    "candidates": [REF_NLI] + [
        f"{REF_POINTER}:{name}" for name in sorted(POINTER_CHECKPOINTS)],
    "decided_on": "the ranking figure and the macro-by-family figure of "
                  "each candidate on this cut, with the joint "
                  "counterfactual figure beside them",
    "rule": "the starting point is the candidate whose ranking interval "
            "sits highest AND whose joint counterfactual figure is not at "
            "or below chance; a candidate that reproduces the documented "
            "invariance is not a starting point, it is the control",
    "contamination": "modernbert-zeroshot-v2 published mix includes "
                     "BANKING77 (model.weights.SCORERS); that caveat "
                     "travels with any figure of that checkpoint",
}


class ProtocolMismatch(ValueError):
    """Las columnas no midieron las mismas filas: no hay comparación."""


class SealedCutTouched(RuntimeError):
    """Una referencia intentó leer el corte sellado."""


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rel(path) -> str:
    """La ruta relativa a la raíz, o la absoluta si cae fuera (tests)."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------- el sellado no se toca

def assert_not_sealed(path) -> str:
    """Rechaza cualquier ruta que huela al corte sellado o al diagnóstico.

    Se comprueba la ruta, no la intención: un `open()` del sellado desde
    una referencia de preflight es una fuga aunque el autor no quisiera.
    """
    text = str(path)
    for marker in SEALED_MARKERS:
        if marker in text:
            raise SealedCutTouched(
                f"{text}: a preflight reference may not read the sealed "
                f"cut (marker {marker!r}). The sealed battery is opened "
                "once, by #T-battery-sealed, with the arm already chosen")
    return text


def sealed_files_present() -> list:
    """Los ficheros del sellado que YA existan, para decirlo en el gate."""
    return [_rel(p) for p in BD.find_sealed_files()]


def load_cut(path=BATTERY) -> list:
    """Las 400 filas de desarrollo. Ninguna otra puerta de entrada.

    Rechaza la ruta si parece del sellado y rechaza el contenido si algún
    `variant_group` no lleva el prefijo `dev-`: el prefijo es la garantía
    de disyunción por construcción que `#T-battery-dev` ya enforce.
    """
    assert_not_sealed(path)
    episodes = BD.load_battery(str(path))
    bad = [ep.get("id") for ep in episodes
           if SEALED_GROUP.match(str(ep.get("variant_group", "")))
           or not DEV_GROUP.match(str(ep.get("variant_group", "")))]
    if bad:
        raise SealedCutTouched(
            f"{len(bad)} rows whose variant_group is not `dev-` "
            f"({bad[:4]}): this loader reads the development cut and "
            "nothing else")
    return episodes


# ------------------------------------------- protocolo de información

def protocol() -> dict:
    """El formato, UNO por modelo, fijado antes de cualquier medición."""
    return {
        "what": "one input format per model, written before any "
                "measurement on this cut and derived from the episode-v1 "
                "comparative contract — never chosen after looking at a "
                "result",
        "hypothesis_format_id": CE.HYPOTHESIS_FORMAT_ID,
        "hypothesis_format_fingerprint": CE.format_fingerprint(),
        "premise": dict(CE.PREMISE),
        "hypothesis": dict(CE.HYPOTHESIS),
        "comparative_context_by_family": dict(CE.COMPARATIVE_CONTEXT),
        "score_mode": CE.DEFAULT_SCORE_MODE,
        "truncation": CE.TRUNCATION_STRATEGY,
        "per_model": {
            REF_QWEN: "the same state, question and candidate block, as a "
                      "structured-choice prompt over the valid ids; the "
                      "candidate block is present for every family "
                      "because a chat model is shown the options it must "
                      "choose between",
            REF_NLI: "premise + hypothesis per candidate, comparative "
                     "context by family as the contract says",
            REF_POINTER: "the head's own interface: state encoded once, "
                         "question and K option texts embedded, "
                         "`unknown` as the extra column",
        },
        "difference_is_documented": "the three interfaces are not the "
                                    "same strings and cannot be: a chat "
                                    "model needs the options enumerated, "
                                    "a NLI head needs one hypothesis at a "
                                    "time, and the pointer head reads "
                                    "embeddings. What is held fixed is "
                                    "the INFORMATION: the same state, the "
                                    "same question, the same K candidate "
                                    "texts, the same rows, and one format "
                                    "per model fixed in advance",
    }


def decision_of(episode: dict) -> CE.Decision:
    """Un episodio como `Decision`, con el contrato de su familia."""
    return CE.Decision(
        state=episode["state"], question=episode["question"],
        candidates=tuple(CE.Candidate(c["id"], c["text"])
                         for c in episode["candidates"]),
        family=episode["family"], lang=episode["lang"],
        gold=episode.get("answer"),
        meta={"id": episode["id"], "group": episode["variant_group"]})


def decisions_of(episodes: list) -> list:
    return [decision_of(ep) for ep in episodes]


def gold_index(episode: dict) -> int:
    ids = [c["id"] for c in episode["candidates"]]
    return ids.index(episode["answer"])


def row_of(episode: dict, pred: int, probs: list) -> dict:
    """Una fila en la forma que `eval.metrics_suite._row` exige."""
    return {
        "row_id": episode["id"],
        "family": episode["family"],
        "variant_group": episode["variant_group"],
        "k": len(episode["candidates"]),
        "gold_index": gold_index(episode),
        "pred": int(pred),
        "probs": [float(p) for p in probs],
    }


def cut_descriptor(episodes: list) -> dict:
    """De dónde salen las filas, con su sello de contenido."""
    return {
        "name": CUT_NAME,
        "reserved": False,
        "split": CUT_SPLIT,
        "task": "T-battery-dev",
        "file": "data/battery_dev.jsonl",
        "manifest": "data/battery_dev_manifest.json",
        "manifest_sha256": sha256_file(BATTERY_MANIFEST),
        "split_sha256": BD.battery_sha(episodes),
        "sealed_cut_read": False,
        "sealed_files_present": sealed_files_present(),
        "rule": "the development cut, read as often as needed. The sealed "
                "cut is opened once by #T-battery-sealed with the arm "
                "already chosen, and nothing here reads it",
    }


# ----------------------------------------------------- las tres columnas

def _onehot(k: int, index: int) -> list:
    """Indicador sobre `[K + 1]`: la elección estructurada no da pesos."""
    probs = [0.0] * (k + 1)
    probs[index] = 1.0
    return probs


def qwen_column(episodes: list, choose=None) -> tuple:
    """Columna 1 — Qwen local por elección estructurada entre ids válidos.

    `choose(state, question, candidates) -> trace` es inyectable: los
    tests le pasan una función determinista y NUNCA levantan un modelo.
    Por defecto es `data.episode_gen.teacher_structured`, que ya descarta
    cualquier confianza que el modelo añada y exige que la evidencia sea
    un tramo del estado.
    """
    if choose is None:
        from data.episode_gen import teacher_structured as choose  # noqa: E501
    rows, traces = [], []
    for ep in episodes:
        ids = [c["id"] for c in ep["candidates"]]
        k = len(ids)
        trace = choose(ep["state"], ep["question"], ep["candidates"]) or {}
        pick = trace.get("choice")
        index = ids.index(pick) if pick in ids else k
        rows.append(row_of(ep, index, _onehot(k, index)))
        traces.append({"row_id": ep["id"], "choice": pick,
                       "abstained": index == k,
                       "reject": trace.get("reject")})
    return rows, {"reference": REF_QWEN, "n": len(rows),
                  "abstained": sum(1 for t in traces if t["abstained"]),
                  "rejected_replies": sum(1 for t in traces
                                          if t.get("reject")),
                  "traces": traces}


def nli_column(episodes: list, score=None,
               mode: str = CE.DEFAULT_SCORE_MODE) -> tuple:
    """Columna 2 — el scorer NLI SIN AJUSTAR, por entailment.

    `score(pairs) -> list[float]` es inyectable y recibe los pares en el
    orden de `CE.flatten_pairs`: el MISMO aplanado que usa el entreno, que
    es la vía por la que dos implementaciones acaban leyendo strings
    distintos (#T-ce-mechanics).
    """
    decisions = decisions_of(episodes)
    pairs, spans = CE.flatten_pairs(decisions)
    scores = list(score(pairs))
    if len(scores) != len(pairs):
        raise ProtocolMismatch(
            f"the scorer returned {len(scores)} scores for {len(pairs)} "
            "pairs: a column with a hole in it is not a measurement")
    rows = []
    for ep, (lo, hi) in zip(episodes, spans):
        z = scores[lo:hi]
        probs = CE.softmax(z)
        pred = max(range(len(z)), key=lambda i: z[i])
        rows.append(row_of(ep, pred, probs))
    return rows, {"reference": REF_NLI, "n": len(rows), "pairs": len(pairs),
                  "score_mode": mode,
                  "abstains": REFERENCES[REF_NLI]["abstains"]}


def pointer_column(episodes: list, logits=None) -> tuple:
    """Columna 3 — el control: el pointer head sobre `[K + 1]`.

    `logits(decisions) -> list[list[float]]`, una lista de K+1 logits por
    fila. Inyectable por la misma razón: el test del mecanismo no carga
    pesos, usa un scorer determinista que reproduce el fallo documentado.
    """
    decisions = decisions_of(episodes)
    out = list(logits(decisions))
    if len(out) != len(episodes):
        raise ProtocolMismatch(
            f"the head returned {len(out)} rows for {len(episodes)} "
            "episodes: a column with a hole in it is not a measurement")
    rows = []
    for ep, values in zip(episodes, out):
        k = len(ep["candidates"])
        if len(values) != k + 1:
            raise ProtocolMismatch(
                f"{ep['id']}: {len(values)} logits for K={k}: the pointer "
                "column is [K + 1], `unknown` included")
        probs = C.softmax(list(values), 1.0)
        pred = max(range(k + 1), key=lambda i: probs[i])
        rows.append(row_of(ep, pred, probs))
    return rows, {"reference": REF_POINTER, "n": len(rows),
                  "abstained": sum(1 for r in rows if r["pred"] >= r["k"])}


# ---------------------------------------- mismas filas, o no hay tabla

def _identity(row: dict) -> tuple:
    return (row["row_id"], row["family"], row["variant_group"],
            row["k"], row["gold_index"])


def same_rows(columns: dict) -> dict:
    """Las columnas midieron las MISMAS filas, o esto levanta.

    El test que el gate de la task pide: mismo corte, mismas filas, mismo
    protocolo, y el runner FALLA si los n no coinciden. Comparar dos
    columnas de n distinto es publicar una diferencia que puede ser
    enteramente del muestreo.
    """
    if len(columns) < 2:
        raise ProtocolMismatch(
            f"{len(columns)} column(s): there is nothing to compare, and a "
            "single column published as a reference table is a table of one")
    sizes = {name: len(rows) for name, rows in columns.items()}
    if len(set(sizes.values())) != 1:
        raise ProtocolMismatch(
            f"the columns do not share their n: {sizes} — same cut, same "
            "rows, or no comparison")
    names = sorted(columns)
    base = [_identity(r) for r in columns[names[0]]]
    for name in names[1:]:
        other = [_identity(r) for r in columns[name]]
        if other != base:
            first = next((i for i, (a, b) in enumerate(zip(base, other))
                          if a != b), None)
            raise ProtocolMismatch(
                f"{name} does not measure the same rows as {names[0]}: "
                f"first difference at index {first} "
                f"({base[first] if first is not None else '?'} vs "
                f"{other[first] if first is not None else '?'})")
    payload = json.dumps(base, sort_keys=True, ensure_ascii=False)
    return {
        "pass": True,
        "columns": names,
        "n": sizes[names[0]],
        "rows_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "compared_on": "row_id, family, variant_group, K and the gold slot "
                       "— identity, not size: two columns of 400 rows each "
                       "can still be two different cuts",
        "rule": "the runner FAILS if the n of the columns do not match",
    }


# ----------------------------- el control: ¿sigue el hecho decisivo?

def _forced(row: dict) -> int:
    """Argmax sobre los candidatos OFRECIDOS — `unknown` fuera."""
    probs = row["probs"][:row["k"]]
    return max(range(row["k"]), key=lambda j: probs[j])


def tracking_control(episodes: list, rows: list, *, source: str) -> dict:
    """¿Se mueve la elección cuando se mueve el hecho decisivo?

    El control que `eval.metrics_suite` NO inventa y espera recibir. La
    batería lo hace medible por construcción: los miembros de un grupo
    comparten pregunta y textos de candidato, y sólo cambia el estado, así
    que el gold se mueve y la elección debería moverse con él.

    Devuelve el bloque que se le pasa a `report(tracking=...)`, con la
    invariancia contada al lado: `followed` y `same_slot` son la misma
    medición leída por sus dos caras.
    """
    by_id = {r["row_id"]: r for r in rows}
    per_group: dict = {}
    for ep in episodes:
        role = BD.role_of(ep.get("id", ""))
        if role in ("orig", "decisive"):
            per_group.setdefault(ep["variant_group"], {})[role] = ep["id"]
    followed, same, pairs = 0, 0, []
    for group, members in sorted(per_group.items()):
        if set(members) != {"orig", "decisive"}:
            continue
        a, b = by_id.get(members["orig"]), by_id.get(members["decisive"])
        if a is None or b is None:
            continue
        if a["gold_index"] == b["gold_index"]:
            continue  # the gold rule moves it; a group where it did not is
                      # no probe of tracking, so it is not counted
        moved = _forced(a) != _forced(b)
        followed += bool(moved)
        same += (not moved)
        pairs.append({"variant_group": group, "moved": bool(moved),
                      "orig_pick": _forced(a), "decisive_pick": _forced(b),
                      "orig_gold": a["gold_index"],
                      "decisive_gold": b["gold_index"]})
    n = len(pairs)
    return {
        "what": TRACKING_WHAT,
        "measured": True,
        "n": n,
        "followed": followed,
        "rate": round(followed / n, 6) if n else None,
        "same_slot": same,
        "same_slot_rate": round(same / n, 6) if n else None,
        "pass": bool(n and followed / n
                     >= REPRO_RULE["battery_invariance_min"]),
        "source": source,
        "how": "per counterfactual group: the members share the question "
               "and the candidate texts and differ only in the decisive "
               "fact of the state, so the gold slot moves; `followed` "
               "counts the groups whose forced pick moved with it and "
               "`same_slot` the groups where it did not",
        "pairs": pairs[:8],
    }


# ------------------------------------- la evidencia del 2026-09-24

def _probe_pairs(cases: list) -> list:
    """Los pares de la evidencia: mismo enunciado, gold distinto.

    Dos formas, las dos del fallo documentado: misma PREGUNTA con el hecho
    del estado movido, y mismo ESTADO con la pregunta movida.
    """
    pairs = []
    for i, a in enumerate(cases):
        for b in cases[i + 1:]:
            if a["gold"] == b["gold"]:
                continue
            if a["question"] == b["question"] and a["state"] != b["state"]:
                shape = "state_moved"
            elif a["state"] == b["state"] and a["question"] != b["question"]:
                shape = "question_moved"
            else:
                continue
            pairs.append({
                "shape": shape, "a": a["name"], "b": b["name"],
                "gold_a": a["gold"], "gold_b": b["gold"],
                "forced_a": a.get("forced_choice"),
                "forced_b": b.get("forced_choice"),
                "invariant": a.get("forced_choice") == b.get("forced_choice"),
            })
    return pairs


def documented_signature(path=EVIDENCE) -> dict:
    """La firma del fallo, LEÍDA de la evidencia, no recordada.

    No carga un peso: lee el JSON que la sonda del 2026-09-24 dejó y
    cuenta en cuántos de sus pares la elección forzada NO se movió aunque
    el gold sí. Ese número es con lo que se compara la batería.
    """
    assert_not_sealed(path)
    doc = json.loads(Path(path).read_text())
    runs = {}
    for run in doc.get("runs", []):
        pairs = _probe_pairs(run.get("cases", []))
        invariant = sum(1 for p in pairs if p["invariant"])
        runs[run["model_version"]] = {
            "checkpoint": run["checkpoint"],
            "model_version": run["model_version"],
            "n_pairs": len(pairs),
            "invariant": invariant,
            "rate": round(invariant / len(pairs), 6) if pairs else None,
            "shapes": sorted({p["shape"] for p in pairs}),
            "pairs": pairs,
        }
    return {
        "source": _rel(path),
        "source_sha256": sha256_file(path),
        "scope": doc.get("scope"),
        "what": "the forced pick does not move when the decisive fact "
                "moves — counted over the probe's own paired cases",
        "runs": runs,
        "citation": True,
        "who": "T-ce-mechanics probe 2026-09-24",
    }


def permutation_from_evidence(model_version: str, path=EVIDENCE) -> dict:
    """La invariancia de la permutación, traída de donde se midió.

    `eval.metrics_suite` exige este control y se niega a inventarlo. Para
    los checkpoints pointer ya está medido caso a caso en la evidencia del
    2026-09-24 (`permutation_max_abs_probability_error`), así que se
    carries in con su procedencia en vez de volver a pasar la GPU.
    """
    assert_not_sealed(path)
    doc = json.loads(Path(path).read_text())
    for run in doc.get("runs", []):
        if run["model_version"] != model_version:
            continue
        key = "permutation_max_abs_probability_error"
        deltas = [c[key] for c in run.get("cases", [])
                  if c.get(key) is not None]
        if not deltas:
            break
        worst = max(deltas)
        return {
            "measured": True,
            "max_abs_prob_delta": worst,
            "tolerance": CE.PERMUTATION_TOL,
            "n_rows": len(deltas),
            "pass": worst <= CE.PERMUTATION_TOL,
            "source": f"{_rel(path)} "
                      f"({model_version}, 2026-09-24)",
            "carried_in": "measured elsewhere and copied with its "
                          "provenance; this module never synthesises a "
                          "control",
        }
    raise ProtocolMismatch(
        f"no permutation measurement for {model_version!r} in "
        f"{_rel(path)}: the control is carried in from "
        "where it was measured, never invented here")


def reproduces(report: dict, tracking: dict, signature: dict,
               model_version: str) -> dict:
    """¿Reproduce el control, en la batería, el fallo ya documentado?

    Las tres condiciones de `REPRO_RULE`, escritas antes de medir. La
    cifra de la parte (c) sale de la suite: aquí se LEE, no se calcula.
    """
    run = signature.get("runs", {}).get(model_version, {})
    documented = run.get("rate")
    battery = tracking.get("same_slot_rate")
    joint = (report or {}).get("counterfactual", {})
    ci = joint.get("accuracy_ci95") or [None, None]
    chance = joint.get("chance")
    checks = {
        "documented_invariance": {
            "value": documented,
            "min": REPRO_RULE["documented_invariance_min"],
            "pass": bool(documented is not None
                         and documented >= REPRO_RULE[
                             "documented_invariance_min"]),
            "n_pairs": run.get("n_pairs"),
        },
        "battery_invariance": {
            "value": battery,
            "min": REPRO_RULE["battery_invariance_min"],
            "pass": bool(battery is not None
                         and battery >= REPRO_RULE["battery_invariance_min"]),
            "n_groups": tracking.get("n"),
        },
        "joint_at_or_below_chance": {
            "joint_ci95": ci,
            "joint_chance": chance,
            "pass": bool(ci[1] is not None and chance is not None
                         and ci[1] <= chance),
            "read_from": f"{MS.__name__}.report /counterfactual — this "
                         "module reads the figure, it does not compute it",
        },
    }
    ok = all(c["pass"] for c in checks.values())
    return {
        "model_version": model_version,
        "rule": REPRO_RULE,
        "checks": checks,
        "reproduced": ok,
        "reading": (
            "the control reproduces on this battery the invariance the "
            "2026-09-24 probe documented: the forced pick stays put while "
            "the decisive fact moves, so a counterfactual group is never "
            "won whole" if ok else
            "the control does NOT reproduce the documented invariance on "
            "this battery: before believing any figure measured here, the "
            "battery itself is what gets revised — a cut that cannot show "
            "a failure already measured elsewhere is not representative"),
        "if_not_reproduced": "revise the battery (#T-battery-dev), not the "
                             "rule: the threshold was written before the "
                             "measurement",
    }


# --------------------------------------------- la temperatura del corte

#: Mitades del corte para la temperatura: por GRUPO, nunca por fila, para
#: que un contrafactual no quede repartido entre fit y verify. Declarado
#: aquí, antes de medir.
TEMP_SALT = "preflight-refs-temp-v1"


def _half(group: str) -> str:
    digest = hashlib.sha256(f"{TEMP_SALT}\x00{group}".encode()).hexdigest()
    return "fit" if int(digest, 16) % 2 == 0 else "verify"


def dev_temperature(rows: list) -> dict:
    """Una temperatura HONESTA sin salir del corte de desarrollo.

    La suite se publica entera o no se publica, y su sección de
    calibración exige una temperatura ajustada en un corte y verificada en
    otro. La del producto es `#T-battery-calib` (depende de
    `#T-ce-finetune` y aún no existe). Aquí se ajusta sobre la MITAD de
    los grupos de desarrollo y se verifica sobre la otra: es una
    temperatura real, medida, y dice exactamente de dónde sale.
    """
    fit = [r for r in rows if _half(r["variant_group"]) == "fit"]
    ver = [r for r in rows if _half(r["variant_group"]) == "verify"]
    if not fit or not ver:
        raise ProtocolMismatch(
            "the development cut did not split into two halves: a "
            "temperature fitted and verified on the same rows is a fit, "
            "not a calibration")
    logits = [[math.log(max(p, 1e-12)) for p in r["probs"][:r["k"]]]
              for r in fit]
    grid = C._fit_temp_grid(logits, [r["gold_index"] for r in fit])
    return {
        "temperature": grid["temperature"],
        "fitted_on": f"{CUT_NAME} fit half ({len(fit)} rows, "
                     f"{len({r['variant_group'] for r in fit})} groups)",
        "verified_on": f"{CUT_NAME} verify half ({len(ver)} rows, "
                       f"{len({r['variant_group'] for r in ver})} groups)",
        "strategy": "argmax over [K + 1]; no abstention threshold is "
                    "fitted here — that is #T-battery-calib",
        "threshold": None,
        "split_rule": f"by variant_group, sha256('{TEMP_SALT}' + group) "
                      "parity, so a counterfactual group never straddles "
                      "the two halves",
        "nll_before": grid["nll_before"],
        "nll_after": grid["nll_after"],
        "not_the_products_calibration": "a temperature fitted inside the "
                                        "development cut to make this "
                                        "report publishable; the "
                                        "product's calibration and its "
                                        "abstention threshold are "
                                        "#T-battery-calib",
    }


# ----------------------------------------------------------- la tabla

def report_for(reference: str, rows: list, *, episodes: list,
               model_version: str, permutation: dict,
               calibration: dict | None = None,
               notes: list | None = None) -> dict:
    """La suite, y nada que este módulo haya calculado por su cuenta."""
    tracking = tracking_control(
        episodes, rows,
        source=f"{TASK} on {CUT_NAME} {BD.battery_sha(episodes)[:12]}")
    doc = MS.report(
        rows,
        cut=cut_descriptor(episodes),
        model_version=model_version,
        task=TASK,
        calibration=calibration or dev_temperature(rows),
        permutation=permutation,
        tracking=tracking,
        notes=list(notes or []) + [
            f"reference: {reference} — {REFERENCES[reference]['role']}",
            f"protocol: {REFERENCES[reference]['protocol']}",
            f"weights: {REFERENCES[reference]['weights']}",
            f"abstention: {REFERENCES[reference]['abstains']}",
            f"format id: {CE.HYPOTHESIS_FORMAT_ID}",
            "the sealed cut was not read: "
            + ", ".join(SOURCES),
        ])
    return MS.require(doc)


# ------------------------------------------------------------- el gate

def measured_now(episodes: list | None = None) -> dict:
    """Todo lo que se puede medir SIN cómputo, medido de verdad.

    El protocolo, la coincidencia de filas de las tres columnas sobre las
    400 filas reales (con columnas deterministas, sin un solo peso), la
    firma documentada leída de la evidencia y la ausencia de lectura del
    sellado. Nada de esto necesita GPU y nada de esto es una promesa.
    """
    episodes = episodes if episodes is not None else load_cut()
    stub = {name: [row_of(ep, 0, _onehot(len(ep["candidates"]), 0))
                   for ep in episodes]
            for name in (REF_QWEN, REF_NLI, REF_POINTER)}
    agreement = same_rows(stub)
    signature = documented_signature()
    return {
        "cut": cut_descriptor(episodes),
        "protocol": protocol(),
        "same_rows": agreement,
        "documented_signature": signature,
        "columns_declared": sorted(REFERENCES),
        "teacher_column": TEACHER_COLUMN,
        "sources_read": list(SOURCES),
    }


def gate(write: bool = True, tests: dict | None = None,
         episodes: list | None = None) -> dict:
    """El gate de la task: lo medido con `pass`, lo pendiente con `null`.

    Un check que exige inferencia real se publica `pass: null` con
    `reason: awaiting-operator-compute` y con el nombre de quién lo firma
    y con qué job. Una cifra que no se ha medido no se escribe.
    """
    facts = measured_now(episodes)
    sig = facts["documented_signature"]["runs"]
    doc = {
        "task": TASK,
        "artifact": _rel(GATE_PATH),
        "generated_utc": utcnow(),
        "question": "do the three references answer this battery at all, "
                    "and does the pointer control reproduce here the "
                    "failure already documented?",
        "trains_nothing": True,
        "publishes_no_figure_of_its_own": (
            "every figure of this task is produced by "
            f"`{MS.__name__}.report` and lives in the per-reference "
            "reports this gate points at; this file publishes none, "
            "because nothing has been measured against a model yet"),
        "runner": "eval/preflight_refs.py",
        "job": {
            "id": "preflight-refs",
            "command": ".venv-train/bin/python -m eval.preflight_refs refs",
            "state": "registered STOPPED — the operator presses play; "
                     "nothing in this task started a model",
        },
        "cut": facts["cut"],
        "protocol": facts["protocol"],
        "references": REFERENCES,
        "teacher_column": TEACHER_COLUMN,
        "rules_written_before_measuring": {
            "reproduction": REPRO_RULE,
            "qwen_gate": QWEN_GATE_RULE,
            "starting_point": STARTING_POINT_RULE,
        },
        "checks": {
            "same_rows_over_the_three_columns": {
                "pass": facts["same_rows"]["pass"],
                "n": facts["same_rows"]["n"],
                "rows_sha256": facts["same_rows"]["rows_sha256"],
                "how_measured": "three deterministic columns over the real "
                                "400 rows through `same_rows()`; no model, "
                                "no weights, milliseconds",
                "needs_compute": False,
                "reading": "the three columns are built from one row list "
                           "and compared by identity, and the runner "
                           "raises ProtocolMismatch when an n or a row "
                           "identity differs",
            },
            "the_runner_fails_when_the_n_differ": {
                "pass": True,
                "how_measured": "eval/test_preflight_refs.py::SameRowsTest "
                                "— a column short by one row, a column "
                                "with a renamed row and a column with a "
                                "moved gold slot each raise "
                                "ProtocolMismatch",
                "needs_compute": False,
            },
            "no_reference_reads_the_sealed_cut": {
                "pass": True,
                "sealed_files_present": facts["cut"]["sealed_files_present"],
                "sources_read": facts["sources_read"],
                "how_measured": "eval/test_preflight_refs.py::SealedCutTest "
                                "— every path opened during a full "
                                "fixture run is recorded and matched "
                                "against the sealed markers; "
                                "`assert_not_sealed` is shown to raise on "
                                "each sealed path and `load_cut` to "
                                "refuse a cut whose groups are not `dev-`",
                "needs_compute": False,
            },
            "the_documented_failure_is_read_not_remembered": {
                "pass": all(r["rate"] is not None for r in sig.values()),
                "source": facts["documented_signature"]["source"],
                "source_sha256": facts["documented_signature"][
                    "source_sha256"],
                "runs": {mv: {"n_pairs": r["n_pairs"],
                              "invariant": r["invariant"],
                              "rate": r["rate"], "shapes": r["shapes"]}
                         for mv, r in sig.items()},
                "citation": True,
                "who": "T-ce-mechanics probe 2026-09-24",
                "how_measured": "documented_signature() over the evidence "
                                "JSON: the paired probes whose gold moved "
                                "and whose forced pick did not",
                "needs_compute": False,
                "reading": "both checkpoints kept the same forced pick in "
                           "every paired probe of the evidence file — the "
                           "invariance the battery has to show too",
            },
            "the_reproduction_check_discriminates": {
                "pass": True,
                "how_measured": "eval/test_preflight_refs.py::"
                                "MechanismReproductionTest — a "
                                "deterministic scorer that reads only the "
                                "option text reproduces the invariance "
                                "and `reproduces()` says so; a scorer "
                                "that reads the state does not, and "
                                "`reproduces()` says that instead. Hand "
                                "fixtures, no weights",
                "needs_compute": False,
                "reading": "the checker is shown to separate the two "
                           "cases, so a green reproduction later is not a "
                           "check that always says yes",
            },
            "c7_over_this_gate_directory": {
                "pass": True,
                "how_measured": "PYTHONPATH=. python3 -m eval.gate_rules "
                                "check artifacts/gates/T-preflight-refs "
                                "— this directory publishes no quality "
                                "metric, so the pilot rule has nothing to "
                                "flag; when the job writes the reports, "
                                "every figure in them carries the "
                                f"`{MS.FORMAT}` stamp",
                "needs_compute": False,
            },
            "the_reference_table": {
                "pass": None,
                "reason": "awaiting-operator-compute",
                "needs_compute": True,
                "what_it_will_publish": [
                    "forced (ranking) accuracy with its chance, K, n and "
                    "95 % interval",
                    "accuracy including abstention, with coverage and "
                    "precision among the answered",
                    "macro by family and the per-family breakdown",
                    "chance by K and the cut's own mean K",
                    "joint success by counterfactual group",
                ],
                "all_of_it_from": f"{MS.__name__}.report",
                "signed_by": "operator",
                "job": "preflight-refs",
                "columns": sorted(REFERENCES),
            },
            "the_qwen_gate": {
                "pass": None,
                "reason": "awaiting-operator-compute",
                "needs_compute": True,
                "rule": QWEN_GATE_RULE,
                "signed_by": "operator",
                "job": "preflight-refs",
            },
            "the_pointer_control_reproduces_on_the_battery": {
                "pass": None,
                "reason": "awaiting-operator-compute",
                "needs_compute": True,
                "rule": REPRO_RULE,
                "checkpoints": POINTER_CHECKPOINTS,
                "already_true_of_the_evidence": {
                    mv: r["rate"] for mv, r in sig.items()},
                "signed_by": "operator",
                "job": "preflight-refs",
            },
            "the_starting_checkpoint_of_the_finetune": {
                "pass": None,
                "reason": "awaiting-operator-compute",
                "needs_compute": True,
                "rule": STARTING_POINT_RULE,
                "signed_by": "operator",
                "job": "preflight-refs",
            },
        },
        "verdict": "AWAITING-COMPUTE",
        "pass": None,
        "honesty": [
            "this gate is HALF WRITTEN on purpose: the operator forbade "
            "loading Qwen, the pointer checkpoints or any inference loop "
            "over the 400 rows in this session, so every check that needs "
            "a forward pass carries `pass: null` and "
            "`reason: awaiting-operator-compute` instead of a number",
            "no figure in this file was computed by hand and none was "
            "estimated: what was not measured was not written",
            "the thresholds of the reproduction rule, the Qwen gate and "
            "the starting-point rule are written here BEFORE any "
            "measurement exists, which is the only moment at which "
            "writing them means anything",
            "the suite's calibration section needs a temperature fitted "
            "on one cut and verified on another; the product's is "
            "#T-battery-calib (it depends on #T-ce-finetune and does not "
            "exist yet), so the runner fits one inside the development "
            "cut, by group halves, and says so in the report",
        ],
    }
    if tests:
        doc["tests"] = tests
    if write:
        GATE_DIR.mkdir(parents=True, exist_ok=True)
        GATE_PATH.write_text(json.dumps(doc, indent=2, ensure_ascii=False,
                                        sort_keys=True) + "\n")
    return doc


# ----------------------------------------------------------- el runner

def _nli_scorer(weight_id: str, device: str = "cpu",
                batch_size: int = CE.DEFAULT_BATCH_SIZE):
    """El scorer NLI real. Necesita torch: sólo lo llama el job."""
    scorer = CE.NliPairScorer(weight_id, device=device,
                              batch_size=batch_size)

    def score(pairs):
        class_logits = scorer.class_logits(pairs)
        return CE.reduce_logits(class_logits, CE.DEFAULT_SCORE_MODE,
                               scorer.entail_index)

    return score, scorer


def _pointer_logits(ckpt_dir: str, device: str = "auto",
                    batch_size: int = 16):
    """El pointer head real. Necesita torch: sólo lo llama el job."""
    from data.optset import Sample
    from training.python import train_decision as T

    engine, manifest = T.load_checkpoint(str(ROOT / ckpt_dir), device)

    def logits(decisions):
        samples = [
            Sample(dataset=CUT_NAME, row_id=d.meta["id"],
                   question_id=d.meta["id"], state=d.state,
                   question=d.question,
                   options=[{"id": c.id, "text": c.text}
                            for c in d.candidates],
                   answer=d.gold or "",
                   gold_index=[c.id for c in d.candidates].index(d.gold))
            for d in decisions]
        entries = C.entries_from_samples(engine, samples, CUT_SPLIT,
                                         CUT_NAME, CUT_NAME, batch_size)
        return [e["logits"] for e in entries]

    return logits, manifest


def run(references: tuple = (REF_QWEN, REF_NLI, REF_POINTER),
        device: str = "auto", write: bool = True,
        nli_weights: str = "minilmv2-l6-mnli-xnli") -> dict:
    """Las tres referencias, de verdad. ESTO ES EL JOB, no un turno.

    Carga Qwen por ollama, el scorer NLI y los dos checkpoints pointer.
    No se llama desde un agente: se registra como job `preflight-refs` y
    lo arranca el operador.
    """
    episodes = load_cut()
    columns, traces, reports = {}, {}, {}

    if REF_NLI in references:
        score, scorer = _nli_scorer(nli_weights, device="cpu")
        columns[REF_NLI], traces[REF_NLI] = nli_column(episodes, score=score)
        traces[REF_NLI]["checkpoint"] = scorer.describe()

    if REF_QWEN in references:
        columns[REF_QWEN], traces[REF_QWEN] = qwen_column(episodes)

    pointer_meta = {}
    if REF_POINTER in references:
        for name, spec in sorted(POINTER_CHECKPOINTS.items()):
            logits, manifest = _pointer_logits(spec["dir"], device)
            key = f"{REF_POINTER}:{name}"
            columns[key], traces[key] = pointer_column(episodes,
                                                       logits=logits)
            pointer_meta[key] = {"manifest": manifest, **spec}

    agreement = same_rows(columns)  # FALLA aquí antes de escribir nada

    for key, rows in sorted(columns.items()):
        if key.startswith(REF_POINTER):
            mv = pointer_meta[key]["model_version"]
            perm = permutation_from_evidence(mv)
            ref = REF_POINTER
        else:
            ref = key
            mv = (traces[key].get("checkpoint", {}) or {}).get(
                "model_version") or traces[key].get("checkpoint")
            mv = mv if isinstance(mv, str) else json.dumps(mv,
                                                           sort_keys=True)
            perm = CE.check_permutation_invariance(
                CE.CrossEncoderScorer(scorer), CE.CONTRACT_CASE) \
                if ref == REF_NLI else None
            if perm is not None:
                perm = {"measured": True, **perm,
                        "source": f"{TASK} — model.ce_scorer."
                                  "check_permutation_invariance on the "
                                  "contract case, same weights, same run"}
        if perm is None:
            raise ProtocolMismatch(
                f"{key}: no permutation control. The suite refuses to "
                "invent it and so does this runner: measure it or do not "
                "publish the column")
        reports[key] = report_for(ref, rows, episodes=episodes,
                                  model_version=mv, permutation=perm)

    out = {"task": TASK, "generated_utc": utcnow(), "same_rows": agreement,
           "reports": reports, "traces": traces}
    if write:
        GATE_DIR.mkdir(parents=True, exist_ok=True)
        for key, doc in reports.items():
            name = "refs-" + key.replace(":", "-") + ".json"
            (GATE_DIR / name).write_text(
                json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return out


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(prog="eval.preflight_refs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("protocol", help="the fixed input format, per model")
    g = sub.add_parser("gate", help="the gate: measured checks + the ones "
                                    "awaiting the operator's compute")
    g.add_argument("--no-write", action="store_true")
    r = sub.add_parser("refs", help="THE JOB: measure the three "
                                    "references (loads models)")
    r.add_argument("--device", default="auto")
    r.add_argument("--nli-weights", default="minilmv2-l6-mnli-xnli")
    args = ap.parse_args(argv)

    if args.cmd == "protocol":
        print(json.dumps(protocol(), indent=2, ensure_ascii=False))
        return 0
    if args.cmd == "gate":
        doc = gate(write=not args.no_write)
        print(json.dumps({"task": doc["task"], "verdict": doc["verdict"],
                          "pass": doc["pass"],
                          "measured": sorted(
                              k for k, v in doc["checks"].items()
                              if v.get("pass") is not None),
                          "awaiting_compute": sorted(
                              k for k, v in doc["checks"].items()
                              if v.get("pass") is None)},
                         indent=2))
        return 0
    out = run(device=args.device, nli_weights=args.nli_weights)
    print(json.dumps({"task": out["task"],
                      "same_rows": out["same_rows"],
                      "columns": sorted(out["reports"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
