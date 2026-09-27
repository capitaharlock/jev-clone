"""Laya sobre nuestra batería — la cota superior externa (#T-laya-baseline).

Todo lo medido hasta ahora se compara contra el azar o contra nosotros
mismos. Laya (`github.com/NandhaKishorM/laya`, Apache-2.0, checkpoints
`convaiinnovations/laya` y `convaiinnovations/laya-multilingual`) responde
EXACTAMENTE nuestro formato de pregunta —`choice` sobre criterios con id—
en una sola pasada no autorregresiva. Pasarla por nuestra batería, SIN
ajustar un peso, convierte «no aprende» en un número con referencia.

Tres medidas, en este orden:

1. **Pares contrafactuales** de la batería de desarrollo (140 grupos): éxito
   conjunto por grupo, con IC95 %, al lado de nuestras columnas de
   `#T-preflight-refs` (pointer 1–2/140, nli-nograd 0,343, Qwen).
2. **La batería de desarrollo completa** (400 filas, mismo `rows_sha256` que
   el preflight): forzada, macro por familia, por idioma, IC95 %, con el
   control de permutación (opciones al revés) como el preflight.
3. **BANKING77 a K=77** por `eval.fullspace` con Laya como scorer externo,
   para saber si su 0,425 reproduce fuera de su harness.

Lo que este módulo garantiza en vez de suponerlo
------------------------------------------------

* **No importa código de Laya al repo.** El runner llama a la librería
  instalada en `.venv-laya` (`pip install laya`); en `.venv-train` no hace
  falta que exista: todo lo que no es la inferencia real —columna,
  control, informe, gate, tests— es stdlib e inyección.
* **Mismas filas, comprobado por hash.** La columna Laya pasa por
  `eval.preflight_refs.same_rows()` contra las columnas medidas del
  preflight, y el gate publica el `rows_sha256` de las cuatro.
* **Cero aritmética a mano.** Cada cifra sale de
  `eval.metrics_suite.report()` y pasa `require()`; el gate copia celdas.
  El control de permutación y el seguimiento del hecho decisivo son
  propiedades contadas, no accuracies.
* **Formato forzado, uno, fijado antes de medir:** el estado tal cual, la
  pregunta como `instructions`, cada candidato como criterio de un
  `choice` con su id opaco como etiqueta y su texto como descripción
  (Laya lo renderiza `c1: <texto>`). El checkpoint se elige por el idioma
  DECLARADO de la fila —`es` → multilingual, `en` → laya— que es lo que su
  `Router` haría por detección de script, sin la detección.
* **El sellado no se toca:** sólo `data/battery_dev.jsonl` por
  `preflight_refs.load_cut()`, que rechaza cualquier otro corte.
* **Mezcla publicada, por dataset:** el gate declara, citando
  `TMP/laya/BENCHMARKS.md` y `TMP/laya/README.md`, qué datasets estaban en
  la mezcla de entreno de Laya. BANKING77 no consta en la tabla de temas y
  su harness lo marca `in_training=False`; spam, phishing, AG News y BoolQ
  sí constan «in training».

CPU: la inferencia va en `--device cpu` (el acelerador lo usa el piloto de
datos). 400 filas + 400 permutadas con un encoder de 300–400 M tardan
minutos; BANKING77 (3 080 × K=77) tarda más: se lanza con nohup y log.

CLI:

    # sólo lo que no carga un modelo (stdlib, cualquier python3)
    PYTHONPATH=. python3 -m eval.laya_baseline protocol
    PYTHONPATH=. python3 -m eval.laya_baseline gate       # de lo ya medido

    # la inferencia real (necesita `laya` instalado: .venv-laya)
    PYTHONPATH=. .venv-laya/bin/python -m eval.laya_baseline battery --device cpu
    PYTHONPATH=. .venv-laya/bin/python -m eval.laya_baseline banking77 --device cpu
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import battery_dev as BD  # noqa: E402
from eval import metrics_suite as MS  # noqa: E402
from eval import preflight_refs as P  # noqa: E402
from model import ce_scorer as CE  # noqa: E402

TASK = "T-laya-baseline"
GATE_DIR = ROOT / "artifacts" / "gates" / TASK
GATE_PATH = GATE_DIR / "gate.json"
COLUMN_DIR = GATE_DIR / "columns"
REPORT_PATH = GATE_DIR / "refs-laya.json"
FULLSPACE_PATH = GATE_DIR / "fullspace.json"

COLUMN = "laya"
PACKAGE = "laya"

#: Los checkpoints, con el commit del Hub que se descarga. Los pins son los
#: `PINNED_REVISIONS` de `laya/revisions.py` en el clon (`TMP/laya`, main
#: 2026-09-27); el paquete 0.3.20 de PyPI no admite `revision`, así que la
#: descarga se hace aquí con `snapshot_download(revision=...)` y se carga
#: el directorio local. Así el gate nombra el commit y el sha256 de los
#: pesos, no «lo que hubiera en el Hub ese día».
CHECKPOINTS = {
    "laya": {
        "repo": "convaiinnovations/laya",
        "revision": "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
        "encoder": "ModernBERT-large (English, ~421 M)",
    },
    "laya-multilingual": {
        "repo": "convaiinnovations/laya-multilingual",
        "revision": "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67",
        "encoder": "mmBERT-base (100+ languages, ~300 M)",
    },
}
#: Qué checkpoint responde cada idioma DECLARADO de la fila.
ROUTE = {"en": "laya", "es": "laya-multilingual"}
#: Lo que el `Agent` descarga de un repo (mismo filtro que `laya.Agent`).
SNAPSHOT_PATTERNS = ("rl_agent_config.json", "model.safetensors",
                     "tokenizer/*", "encoder/*")

#: Cuántos decimales publica la API de Laya en `probabilities`. Un control
#: de permutación con tolerancia 1e-5 no puede distinguir ruido de
#: redondeo a 1e-4 de sensibilidad real, y se dice.
API_PROB_DECIMALS = 4

#: Lo que el harness de Laya declara de cada dataset que tocamos. Se cita
#: el fichero y la línea; nada de esto se afirma de memoria.
MIX_DECLARATION = {
    "battery-dev": {
        "in_laya_training_mix": False,
        "source": "data/battery_dev.jsonl is hand-written (#T-battery-dev, "
                  "2026-09-26) and has never left this repo",
        "transfer": "clean",
    },
    "banking77": {
        "in_laya_training_mix": False,
        "source": "TMP/laya/BENCHMARKS.md §'On the public datasets where Jev "
                  "numbers exist' lists banking77 without an in-training "
                  "mark; TMP/laya/research/scripts/bench_apps.py registers "
                  "`jev.banking77_full` with `in_training=False`; "
                  "build_benchmark_nb.py: 'sst5, emotion, prompt_injections, "
                  "banking77 were held out'",
        "transfer": "clean, as declared by the authors — the mix itself is "
                    "not published row by row, so this is their word",
    },
    "email-spam": {
        "in_laya_training_mix": True,
        "source": "TMP/laya/BENCHMARKS.md §Themes: 'Email spam … in training'",
        "transfer": "NOT clean",
    },
    "phishing": {
        "in_laya_training_mix": True,
        "source": "TMP/laya/BENCHMARKS.md §Themes: 'Phishing … in training'",
        "transfer": "NOT clean",
    },
    "ag_news": {
        "in_laya_training_mix": True,
        "source": "TMP/laya/README.md §'English tasks': 'AG News … in "
                  "training mix'",
        "transfer": "NOT clean",
    },
    "boolq": {
        "in_laya_training_mix": True,
        "source": "TMP/laya/README.md §'English tasks': 'BoolQ … in "
                  "training mix'",
        "transfer": "NOT clean",
    },
    "rag-passage-relevance": {
        "in_laya_training_mix": True,
        "source": "TMP/laya/BENCHMARKS.md §Themes: 'RAG passage relevance … "
                  "in training'",
        "transfer": "NOT clean",
    },
    "support-triage": {
        "in_laya_training_mix": True,
        "source": "TMP/laya/BENCHMARKS.md §Themes: 'Support triage … in "
                  "training'",
        "transfer": "NOT clean",
    },
}

#: Las cifras publicadas por Laya que este gate contrasta. Citas, no
#: mediciones nuestras: llevan `citation: true` y su fuente.
PUBLISHED = {
    "banking77_k77": {
        "who": "Laya (both base checkpoints)",
        "citation": True,
        "accuracy": 0.425,
        "n": 400,
        "setup": "400 test cases, seed 13, CPU, criteria = the 77 labels "
                 "with underscores replaced by spaces, state {'message': "
                 "text}, laya 0.2.1",
        "source": "TMP/laya/BENCHMARKS.md §'On the public datasets where Jev "
                  "numbers exist'; research/scripts/bench_apps.py",
        "caveat": "their harness, their 400 rows; ours below is the full "
                  "official test split through eval.fullspace",
    },
}

#: La regla del veredicto, escrita ANTES de medir: dos intervalos de la
#: misma cifra sobre las mismas filas. «Gana» = el inferior de Laya por
#: encima del superior nuestro; «pierde» = al revés; «empata» = se solapan;
#: «falla igual» = ninguno de los dos despeja su azar.
VERDICT_RULE = {
    "wins": "laya.accuracy_ci95[0] > ours.accuracy_ci95[1]",
    "loses": "laya.accuracy_ci95[1] < ours.accuracy_ci95[0]",
    "ties": "the intervals overlap",
    "fails_alike": "neither interval clears its chance by the lower bound",
    "why_intervals": "same rows, same K, so the two Wilson intervals are on "
                     "the same axis; a paired test is not published because "
                     "the suite does not compute one and no arithmetic is "
                     "done here",
}


class ProtocolMismatch(P.ProtocolMismatch):
    """La columna Laya no midió lo que dice medir."""


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rel(path) -> str:
    return P._rel(path)


# ------------------------------------------------ el formato forzado

def question_of(question: str, candidates: list) -> dict:
    """UNA pregunta `choice` con cada candidato como criterio con id.

    El id opaco es la etiqueta y el texto la descripción; Laya lo
    renderiza `c1: <texto>`. Cuando el texto ES el id (BANKING77: la
    etiqueta es su propio texto) se pasa sin descripción para no gastar el
    presupuesto de tokens del cabezal en repetirlo.
    """
    crit = {}
    for c in candidates:
        cid, text = str(c["id"]), str(c["text"])
        crit[cid] = None if text == cid else text
    return {"q": {"type": "choice", "instructions": question,
                  "criteria": crit}}


def align(probabilities: dict, ids: list) -> list:
    """Los pesos de la respuesta, en el orden de los candidatos ofrecidos."""
    missing = [i for i in ids if str(i) not in probabilities]
    if missing:
        raise ProtocolMismatch(
            f"the answer carries no weight for {missing}: a column with a "
            "hole in it is not a measurement")
    return [float(probabilities[str(i)]) for i in ids]


def protocol() -> dict:
    return {
        "what": "one input format, written before any measurement on this "
                "cut and never changed after seeing a figure",
        "state": "the episode state, verbatim, as a plain string",
        "question": "the episode question as `instructions` of ONE `choice`",
        "candidates": "each candidate is a criterion: its opaque id is the "
                      "label, its text the description — rendered by Laya "
                      "as `<id>: <text>`; a candidate whose text is its own "
                      "id (BANKING77) is rendered as the label alone",
        "routing": {**ROUTE,
                    "how": "by the row's DECLARED lang, not by script "
                           "detection: what Laya's Router would do without "
                           "the detector"},
        "weights": "Laya's `probabilities` per label, realigned by id to "
                   "the offered order — softmax over the K candidates, no "
                   "`unknown` column; the pick is the argmax",
        "abstains": "no — the API publishes no abstention, so the "
                    "abstention figure equals the forced figure by "
                    "construction",
        "api_rounding": f"{API_PROB_DECIMALS} decimals per weight, as the "
                        "API returns them",
        "budget": "the checkpoint's defaults (`max_len`, `head_max_len`): "
                  "nothing is raised, because the question is whether the "
                  "published number reproduces as shipped",
        "untuned": "no weight, temperature or threshold is touched",
        "device": "cpu",
    }


# ------------------------------------------------ la columna medida

def _trace_log(path) -> dict:
    return P._trace_log(path)


def _trace_append(path, entry: dict) -> None:
    P._trace_append(path, entry)


def laya_column(episodes: list, predict, log=None) -> tuple:
    """La columna Laya: una pasada por fila, reanudable por `row_id`.

    `predict(state, question, candidates, lang) -> {"probs": [...],
    "model": str, "ms": float}` es inyectable: los tests le pasan una
    función determinista y NUNCA descargan un peso. `probs` viene en el
    orden de `candidates`.
    """
    done = _trace_log(log)
    resumed = len(done)
    rows, traces = [], []
    for ep in episodes:
        ids = [c["id"] for c in ep["candidates"]]
        k = len(ids)
        seen = done.get(ep["id"])
        if seen is None:
            got = predict(ep["state"], ep["question"], ep["candidates"],
                          ep["lang"]) or {}
            probs = [float(p) for p in got.get("probs") or []]
            if len(probs) != k:
                raise ProtocolMismatch(
                    f"{ep['id']}: {len(probs)} weights for K={k}: the Laya "
                    "column is K wide, one weight per offered candidate")
            seen = {"row_id": ep["id"], "probs": probs,
                    "model": got.get("model"), "ms": got.get("ms")}
            _trace_append(log, seen)
            done[ep["id"]] = seen
        probs = [float(p) for p in seen["probs"]]
        pred = max(range(k), key=lambda i: probs[i])
        rows.append(P.row_of(ep, pred, probs))
        traces.append(seen)
    by_model: dict = {}
    for t in traces:
        by_model[t.get("model")] = by_model.get(t.get("model"), 0) + 1
    ms = [t["ms"] for t in traces if isinstance(t.get("ms"), (int, float))]
    return rows, {"reference": COLUMN, "n": len(rows),
                  "rows_by_model": by_model, "resumed_rows": resumed,
                  "ms_per_row_mean": round(sum(ms) / len(ms), 1) if ms
                  else None,
                  "abstains": False}


def permutation_control(episodes: list, rows: list, predict,
                        log=None) -> dict:
    """El control `#T-option-text`: las opciones al revés, realineadas por id.

    Laya devuelve una distribución, así que el control se lee como el del
    scorer NLI: el peso de cada id tiene que quedarse donde estaba. Se
    mide sobre TODAS las filas del corte, como pide la task, y publica
    además cuántas elecciones cambian de id (`flipped`), que es lo que el
    propio BENCHMARKS.md de Laya reporta como «option-order robustness».
    """
    by_id = {r["row_id"]: r for r in rows}
    done = _trace_log(log)
    worst, flips, pairs = 0.0, 0, []
    for ep in episodes:
        ids = [c["id"] for c in ep["candidates"]]
        seen = done.get(ep["id"])
        if seen is None:
            reversed_c = list(reversed(ep["candidates"]))
            got = predict(ep["state"], ep["question"], reversed_c,
                          ep["lang"]) or {}
            probs = [float(p) for p in got.get("probs") or []]
            if len(probs) != len(ids):
                raise ProtocolMismatch(
                    f"{ep['id']}: {len(probs)} weights for K={len(ids)} on "
                    "the reversed offer")
            # realigned to the ORIGINAL offer order, by id
            rev_ids = [c["id"] for c in reversed_c]
            seen = {"row_id": ep["id"],
                    "probs": [probs[rev_ids.index(i)] for i in ids]}
            _trace_append(log, seen)
            done[ep["id"]] = seen
        base = by_id[ep["id"]]["probs"]
        again = [float(p) for p in seen["probs"]]
        delta = max(abs(a - b) for a, b in zip(base, again))
        worst = max(worst, delta)
        first = max(range(len(base)), key=lambda i: base[i])
        second = max(range(len(again)), key=lambda i: again[i])
        moved = first != second
        flips += bool(moved)
        pairs.append({"row_id": ep["id"], "max_abs_delta": round(delta, 6),
                      "moved": bool(moved)})
    n = len(pairs)
    return {
        "measured": True,
        "max_abs_prob_delta": round(worst, 6),
        "tolerance": CE.PERMUTATION_TOL,
        "n_rows": n,
        "n_perms": n,
        "flipped": flips,
        "flip_rate": round(flips / n, 6) if n else None,
        "pass": n > 0 and worst <= CE.PERMUTATION_TOL,
        "how": "every row is asked again with the candidates in reverse "
               "order; the weights come back keyed by id and are realigned "
               "to the original order, so `max_abs_prob_delta` is the "
               "largest movement of any candidate's weight and `flipped` "
               "counts the rows whose argmax id changed",
        "sample": f"all {n} rows of the cut, one reversal each",
        "caveat": f"the API rounds each weight to {API_PROB_DECIMALS} "
                  "decimals, so a delta at 1e-4 is rounding and a delta "
                  "well above it is order sensitivity; the tolerance is the "
                  "suite's and is not moved",
        "source": f"{TASK} — measured in this run, same checkpoints, same "
                  "protocol, same rows",
        "pairs": sorted(pairs, key=lambda p: -p["max_abs_delta"])[:8],
    }


# ------------------------------------------------ los informes

def report_for(rows: list, *, episodes: list, model_version: str,
               permutation: dict, subset: str = "whole cut",
               notes: list | None = None) -> dict:
    """La suite sobre estas filas y nada calculado aquí."""
    tracking = P.tracking_control(
        episodes, rows,
        source=f"{TASK} on {P.CUT_NAME} {BD.battery_sha(episodes)[:12]} "
               f"({subset})")
    doc = MS.report(
        rows,
        cut={**P.cut_descriptor(episodes), "subset": subset},
        model_version=model_version,
        task=TASK,
        calibration=P.dev_temperature(rows),
        permutation=permutation,
        tracking=tracking,
        notes=list(notes or []) + [
            f"reference: {COLUMN} — external upper bound, untuned",
            "protocol: " + protocol()["candidates"],
            "weights: " + protocol()["weights"],
            "abstention: " + protocol()["abstains"],
            "the sealed cut was not read: " + ", ".join(P.SOURCES[:2]),
        ])
    return MS.require(doc)


def by_lang_reports(episodes: list, rows: list, *, model_version: str,
                    permutation: dict) -> dict:
    """La misma suite, por idioma: el subconjunto de filas de cada `lang`."""
    lang_of = {ep["id"]: ep["lang"] for ep in episodes}
    out = {}
    for lang in sorted({ep["lang"] for ep in episodes}):
        eps = [ep for ep in episodes if ep["lang"] == lang]
        sub = [r for r in rows if lang_of[r["row_id"]] == lang]
        out[lang] = report_for(sub, episodes=eps,
                               model_version=model_version,
                               permutation=permutation,
                               subset=f"lang={lang}",
                               notes=[f"subset: rows whose lang is {lang}, "
                                      f"answered by {ROUTE.get(lang)}"])
    return out


# ------------------------------------------------ mismas filas que el preflight

def preflight_columns(episodes: list) -> dict:
    """Las columnas del preflight medidas SOBRE ESTAS FILAS, por su sello."""
    out = {}
    for key in P.column_keys((P.REF_QWEN, P.REF_NLI, P.REF_POINTER)):
        doc = P.load_measured_column(key, episodes)
        if doc is not None:
            out[key] = doc
    return out


def same_rows_as_preflight(episodes: list, laya_rows: list) -> dict:
    """`same_rows()` sobre Laya + cada columna del preflight, o levanta."""
    ours = preflight_columns(episodes)
    columns = {COLUMN: laya_rows}
    columns.update({k: v["rows"] for k, v in ours.items()})
    agreement = P.same_rows(columns)
    agreement["preflight_columns"] = sorted(ours)
    agreement["preflight_gate_rows_sha256"] = _preflight_gate_sha()
    agreement["matches_preflight_gate"] = (
        agreement["rows_sha256"] == agreement["preflight_gate_rows_sha256"])
    return agreement


def _preflight_gate_sha():
    try:
        doc = json.loads(P.GATE_PATH.read_text())
    except (OSError, ValueError):
        return None
    return ((doc.get("checks") or {}).get("same_rows_over_the_three_columns")
            or {}).get("rows_sha256")


# ------------------------------------------------ la comparación

def _fig(node: dict) -> dict:
    return P._fig(node)


def _clears(node: dict) -> bool:
    return P._clears(node)


def outcome(laya: dict, ours: dict) -> str:
    """`VERDICT_RULE` sobre dos cifras de la suite."""
    a = (laya or {}).get("accuracy_ci95") or [None, None]
    b = (ours or {}).get("accuracy_ci95") or [None, None]
    if None in a or None in b:
        return "unmeasured"
    if not _clears(laya) and not _clears(ours):
        return "fails_alike"
    if a[0] > b[1]:
        return "wins"
    if a[1] < b[0]:
        return "loses"
    return "ties"


def paired_counts(laya_rows: list, other_rows: list) -> dict:
    """Cuántas filas acierta cada uno y no el otro: contado, no estimado."""
    by_id = {r["row_id"]: r for r in other_rows}
    both = laya_only = other_only = neither = 0
    for r in laya_rows:
        o = by_id[r["row_id"]]
        a = P._forced(r) == r["gold_index"]
        b = P._forced(o) == o["gold_index"]
        both += a and b
        laya_only += a and not b
        other_only += b and not a
        neither += not a and not b
    return {"both_right": both, "laya_only": laya_only,
            "ours_only": other_only, "neither": neither,
            "what": "forced pick per row, counted; a McNemar test is not "
                    "published because the suite has none"}


def comparison(laya_report: dict, laya_rows: list, episodes: list) -> dict:
    """Laya al lado de cada columna del preflight, celda a celda copiada."""
    reports = P.measured_reports()
    columns = preflight_columns(episodes)
    out = {}
    for key in sorted(reports):
        doc = reports[key]
        cell = {
            "model_version": doc.get("model_version"),
            "report": f"artifacts/gates/{P.TASK}/refs-"
                      f"{key.replace(':', '-')}.json",
            "ranking": {"laya": _fig(laya_report.get("ranking") or {}),
                        "ours": _fig(doc.get("ranking") or {})},
            "counterfactual": {
                "laya": _fig(laya_report.get("counterfactual") or {}),
                "ours": _fig(doc.get("counterfactual") or {})},
            "macro_ranking": {
                "laya": (laya_report.get("macro") or {}).get(
                    "accuracy_ranking"),
                "ours": (doc.get("macro") or {}).get("accuracy_ranking")},
            "by_family": {
                fam: {"laya": _fig((laya_report.get("by_family") or {})
                                   .get(fam, {}).get("ranking") or {}),
                      "ours": _fig(node.get("ranking") or {}),
                      "outcome": outcome(
                          (laya_report.get("by_family") or {})
                          .get(fam, {}).get("ranking") or {},
                          node.get("ranking") or {})}
                for fam, node in sorted((doc.get("by_family") or {}).items())
            },
            "outcome": {
                "ranking": outcome(laya_report.get("ranking") or {},
                                   doc.get("ranking") or {}),
                "counterfactual": outcome(
                    laya_report.get("counterfactual") or {},
                    doc.get("counterfactual") or {}),
                "rule": VERDICT_RULE,
            },
        }
        if key in columns:
            cell["paired"] = paired_counts(laya_rows, columns[key]["rows"])
        out[key] = cell
    return out


def verdict_line(comp: dict, laya_report: dict, fullspace: dict | None) -> str:
    """Una línea: dónde nos gana, dónde empata, dónde falla igual."""
    def names(field: str, want: str) -> str:
        got = sorted(k for k, c in comp.items()
                     if c["outcome"][field] == want)
        return ", ".join(got) if got else "none"

    rk = laya_report.get("ranking") or {}
    cf = laya_report.get("counterfactual") or {}
    parts = [
        f"Laya untuned, forced {rk.get('accuracy')} CI {rk.get('accuracy_ci95')}"
        f" (chance {rk.get('chance')}), joint counterfactual "
        f"{cf.get('hits')}/{cf.get('n')} = {cf.get('accuracy')} CI "
        f"{cf.get('accuracy_ci95')} (chance {cf.get('chance')})",
        f"wins on forced vs [{names('ranking', 'wins')}]",
        f"ties vs [{names('ranking', 'ties')}]",
        f"loses vs [{names('ranking', 'loses')}]",
        f"fails alike (neither clears chance) vs "
        f"[{names('ranking', 'fails_alike')}]",
        f"counterfactual: wins vs [{names('counterfactual', 'wins')}], "
        f"ties vs [{names('counterfactual', 'ties')}], loses vs "
        f"[{names('counterfactual', 'loses')}], fails alike vs "
        f"[{names('counterfactual', 'fails_alike')}]",
    ]
    if fullspace:
        ours = ((fullspace.get("table") or {}).get("banking77") or {}).get(
            "ours") or {}
        pub = PUBLISHED["banking77_k77"]["accuracy"]
        acc = ours.get("accuracy")
        ci = ours.get("accuracy_ci95") or [None, None]
        where = ("unmeasured" if acc is None else
                 "reproduces (0.425 inside the interval)"
                 if ci[0] is not None and ci[0] <= pub <= ci[1] else
                 "does NOT reproduce (0.425 outside the interval)")
        parts.append(f"BANKING77 K=77 in our harness: {acc} CI {ci} "
                     f"(chance {ours.get('chance')}) — published 0.425 "
                     f"{where}; our pointer head measured 0.0123 there")
    return "; ".join(parts)


# ------------------------------------------------ una columna en disco

def column_path() -> Path:
    return COLUMN_DIR / f"{COLUMN}.json"


def load_measured_column(episodes: list) -> dict | None:
    path = column_path()
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    if doc.get("rows_sha256") != P.rows_sha(episodes):
        return None
    if len(doc.get("rows") or []) != len(episodes):
        return None
    return doc


def save_measured_column(episodes: list, rows: list, trace: dict, *,
                         model_version: str, permutation: dict,
                         checkpoints: dict, package: dict,
                         command: str) -> dict:
    doc = {
        "column": COLUMN,
        "task": TASK,
        "cut": P.CUT_NAME,
        "rows_sha256": P.rows_sha(episodes),
        "measured_utc": utcnow(),
        "model_version": model_version,
        "package": package,
        "checkpoints": checkpoints,
        "command": command,
        "permutation": permutation,
        "trace": trace,
        "rows": rows,
        "what": "the MEASURED column: one forward pass per row, kept so "
                "the table and the gate are rebuilt without a model. "
                "Reused only when `rows_sha256` matches the cut",
    }
    COLUMN_DIR.mkdir(parents=True, exist_ok=True)
    column_path().write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return doc


# ------------------------------------------------ el gate

def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def gate(write: bool = True, episodes: list | None = None,
         tests: dict | None = None) -> dict:
    """El gate: lo medido con cifras copiadas, lo pendiente en `null`."""
    episodes = episodes if episodes is not None else P.load_cut()
    column = load_measured_column(episodes)
    report = _read_json(REPORT_PATH)
    if report is not None and report.get("format") != MS.FORMAT:
        report = None
    fullspace = _read_json(FULLSPACE_PATH)
    by_lang = {}
    for lang in sorted({ep["lang"] for ep in episodes}):
        doc = _read_json(GATE_DIR / f"refs-laya-lang-{lang}.json")
        if doc is not None and doc.get("format") == MS.FORMAT:
            by_lang[lang] = doc

    same = None
    if column is not None:
        same = same_rows_as_preflight(episodes, column["rows"])
    comp = comparison(report, column["rows"], episodes) \
        if report and column else {}

    checks = {
        "counterfactual_pairs": (
            {"pass": True, "measured": True,
             **_fig(report["counterfactual"]),
             "groups_with_one_half_only":
                 report["counterfactual"].get("groups_with_one_half_only"),
             "tracking": {k: report["tracking"].get(k) for k in
                          ("n", "followed", "rate", "same_slot",
                           "same_slot_rate")},
             "beside_ours": {k: c["counterfactual"]["ours"]
                             for k, c in comp.items()},
             "outcome_vs_ours": {k: c["outcome"]["counterfactual"]
                                 for k, c in comp.items()},
             "read_from": _rel(REPORT_PATH) + " /counterfactual"}
            if report else
            {"pass": None, "reason": "awaiting-compute",
             "how": "PYTHONPATH=. .venv-laya/bin/python -m "
                    "eval.laya_baseline battery --device cpu"}),
        "battery_dev_forced": (
            {"pass": True, "measured": True,
             **_fig(report["ranking"]),
             "macro_by_family": {
                 "accuracy_ranking": report["macro"].get("accuracy_ranking"),
                 "chance_ranking": report["macro"].get("chance_ranking"),
                 "worst_family": report["macro"].get("worst_family"),
                 "ci95_why_absent": report["macro"].get("ci95_why_absent")},
             "by_family": {fam: _fig(node.get("ranking") or {})
                           for fam, node in sorted(
                               report["by_family"].items())},
             "by_lang": {lang: {"ranking": _fig(doc["ranking"]),
                                "macro_ranking":
                                    doc["macro"].get("accuracy_ranking"),
                                "counterfactual":
                                    _fig(doc["counterfactual"]),
                                "answered_by": ROUTE.get(lang),
                                "report": _rel(GATE_DIR /
                                               f"refs-laya-lang-{lang}.json")}
                         for lang, doc in sorted(by_lang.items())},
             "permutation_control": P._perm_cell(report),
             "rows_by_model": (column.get("trace") or {}).get(
                 "rows_by_model"),
             "read_from": _rel(REPORT_PATH)}
            if report else
            {"pass": None, "reason": "awaiting-compute"}),
        "same_rows_as_preflight": (
            {"pass": bool(same["pass"] and same["matches_preflight_gate"]),
             **same,
             "how_measured": "eval.preflight_refs.same_rows() over the Laya "
                             "column and every preflight column measured "
                             "on these rows, compared by identity; the "
                             "sha is then matched against the one the "
                             "preflight gate signed"}
            if same else
            {"pass": None, "reason": "awaiting-compute"}),
        "banking77_k77": (
            {"pass": True, "measured": True,
             "ours_harness": (fullspace.get("table") or {}).get(
                 "banking77", {}).get("ours"),
             "published": PUBLISHED["banking77_k77"],
             "pointer_head_for_scale": {
                 "citation": True,
                 "who": "our pointer head (#T-fullspace-objective)",
                 "accuracy": 0.0123, "chance": 0.013,
                 "source": "artifacts/gates/T-fullspace-objective/"},
             "checkpoint": fullspace.get("checkpoint"),
             "model_version": fullspace.get("model_version"),
             "cardinality_sweep": fullspace.get("cardinality_sweep"),
             "cardinality_reading": fullspace.get("cardinality_reading"),
             "test_cut_query_logged": fullspace.get(
                 "test_cut_query_logged"),
             "read_from": _rel(FULLSPACE_PATH)}
            if fullspace and (fullspace.get("table") or {}).get("banking77")
            else
            {"pass": None, "reason": "awaiting-compute",
             "how": "PYTHONPATH=. .venv-laya/bin/python -m "
                    "eval.laya_baseline banking77 --device cpu"}),
        "mix_declared_per_dataset": {
            "pass": True,
            "needs_compute": False,
            "datasets": MIX_DECLARATION,
            "measured_here": ["battery-dev", "banking77"],
            "how_measured": "read off the cited files of the Laya clone; "
                            "the mix itself is not published row by row",
        },
    }
    verdicts = [c.get("pass") for c in checks.values()]
    pending = any(v is None for v in verdicts)
    doc = {
        "task": TASK,
        "artifact": _rel(GATE_PATH),
        "generated_utc": utcnow(),
        "question": "does an untuned public checkpoint of the same product "
                    "class answer our battery, and where exactly does it "
                    "beat, tie or fail like our columns?",
        "trains_nothing": True,
        "tunes_nothing": True,
        "publishes_no_figure_of_its_own": (
            f"every figure here is copied from a `{MS.FORMAT}` report in "
            "this directory or from eval.fullspace's gate; nothing is "
            "recomputed here"),
        "runner": "eval/laya_baseline.py",
        "package": (column or {}).get("package"),
        "checkpoints": (column or {}).get("checkpoints") or CHECKPOINTS,
        "commands": {
            "battery": (column or {}).get("command"),
            "banking77": (fullspace or {}).get("command"),
            "gate": "PYTHONPATH=. .venv-train/bin/python -m "
                    "eval.laya_baseline gate",
        },
        "cut": P.cut_descriptor(episodes),
        "protocol": protocol(),
        "verdict_rule": VERDICT_RULE,
        "comparison": comp,
        "checks": checks,
    }
    doc["verdict"] = ("AWAITING-COMPUTE" if pending
                      else "PASS" if all(verdicts) else "FAIL")
    doc["pass"] = None if pending else all(verdicts)
    doc["verdict_line"] = (verdict_line(comp, report, fullspace)
                           if report else None)
    doc["honesty"] = [
        "no figure in this file was computed by hand: what was not "
        "measured is `null`, and every number is copied from a suite "
        "report this gate points at",
        "Laya is untuned: no weight, temperature or threshold was touched, "
        "and the option budget is the checkpoint's default",
        "the checkpoint per row is chosen by the row's declared language, "
        "which is what Laya's Router would do by script detection",
        "BANKING77 is the official test split through eval.fullspace and "
        "its read is in the reserved-cut ledger; Laya's own 0.425 is on "
        "400 rows of its own harness and is quoted as a citation",
        "the permutation control is measured on every row with the API's "
        "4-decimal weights: a delta at the rounding scale is not order "
        "sensitivity, and the suite's tolerance is not moved to fit",
    ]
    if tests:
        doc["tests"] = tests
    if write:
        GATE_DIR.mkdir(parents=True, exist_ok=True)
        GATE_PATH.write_text(json.dumps(doc, indent=2, ensure_ascii=False,
                                        sort_keys=True) + "\n")
    return doc


def build_reports(episodes: list, column: dict, write: bool = True) -> dict:
    """Los informes (entero y por idioma) de una columna ya medida."""
    same_rows_as_preflight(episodes, column["rows"])
    whole = report_for(column["rows"], episodes=episodes,
                       model_version=column["model_version"],
                       permutation=column["permutation"])
    whole["column"] = COLUMN
    langs = by_lang_reports(episodes, column["rows"],
                            model_version=column["model_version"],
                            permutation=column["permutation"])
    for lang, doc in langs.items():
        doc["column"] = f"{COLUMN}:lang={lang}"
    if write:
        GATE_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(whole, indent=2,
                                          ensure_ascii=False) + "\n")
        for lang, doc in langs.items():
            (GATE_DIR / f"refs-laya-lang-{lang}.json").write_text(
                json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return {"whole": whole, "by_lang": langs}


# ------------------------------------------------ la inferencia real

def sha256_file(path) -> str:
    return P.sha256_file(path)


class LayaPredictor:
    """Los checkpoints reales, cargados una vez, en CPU. Sólo el job."""

    def __init__(self, device: str = "cpu", log=print):
        import laya  # the installed package, never the clone
        self._laya = laya
        self.device = device
        self.log = log
        self.agents: dict = {}
        self.package = {"name": PACKAGE,
                        "version": getattr(laya, "__version__", None),
                        "interpreter": sys.executable}
        self.checkpoints: dict = {}

    def agent(self, name: str):
        if name in self.agents:
            return self.agents[name]
        from huggingface_hub import snapshot_download
        spec = CHECKPOINTS[name]
        started = time.perf_counter()
        local = snapshot_download(spec["repo"], revision=spec["revision"],
                                  allow_patterns=list(SNAPSHOT_PATTERNS))
        agent = self._laya.load(local, device=self.device)
        weights = Path(local) / "model.safetensors"
        self.checkpoints[name] = {
            **spec,
            "local_dir": str(local),
            "weights_sha256": sha256_file(weights) if weights.exists()
            else None,
            "cfg": {k: agent.cfg.get(k) for k in
                    ("max_len", "head_max_len", "encoder", "encoder_name",
                     "model_type", "temperature")
                    if k in agent.cfg},
            "device": str(agent.device),
            "load_seconds": round(time.perf_counter() - started, 1),
        }
        self.log(f"[laya] {name}: loaded {spec['repo']}@{spec['revision'][:12]}"
                 f" on {agent.device} in "
                 f"{self.checkpoints[name]['load_seconds']} s")
        self.agents[name] = agent
        return agent

    def model_version(self) -> str:
        return " + ".join(
            f"{CHECKPOINTS[n]['repo']}@{CHECKPOINTS[n]['revision'][:12]}"
            for n in sorted(CHECKPOINTS))

    def predict_with(self, name: str, state: str, question: str,
                     candidates: list) -> dict:
        agent = self.agent(name)
        ids = [c["id"] for c in candidates]
        started = time.perf_counter()
        result = agent.predict(state, question_of(question, candidates))
        ms = round(1000 * (time.perf_counter() - started), 1)
        answer = (result.get("answers") or {}).get("q") or {}
        probs = align(answer.get("probabilities") or {}, ids)
        return {"probs": probs, "model": name, "ms": ms}

    def __call__(self, state: str, question: str, candidates: list,
                 lang: str) -> dict:
        name = ROUTE.get(lang)
        if name is None:
            raise ProtocolMismatch(
                f"no checkpoint routed for lang {lang!r}: ROUTE is written "
                "before measuring and is not extended on the fly")
        return self.predict_with(name, state, question, candidates)


class LayaEngine:
    """Laya como scorer externo de `eval.fullspace` (K = el espacio entero).

    `entries()` devuelve la misma forma que `eval.calib.entries_from_samples`:
    `probs` sobre `[K + 1]` con `unknown` a 0 (la API no abstiene), `pred`
    el argmax sobre los K, `label` el gold. `tally()` no cambia.
    """

    def __init__(self, predictor: LayaPredictor, name: str = "laya",
                 log=print):
        self.predictor = predictor
        self.name = name
        self.device = predictor.device
        self.log = log

    def entries(self, samples: list, split: str, cut: str, dataset: str,
                batch_size: int = 16) -> list:
        out = []
        started = time.perf_counter()
        for i, sample in enumerate(samples):
            got = self.predictor.predict_with(self.name, sample.state,
                                              sample.question,
                                              sample.options)
            probs = list(got["probs"]) + [0.0]
            logits = [math.log(max(p, 1e-12)) for p in got["probs"]] + [-30.0]
            k = len(sample.options)
            pred = max(range(k), key=lambda j: probs[j])
            out.append({
                "id": f"{sample.dataset}:{sample.row_id}:"
                      f"{sample.question_id}",
                "split": split, "cut": cut,
                "dataset": dataset or sample.dataset,
                "cardinality": k,
                "logits": logits, "probs": probs,
                "label": sample.gold_index, "pred": pred,
                "locale": "",
                "ms": got["ms"],
            })
            if (i + 1) % 100 == 0:
                rate = (time.perf_counter() - started) / (i + 1)
                self.log(f"[laya] {i + 1}/{len(samples)} rows, "
                         f"{rate * 1000:.0f} ms/row, ETA "
                         f"{rate * (len(samples) - i - 1) / 60:.1f} min")
        return out


def run_battery(device: str = "cpu", resume: bool = True,
                write: bool = True, log=print) -> dict:
    """LA COLUMNA, de verdad: 400 filas + 400 al revés, y los informes."""
    episodes = P.load_cut()
    command = ("PYTHONPATH=. .venv-laya/bin/python -m eval.laya_baseline "
               f"battery --device {device}")
    column = load_measured_column(episodes) if resume else None
    if column is None:
        predictor = LayaPredictor(device=device, log=log)
        COLUMN_DIR.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        rows, trace = laya_column(
            episodes, predictor,
            log=COLUMN_DIR / f"{COLUMN}.picks.jsonl" if resume else None)
        log(f"[laya] column: {len(rows)} rows in "
            f"{time.perf_counter() - started:.0f} s "
            f"({trace['ms_per_row_mean']} ms/row)")
        perm = permutation_control(
            episodes, rows, predictor,
            log=COLUMN_DIR / f"{COLUMN}.reversed.jsonl" if resume else None)
        log(f"[laya] permutation: max_abs_delta {perm['max_abs_prob_delta']}"
            f", flipped {perm['flipped']}/{perm['n_rows']}")
        trace["device"] = device
        column = save_measured_column(
            episodes, rows, trace, model_version=predictor.model_version(),
            permutation=perm, checkpoints=predictor.checkpoints,
            package=predictor.package, command=command)
    else:
        log("[laya] column reused from disk (rows_sha256 matches)")
    built = build_reports(episodes, column, write=write)
    doc = gate(write=write, episodes=episodes)
    rk = built["whole"]["ranking"]
    cf = built["whole"]["counterfactual"]
    log(f"[laya] forced {rk['accuracy']} CI {rk['accuracy_ci95']} "
        f"(chance {rk['chance']}); joint {cf['hits']}/{cf['n']} = "
        f"{cf['accuracy']} CI {cf['accuracy_ci95']} (chance {cf['chance']})")
    return {"column": {k: column[k] for k in ("model_version", "rows_sha256",
                                             "package", "checkpoints")},
            "ranking": rk, "counterfactual": cf,
            "verdict": doc["verdict"], "verdict_line": doc["verdict_line"]}


def run_banking77(device: str = "cpu", limit: int | None = None,
                  sweep: bool = False, reason: str = "", write: bool = True,
                  log=print) -> dict:
    """BANKING77 a K=77 por `eval.fullspace`, con Laya como scorer externo."""
    from eval import fullspace as F

    predictor = LayaPredictor(device=device, log=log)
    engine = LayaEngine(predictor, "laya", log=log)
    predictor.agent("laya")
    spec = predictor.checkpoints["laya"]
    label = f"laya:{spec['repo']}@{spec['revision']}"
    command = ("PYTHONPATH=. .venv-laya/bin/python -m eval.laya_baseline "
               f"banking77 --device {device}"
               + (f" --limit {limit}" if limit else "")
               + ("" if sweep else " --no-sweep"))
    manifest = {"model_version": label, "run_id": f"{TASK}-banking77",
                "package": predictor.package, "checkpoint": spec,
                "protocol": protocol(), "command": command}
    doc = F.run(label, ("banking77",), limit or F.DEFAULT_ROWS, device,
                write=write, sweep=sweep, log=log, engine=engine,
                manifest=manifest, gate_path=FULLSPACE_PATH, parity=False,
                reason=reason or f"{TASK}: Laya (untuned, {label}) on the "
                                 "full label space, to see whether its "
                                 "published 0.425 reproduces in our "
                                 "harness")
    doc["package"] = predictor.package
    doc["checkpoint_detail"] = spec
    doc["command"] = command
    doc["published"] = PUBLISHED["banking77_k77"]
    doc["mix"] = MIX_DECLARATION["banking77"]
    if write:
        FULLSPACE_PATH.write_text(json.dumps(doc, indent=2,
                                             ensure_ascii=False) + "\n")
        gate(write=True)
    ours = doc["table"]["banking77"]["ours"]
    log(f"[laya] banking77 K={ours['cardinality']}: {ours['accuracy']} CI "
        f"{ours['accuracy_ci95']} (chance {ours['chance']}, n {ours['n']})")
    return doc


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(prog="eval.laya_baseline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("protocol", help="the fixed input format")
    g = sub.add_parser("gate", help="the gate from what is measured on disk")
    g.add_argument("--no-write", action="store_true")
    b = sub.add_parser("battery", help="THE JOB: the 400 rows + reversed")
    b.add_argument("--device", default="cpu")
    b.add_argument("--no-resume", action="store_true")
    k = sub.add_parser("banking77", help="THE JOB: K=77 via eval.fullspace")
    k.add_argument("--device", default="cpu")
    k.add_argument("--limit", type=int, default=None)
    k.add_argument("--no-sweep", action="store_true")
    k.add_argument("--reason", default="")
    args = ap.parse_args(argv)

    if args.cmd == "protocol":
        print(json.dumps(protocol(), indent=2, ensure_ascii=False))
        return 0
    if args.cmd == "gate":
        doc = gate(write=not args.no_write)
        print(json.dumps({"task": TASK, "verdict": doc["verdict"],
                          "verdict_line": doc["verdict_line"],
                          "awaiting_compute": sorted(
                              k for k, v in doc["checks"].items()
                              if v.get("pass") is None)}, indent=2,
                         ensure_ascii=False))
        return 0
    if args.cmd == "battery":
        out = run_battery(device=args.device, resume=not args.no_resume)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0
    out = run_banking77(device=args.device, limit=args.limit,
                        sweep=not args.no_sweep, reason=args.reason)
    print(json.dumps(out["table"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
