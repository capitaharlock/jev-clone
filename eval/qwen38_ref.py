"""Qwen 3.8 sobre la batería de desarrollo — el profesor nuevo, MEDIDO antes
de que genere (#T-qwen38-ref).

El operador cambió el profesor local a `qwen3.8:27b-mlx`
(`data.episode_gen.MODEL`). Todo el volumen del bucle —los episodios y su
verificación— lo produce ese modelo, así que su capacidad es el techo de
lo que el bucle puede enseñar. Del 3.6 hay una referencia medida sobre las
400 filas de desarrollo: forzada 0,965
(`artifacts/gates/T-preflight-refs/refs-qwen-local.json`). Este módulo mide
la MISMA columna con el modelo nuevo y firma la comparación.

Qué hace honesto a esto
-----------------------

* **Mismas filas, mismo protocolo.** La columna se mide con
  `eval.preflight_refs.qwen_column` —elección estructurada entre ids
  válidos, evidencia que tiene que ser un tramo del estado— sobre
  `eval.preflight_refs.load_cut()`. El sello de filas
  (`rows_sha256`) de las dos columnas se compara antes de publicar nada y
  `same_rows()` las compara fila a fila: dos columnas de n igual pero
  filas distintas no son una comparación.
* **La columna del 3.6 NO se toca.** Se lee de `#T-preflight-refs` y se
  reutiliza; lo medido aquí se escribe en `artifacts/gates/T-qwen38-ref/`.
* **Cero aritmética a mano.** Las accuracies salen de
  `eval.metrics_suite.report()` (vía `preflight_refs.report_for`, firmado a
  nombre de esta task) y la diferencia sale del bootstrap PAREADO de
  `training.python.ce_finetune._paired_bootstrap`, remuestreando las mismas
  unidades. Este módulo no calcula una media por su cuenta.
* **La regla está escrita antes de medir** (`GATE_RULE`, abajo): el gate
  no elige su umbral después de ver la cifra.
* **El control de permutación, a las dos temperaturas.** La columna se
  mide a la temperatura del generador (0,7, la misma a la que se midió el
  3.6, que falló el control con `flip_rate` 0,125) y el control se repite
  en greedy (temperatura 0, la del verificador). A 0,7 un cambio de
  elección es orden O muestreo; a 0 el muestreo no interviene. Las dos
  cifras van al gate.
* **El sellado no se lee.** `load_cut()` rechaza cualquier ruta del corte
  sellado y cualquier fila cuyo `variant_group` no empiece por `dev-`.

Lo que NO hace: entrenar, generar episodios, ni cambiar el modelo del
productor. Escribe la DECISIÓN de qué `QWEN_MODEL` debe usar `data.stream`
y el bucle la obedece.

CLI (stdlib, cualquier python3, desde la raíz del repo):

    PYTHONPATH=. python3 -m eval.qwen38_ref plan    # la regla, sin medir
    PYTHONPATH=. python3 -m eval.qwen38_ref gate    # gate sobre lo medido
    # EL JOB (`evalgate`), ~15 min con OLLAMA_NUM_PARALLEL=4:
    QWEN_MODEL=qwen3.8:27b-mlx OLLAMA_NUM_PARALLEL=4 \
        PYTHONPATH=. python3 -m eval.qwen38_ref measure
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval import metrics_suite as MS
from eval import preflight_refs as P
from training.python import ce_finetune as FT

TASK = "T-qwen38-ref"
ROOT = P.ROOT
GATE_DIR = ROOT / "artifacts" / "gates" / TASK
COLUMN_DIR = GATE_DIR / "columns"
GATE_PATH = GATE_DIR / "gate.json"

#: La columna nueva y su informe. El nombre lo fija la task.
COLUMN_KEY = "qwen-local-3.8"
REPORT_PATH = GATE_DIR / "refs-qwen-local-3.8.json"

#: Lo medido del 3.6, que se reutiliza y no se reescribe.
BASELINE_REPORT = P.GATE_DIR / "refs-qwen-local.json"
BASELINE_COLUMN = P.COLUMN_DIR / "qwen-local.json"

#: La temperatura del verificador: el control de permutación se repite
#: aquí porque es la temperatura a la que el profesor DECIDE si un
#: episodio se publica.
PERM_TEMPERATURE = 0.0

BOOTSTRAP_B = FT.BOOTSTRAP_B
BOOTSTRAP_SEED = FT.BOOTSTRAP_SEED

#: La regla, escrita el 2026-09-27 ANTES de medir la columna del 3.8.
GATE_RULE = {
    "cut": "the 400 development rows of #T-battery-dev, by "
           "`eval.preflight_refs.load_cut()`; the sealed cut is not read",
    "columns": "qwen-local (qwen3.6, reused from #T-preflight-refs) vs "
               "qwen-local-3.8 (measured here) — same rows, same "
               "structured-choice protocol, same `rows_sha256`",
    "difference": "paired bootstrap of (3.8 − 3.6) over the same rows, "
                  f"B={BOOTSTRAP_B}, seed {BOOTSTRAP_SEED}, percentile "
                  "CI95; the counterfactual difference is paired by "
                  "variant_group",
    "go_if": "the forced accuracy of 3.8 does not sit below the 3.6 "
             "figure: the CI95 of the paired difference has an upper "
             "bound >= 0",
    "no_go_if": "the CI95 of the paired difference lies entirely below 0 "
                "— the new teacher is measurably worse, and the producer "
                "(`data.stream`) goes back to QWEN_MODEL=qwen3.6:27b-mlx "
                "until the operator decides",
    "better_is_not_required": "GO means `not worse`. Whether 3.8 is "
                              "BETTER is published apart "
                              "(`forced.ci95[0] > 0`) and changes no "
                              "decision here",
    "families_are_informative": "a family that falls is published with "
                                "its delta but does not by itself flip "
                                "the verdict: n per family is 80 and this "
                                "gate decides which teacher generates, "
                                "not which checkpoint is promoted",
    "permutation": "measured at the generator's temperature (comparable "
                   "with the 3.6 column) and repeated at temperature 0 "
                   "(the verifier's); a failed control bounds the "
                   "column's figure and is published, never hidden",
}

#: El modelo al que vuelve el productor si el 3.8 sale peor.
FALLBACK_MODEL = "qwen3.6:27b-mlx"


class Missing(P.ProtocolMismatch):
    """Falta una columna medida: el gate no la inventa."""


# ------------------------------------------------- el profesor, inyectable

def chooser(temperature: float):
    """`choose(state, question, candidates)` a UNA temperatura fijada.

    `preflight_refs` llama al chooser con tres argumentos, así que la
    temperatura viaja cerrada en el closure y no en la llamada: el
    protocolo de la columna no cambia por medir dos controles.
    """
    from data import episode_gen as EG

    def choose(state: str, question: str, candidates: list) -> dict:
        return EG.teacher_structured(state, question, candidates,
                                     temperature=temperature)

    return choose


def qwen_model_id() -> str:
    from data import episode_gen as EG
    return EG.MODEL


def generator_temperature() -> float:
    from data import episode_gen as EG
    return EG.TEMPERATURE


# --------------------------------------------- lo del 3.6, leído de disco

def baseline_report() -> dict:
    if not BASELINE_REPORT.exists():
        raise Missing(f"{P._rel(BASELINE_REPORT)} is not on disk: the 3.6 "
                      "reference is what this task compares against")
    doc = json.loads(BASELINE_REPORT.read_text())
    if doc.get("format") != MS.FORMAT:
        raise Missing(f"{P._rel(BASELINE_REPORT)} is not a {MS.FORMAT} "
                      "document")
    return doc


def baseline_column() -> dict:
    if not BASELINE_COLUMN.exists():
        raise Missing(f"{P._rel(BASELINE_COLUMN)} is not on disk: the "
                      "paired difference needs the 3.6 rows, not its "
                      "summary")
    return json.loads(BASELINE_COLUMN.read_text())


# ----------------------------------------- la columna nueva, medida y sellada

def column_path() -> Path:
    return COLUMN_DIR / f"{COLUMN_KEY}.json"


def load_column(episodes: list) -> dict | None:
    """La columna medida SOBRE ESTAS FILAS, o `None`."""
    path = column_path()
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    if doc.get("rows_sha256") != P.rows_sha(episodes):
        return None
    if len(doc.get("rows") or []) != len(episodes):
        return None
    return doc


def measure(episodes: list, choose=None, perm_choose=None,
            workers: int = P.QWEN_WORKERS) -> dict:
    """La columna del profesor ACTUAL + sus dos controles de permutación.

    `choose` / `perm_choose` son inyectables: los tests pasan funciones
    deterministas y no levantan un modelo. Por defecto salen de
    `data.episode_gen.teacher_structured` a la temperatura del generador y
    a 0.
    """
    model = qwen_model_id()
    base_mv = baseline_report().get("model_version") or ""
    if base_mv.endswith(model):
        raise P.ProtocolMismatch(
            f"QWEN_MODEL={model} is the model already measured in "
            f"{P._rel(BASELINE_REPORT)} ({base_mv}): this task measures a "
            "DIFFERENT teacher on the same rows, and re-measuring the "
            "same one under another name would be two columns of the same "
            "thing")
    gen_temp = generator_temperature()
    COLUMN_DIR.mkdir(parents=True, exist_ok=True)
    rows, trace = P.qwen_column(
        episodes, choose=choose or chooser(gen_temp),
        log=COLUMN_DIR / f"{COLUMN_KEY}.picks.jsonl", workers=workers)
    picks = {t["row_id"]: t.get("choice") for t in trace["traces"]}
    perm = P.qwen_permutation(
        episodes, picks, choose=choose or chooser(gen_temp),
        log=COLUMN_DIR / f"{COLUMN_KEY}.reversed.jsonl",
        temperature=gen_temp)
    perm_zero = P.qwen_permutation(
        episodes, picks, choose=perm_choose or chooser(PERM_TEMPERATURE),
        log=COLUMN_DIR / f"{COLUMN_KEY}.reversed-t0.jsonl",
        temperature=PERM_TEMPERATURE)
    trace["permutation_sample"] = perm["n_rows"]
    trace["temperature"] = gen_temp
    trace["permutation_temperatures"] = [gen_temp, PERM_TEMPERATURE]
    doc = {
        "column": COLUMN_KEY,
        "task": TASK,
        "cut": P.CUT_NAME,
        "rows_sha256": P.rows_sha(episodes),
        "measured_utc": P.utcnow(),
        "model_version": f"qwen-local:{model}",
        "permutation": perm,
        "permutation_temperature_0": perm_zero,
        "trace": trace,
        "rows": rows,
        "what": "the MEASURED column of the NEW teacher: one structured "
                "choice per row at the generator's temperature, plus the "
                "permutation control at that temperature and at 0",
    }
    column_path().write_text(json.dumps(doc, indent=2, ensure_ascii=False)
                             + "\n")
    return doc


def publish(episodes: list, column: dict, write: bool = True) -> dict:
    """El informe de la columna nueva, por la suite y firmado por esta task."""
    doc = P.report_for(
        P.REF_QWEN, column["rows"], episodes=episodes,
        model_version=column["model_version"],
        permutation=column["permutation"], task=TASK,
        notes=[
            f"teacher measured because the operator changed "
            f"`data.episode_gen.MODEL` to {qwen_model_id()}: the whole "
            "volume of the loop is generated and verified by it",
            f"the 3.6 column is reused from #T-preflight-refs "
            f"({P._rel(BASELINE_REPORT)}) and was not re-measured",
            "the permutation control at temperature 0 (the verifier's) is "
            "published in the gate and in the column document, apart from "
            "this report's control, which is at the generator's "
            "temperature so it compares with the 3.6 figure",
        ])
    doc["column"] = COLUMN_KEY
    if write:
        GATE_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(doc, indent=2, ensure_ascii=False)
                               + "\n")
    return doc


# ------------------------------------------------------- la comparación

def _with_lang(rows: list, episodes: list) -> list:
    """Las filas con su idioma, que el comparador agrupa por idioma."""
    lang = {ep["id"]: ep.get("lang") for ep in episodes}
    return [{**r, "lang": lang.get(r["row_id"])} for r in rows]


def compare(base: dict, new: dict, episodes: list | None = None,
            seed: int = BOOTSTRAP_SEED, reps: int = BOOTSTRAP_B) -> dict:
    """La diferencia PAREADA entre los dos profesores, con MI regla encima.

    La medición es la de `#T-ce-finetune` (`compare`): bootstrap pareado de
    la forzada y del contrafactual conjunto, y baldes por familia e
    idioma. El VEREDICTO es el de esta task —«no peor», no «mejor»—, así
    que se recalcula aquí y el de allí no se copia.
    """
    if base.get("rows_sha256") != new.get("rows_sha256"):
        raise P.ProtocolMismatch(
            f"the two columns carry different rows_sha256 "
            f"({base.get('rows_sha256')} vs {new.get('rows_sha256')}): not "
            "the same cut, and not a comparison")
    eps = episodes if episodes is not None else []
    b_rows = _with_lang(base["rows"], eps)
    n_rows = _with_lang(new["rows"], eps)
    raw = FT.compare({"rows": b_rows,
                      "battery": {"rows_sha256": base["rows_sha256"]}},
                     {"rows": n_rows,
                      "battery": {"rows_sha256": new["rows_sha256"]}},
                     seed=seed, reps=reps)
    forced = dict(raw["forced"])
    forced["baseline"] = forced.pop("control")
    forced["candidate"] = forced.pop("tuned")
    joint = dict(raw["counterfactual_joint"])
    joint["baseline"] = joint.pop("control")
    joint["candidate"] = joint.pop("tuned")
    hi = forced["ci95"][1]
    lo = forced["ci95"][0]
    not_worse = bool(hi is not None and hi >= 0)
    better = bool(lo is not None and lo > 0)
    verdict = "GO" if not_worse else "NO-GO"
    if raw["identical_picks"]:
        reason = ("every forced pick is identical on all rows: the same "
                  "decisions, so the new teacher is not worse")
    elif not_worse:
        reason = (f"the paired CI95 of the forced difference is "
                  f"{forced['ci95']}, whose upper bound is not below 0"
                  + (" and whose lower bound clears 0: measurably better"
                     if better else ": not measurably worse"))
    else:
        reason = (f"the paired CI95 of the forced difference is "
                  f"{forced['ci95']}, entirely below 0: the new teacher "
                  "is measurably worse on the same rows")
    families_down = [name for name, v in raw["by_family"].items()
                     if v["below_control"]]
    return {
        "same_rows": raw["same_rows"],
        "rows_sha256": raw["rows_sha256"],
        "n": raw["n"],
        "baseline_model": base.get("model_version"),
        "candidate_model": new.get("model_version"),
        "forced": forced,
        "counterfactual_joint": joint,
        "by_family": raw["by_family"],
        "by_lang": raw["by_lang"],
        "families_below_baseline": families_down,
        "identical_picks": raw["identical_picks"],
        "not_worse": not_worse,
        "better": better,
        "verdict": verdict,
        "pass": verdict == "GO",
        "reason": reason,
        "measured_by": "training.python.ce_finetune.compare (paired "
                       "bootstrap over the same rows); the verdict rule is "
                       f"{TASK}'s, not that gate's",
    }


def producer_decision(comparison: dict) -> dict:
    """Qué `QWEN_MODEL` usa el productor. La consecuencia, escrita."""
    go = comparison["pass"]
    model = (comparison["candidate_model"] or "").split(":", 1)[-1]
    chosen = model if go else FALLBACK_MODEL
    return {
        "qwen_model": chosen,
        "applies_to": "data.stream (the producer) and data.episode_verify "
                      "(the verifier): both read data.episode_gen.MODEL",
        "why": (comparison["reason"] if go else
                comparison["reason"] + " — the producer goes back to "
                f"{FALLBACK_MODEL} until the operator decides"),
        "how": ("nothing to change: it is already the default of "
                "`data.episode_gen.MODEL`" if go else
                f"set QWEN_MODEL={FALLBACK_MODEL} in the `datagen` job "
                "command"),
        "operator_decides": not go,
    }


# ------------------------------------------------------------- el gate

def gate(write: bool = True, seed: int = BOOTSTRAP_SEED,
         reps: int = BOOTSTRAP_B) -> dict:
    """Las dos columnas, la diferencia pareada, el veredicto y la decisión."""
    episodes = P.load_cut()
    base_col, base_rep = baseline_column(), baseline_report()
    new_col = load_column(episodes)
    if new_col is None:
        raise Missing(
            f"{P._rel(column_path())} is not on disk for these rows "
            f"({P.rows_sha(episodes)[:12]}): run `measure` (the job) "
            "first. This gate does not estimate a column it has not "
            "measured")
    new_rep = (json.loads(REPORT_PATH.read_text())
               if REPORT_PATH.exists() else publish(episodes, new_col,
                                                    write=write))
    comparison = compare(base_col, new_col, episodes, seed=seed, reps=reps)
    doc = {
        "task": TASK,
        "artifact": "gate",
        "generated_utc": P.utcnow(),
        "question": "is the new local teacher worse than the one whose "
                    "reference we have, on the same development rows?",
        "rule": GATE_RULE,
        "cut": {"name": P.CUT_NAME, "n": len(episodes),
                "rows_sha256": P.rows_sha(episodes),
                "sealed_cut_read": False},
        "columns": {
            "qwen3.6": _cell(base_rep, base_col),
            "qwen3.8": _cell(new_rep, new_col),
        },
        "comparison": comparison,
        "permutation_at_temperature_0": new_col.get(
            "permutation_temperature_0"),
        "verdict": comparison["verdict"],
        "pass": comparison["pass"],
        "producer": producer_decision(comparison),
        "reports": {
            "qwen3.6": P._rel(BASELINE_REPORT),
            "qwen3.8": P._rel(REPORT_PATH),
            "columns": P._rel(COLUMN_DIR),
        },
        "not_touched": [
            "the sealed cut (no path of it was opened)",
            f"{P._rel(BASELINE_REPORT)} and {P._rel(BASELINE_COLUMN)} "
            "(reused, not rewritten)",
        ],
    }
    if write:
        GATE_DIR.mkdir(parents=True, exist_ok=True)
        GATE_PATH.write_text(json.dumps(doc, indent=2, ensure_ascii=False)
                             + "\n")
    return doc


def _cell(report: dict, column: dict) -> dict:
    """Una columna en el gate: cifras de la suite, copiadas enteras."""
    return {
        "model_version": report.get("model_version"),
        "task_that_measured_it": report.get("task"),
        "measured_utc": column.get("measured_utc"),
        "forced": P._fig(report.get("ranking") or {}),
        "with_abstention": P._fig(report.get("abstention") or {}),
        "macro_by_family": (report.get("macro") or {}).get("ranking"),
        "counterfactual_joint": P._fig(report.get("counterfactual") or {}),
        "permutation_control": {
            k: (report.get("permutation_invariance") or {}).get(k)
            for k in P._PERM_FIELDS + ("temperature",)},
        "abstained_rows": (column.get("trace") or {}).get("abstained"),
        "rejected_replies": (column.get("trace") or {}).get(
            "rejected_replies"),
    }


# --------------------------------------------------------------- la CLI

def plan() -> dict:
    """La regla y lo que falta por medir. Sin tocar un modelo."""
    episodes = P.load_cut()
    return {
        "task": TASK,
        "rule": GATE_RULE,
        "cut": {"name": P.CUT_NAME, "n": len(episodes),
                "rows_sha256": P.rows_sha(episodes)},
        "teacher_now": qwen_model_id(),
        "generator_temperature": generator_temperature(),
        "permutation_temperature": PERM_TEMPERATURE,
        "baseline": {"report": P._rel(BASELINE_REPORT),
                     "on_disk": BASELINE_REPORT.exists(),
                     "forced": (baseline_report().get("ranking") or {}).get(
                         "accuracy") if BASELINE_REPORT.exists() else None},
        "candidate_column_measured": load_column(episodes) is not None,
    }


def _line(doc: dict) -> str:
    c = doc["comparison"]
    return (f"{doc['verdict']} · forzada 3.8 "
            f"{doc['columns']['qwen3.8']['forced']['accuracy']} vs 3.6 "
            f"{doc['columns']['qwen3.6']['forced']['accuracy']} · "
            f"diferencia pareada {c['forced']['point']} {c['forced']['ci95']}"
            f" · productor → {doc['producer']['qwen_model']}")


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(prog="eval.qwen38_ref")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    m = sub.add_parser("measure")
    m.add_argument("--workers", type=int, default=P.QWEN_WORKERS)
    m.add_argument("--no-gate", action="store_true",
                   help="measure and publish, without writing the gate")
    g = sub.add_parser("gate")
    g.add_argument("--no-write", action="store_true")
    g.add_argument("--bootstrap", type=int, default=BOOTSTRAP_B)
    g.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = ap.parse_args(argv)
    if args.cmd == "plan":
        print(json.dumps(plan(), indent=2, ensure_ascii=False))
        return 0
    if args.cmd == "measure":
        episodes = P.load_cut()
        column = load_column(episodes) or measure(episodes,
                                                 workers=args.workers)
        report = publish(episodes, column)
        print(f"[qwen38] {report['model_version']}: forzada "
              f"{report['ranking']['accuracy']} "
              f"(n={report['ranking']['n']}) wrote {P._rel(REPORT_PATH)}")
        if args.no_gate:
            return 0
    doc = gate(write=not getattr(args, "no_write", False),
               seed=getattr(args, "bootstrap_seed", BOOTSTRAP_SEED),
               reps=getattr(args, "bootstrap", BOOTSTRAP_B))
    print(_line(doc))
    return 0 if doc["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
