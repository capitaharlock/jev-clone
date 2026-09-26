"""Los dos checkpoints de partida, medidos SIN ENTRENAR (#T-ce-scorer).

El paso 2 del piloto (`.meshkore/docs/plan-recuperacion-2026-09-24.md` §7)
dice medir la referencia preentrenada ANTES de tocarla, con un único
formato por modelo elegido sin mirar el test final. Esto es esa medición:
los dos checkpoints de `model.weights.SCORERS` cargan, puntúan por
entailment con las hipótesis explícitas de `model.ce_scorer`, y la cifra
se publica en `artifacts/gates/T-ce-scorer/nograd.json`.

QUÉ ES Y QUÉ NO ES ESTA CIFRA
-----------------------------

**No es la batería.** La batería de desarrollo son 400 casos revisados y
la escribe `#T-battery-dev`, que aún no existe. Esto son
`len(CASES)` casos escritos a mano aquí, con su `variant_group`, para
tener una referencia antes-de-entrenar reproducible y un control de
mecanismo. Un n de dos dígitos no estima una accuracy de producto: el
IC95 % de Wilson se publica precisamente para que nadie lea el punto
suelto. Cualquier comparación contra el 70 % del operador es inválida
hasta que exista el corte representativo.

**No es una comparación limpia en BANKING77.** Ninguno de los dos
checkpoints se evalúa aquí sobre BANKING77, y `modernbert-zeroshot-v2`
no podría: su mezcla publicada lo incluye. El aviso viaja en el
artefacto (`contaminated_benchmarks`), no sólo en este docstring.

**Sí es** la primera evidencia de que la pieza corre de punta a punta con
pesos reales: los dos checkpoints se cargan verificados por sha-256,
producen K puntuaciones por pregunta, y las tres comprobaciones del
`Verification gate` se repiten contra los pesos de verdad — no sólo
contra el juguete determinista de los tests.

Los casos cubren las cinco familias del plan §6 en ES y EN, con
contrafactuales agrupados por `variant_group`: el mismo estado con otra
pregunta, el mismo par pregunta/opciones con otro hecho decisivo. El
`pair_joint` que se publica exige acertar LAS DOS variantes del par —
acertar una sola es lo que consigue una preferencia fija por un texto.

Medio gas: CPU por defecto, lotes de 16, `OMP_NUM_THREADS` fuera. No
entrena nada; el ajuste es `#T-ce-finetune`.

CLI:
    PYTHONPATH=. .venv-train/bin/python -m eval.ce_nograd measure
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.calib import wilson_interval  # noqa: E402
from model import ce_scorer as CE  # noqa: E402
from model.weights import SCORERS, load_manifest  # noqa: E402

TASK = "T-ce-scorer"
GATE_DIR = ROOT / "artifacts" / "gates" / TASK
GATE_PATH = GATE_DIR / "nograd.json"

#: Los casos. `group` agrupa un caso con su contrafactual: ninguna
#: variante del mismo grupo cuenta como evidencia independiente de la
#: otra, y `pair_joint` mide el par entero.
CASES = [
    # -- extracción y paráfrasis: el hecho está en el estado ------------
    {"id": "color-en-green", "group": "color-en", "family": CE.EXTRACTION,
     "lang": "en", "gold": "c3",
     "state": "Pedro explicitly says: my favorite color is green.",
     "question": "What is Pedro's favorite color?",
     "candidates": [("c1", "red"), ("c2", "yellow"), ("c3", "green")]},
    {"id": "color-en-red", "group": "color-en", "family": CE.EXTRACTION,
     "lang": "en", "gold": "c1",
     "state": "Pedro explicitly says: my favorite color is red.",
     "question": "What is Pedro's favorite color?",
     "candidates": [("c1", "red"), ("c2", "yellow"), ("c3", "green")]},
    {"id": "color-es-verde", "group": "color-es", "family": CE.EXTRACTION,
     "lang": "es", "gold": "c3",
     "state": "Pedro dice explícitamente: mi color favorito es el verde.",
     "question": "¿Cuál es el color favorito de Pedro?",
     "candidates": [("c1", "rojo"), ("c2", "amarillo"), ("c3", "verde")]},
    {"id": "color-es-rojo", "group": "color-es", "family": CE.EXTRACTION,
     "lang": "es", "gold": "c1",
     "state": "Pedro dice explícitamente: mi color favorito es el rojo.",
     "question": "¿Cuál es el color favorito de Pedro?",
     "candidates": [("c1", "rojo"), ("c2", "amarillo"), ("c3", "verde")]},
    # mismo ESTADO, cambia la PREGUNTA: la exigencia del plan §4
    {"id": "city-en-lives", "group": "city-en", "family": CE.EXTRACTION,
     "lang": "en", "gold": "c2",
     "state": "Marta was born in Oviedo and now lives in Cadiz.",
     "question": "Where does Marta live now?",
     "candidates": [("c1", "Oviedo"), ("c2", "Cadiz"), ("c3", "Vigo")]},
    {"id": "city-en-born", "group": "city-en", "family": CE.EXTRACTION,
     "lang": "en", "gold": "c1",
     "state": "Marta was born in Oviedo and now lives in Cadiz.",
     "question": "Where was Marta born?",
     "candidates": [("c1", "Oviedo"), ("c2", "Cadiz"), ("c3", "Vigo")]},
    {"id": "city-es-vive", "group": "city-es", "family": CE.EXTRACTION,
     "lang": "es", "gold": "c2",
     "state": "Marta nació en Oviedo y ahora vive en Cádiz.",
     "question": "¿Dónde vive Marta ahora?",
     "candidates": [("c1", "Oviedo"), ("c2", "Cádiz"), ("c3", "Vigo")]},
    {"id": "city-es-nacio", "group": "city-es", "family": CE.EXTRACTION,
     "lang": "es", "gold": "c1",
     "state": "Marta nació en Oviedo y ahora vive en Cádiz.",
     "question": "¿Dónde nació Marta?",
     "candidates": [("c1", "Oviedo"), ("c2", "Cádiz"), ("c3", "Vigo")]},

    # -- comparación de atributos: los datos viven en las opciones ------
    {"id": "lamp-en-cheap", "group": "lamp-en", "family": CE.COMPARISON,
     "lang": "en", "gold": "c1",
     "state": "Catalogue: three lamps are on offer and all ship tomorrow.",
     "question": "Which lamp is cheaper?",
     "candidates": [("c1", "Lamp Aurora: 20 euros, lasts 2 years"),
                    ("c2", "Lamp Borealis: 50 euros, lasts 5 years"),
                    ("c3", "Lamp Caldera: 35 euros, lasts 3 years")]},
    {"id": "lamp-en-long", "group": "lamp-en", "family": CE.COMPARISON,
     "lang": "en", "gold": "c2",
     "state": "Catalogue: three lamps are on offer and all ship tomorrow.",
     "question": "Which lamp lasts longer?",
     "candidates": [("c1", "Lamp Aurora: 20 euros, lasts 2 years"),
                    ("c2", "Lamp Borealis: 50 euros, lasts 5 years"),
                    ("c3", "Lamp Caldera: 35 euros, lasts 3 years")]},
    {"id": "lamp-en-cheap-cf", "group": "lamp-en-cf", "family": CE.COMPARISON,
     "lang": "en", "gold": "c3",
     "state": "Catalogue: three lamps are on offer and all ship tomorrow.",
     "question": "Which lamp is cheaper?",
     "candidates": [("c1", "Lamp Aurora: 40 euros, lasts 2 years"),
                    ("c2", "Lamp Borealis: 50 euros, lasts 5 years"),
                    ("c3", "Lamp Caldera: 15 euros, lasts 3 years")]},
    {"id": "lamp-en-long-cf", "group": "lamp-en-cf", "family": CE.COMPARISON,
     "lang": "en", "gold": "c1",
     "state": "Catalogue: three lamps are on offer and all ship tomorrow.",
     "question": "Which lamp lasts longer?",
     "candidates": [("c1", "Lamp Aurora: 40 euros, lasts 8 years"),
                    ("c2", "Lamp Borealis: 50 euros, lasts 5 years"),
                    ("c3", "Lamp Caldera: 15 euros, lasts 3 years")]},
    {"id": "lamp-es-barata", "group": "lamp-es", "family": CE.COMPARISON,
     "lang": "es", "gold": "c1",
     "state": "Catálogo: tres lámparas en oferta, todas se envían mañana.",
     "question": "¿Qué lámpara es más barata?",
     "candidates": [("c1", "Lámpara Aurora: 20 euros, dura 2 años"),
                    ("c2", "Lámpara Boreal: 50 euros, dura 5 años"),
                    ("c3", "Lámpara Caldera: 35 euros, dura 3 años")]},
    {"id": "lamp-es-dura", "group": "lamp-es", "family": CE.COMPARISON,
     "lang": "es", "gold": "c2",
     "state": "Catálogo: tres lámparas en oferta, todas se envían mañana.",
     "question": "¿Qué lámpara dura más?",
     "candidates": [("c1", "Lámpara Aurora: 20 euros, dura 2 años"),
                    ("c2", "Lámpara Boreal: 50 euros, dura 5 años"),
                    ("c3", "Lámpara Caldera: 35 euros, dura 3 años")]},

    # -- clasificación por descripciones: la definición va con la opción -
    {"id": "ticket-en-refund", "group": "ticket-en", "family": CE.DESCRIPTION,
     "lang": "en", "gold": "c1",
     "state": ("Customer message: I was charged twice for the same order "
               "and I want my money back for the extra charge."),
     "question": "Which category does this message belong to?",
     "candidates": [
         ("c1", "Refund request: the customer asks for money to be "
                "returned for a charge already made"),
         ("c2", "Delivery delay: the customer reports that an order has "
                "not arrived on the promised date"),
         ("c3", "Account access: the customer cannot sign in or has lost "
                "the credentials of the account")]},
    {"id": "ticket-en-access", "group": "ticket-en", "family": CE.DESCRIPTION,
     "lang": "en", "gold": "c3",
     "state": ("Customer message: I cannot sign in any more, my password "
               "stopped working after the weekend."),
     "question": "Which category does this message belong to?",
     "candidates": [
         ("c1", "Refund request: the customer asks for money to be "
                "returned for a charge already made"),
         ("c2", "Delivery delay: the customer reports that an order has "
                "not arrived on the promised date"),
         ("c3", "Account access: the customer cannot sign in or has lost "
                "the credentials of the account")]},
    {"id": "ticket-es-reembolso", "group": "ticket-es",
     "family": CE.DESCRIPTION, "lang": "es", "gold": "c1",
     "state": ("Mensaje del cliente: me han cobrado dos veces el mismo "
               "pedido y quiero que me devuelvan el cargo de más."),
     "question": "¿A qué categoría pertenece este mensaje?",
     "candidates": [
         ("c1", "Solicitud de reembolso: el cliente pide que se le "
                "devuelva el dinero de un cargo ya realizado"),
         ("c2", "Retraso de entrega: el cliente informa de que un pedido "
                "no ha llegado en la fecha prometida"),
         ("c3", "Acceso a la cuenta: el cliente no puede entrar o ha "
                "perdido las credenciales de su cuenta")]},
    {"id": "ticket-es-acceso", "group": "ticket-es",
     "family": CE.DESCRIPTION, "lang": "es", "gold": "c3",
     "state": ("Mensaje del cliente: ya no consigo entrar, mi contraseña "
               "dejó de funcionar después del fin de semana."),
     "question": "¿A qué categoría pertenece este mensaje?",
     "candidates": [
         ("c1", "Solicitud de reembolso: el cliente pide que se le "
                "devuelva el dinero de un cargo ya realizado"),
         ("c2", "Retraso de entrega: el cliente informa de que un pedido "
                "no ha llegado en la fecha prometida"),
         ("c3", "Acceso a la cuenta: el cliente no puede entrar o ha "
                "perdido las credenciales de su cuenta")]},

    # -- inferencia textual y negación ----------------------------------
    {"id": "neg-en-no", "group": "neg-en", "family": CE.INFERENCE,
     "lang": "en", "gold": "c2",
     "state": ("Warehouse log: the shipment did not leave the warehouse "
               "on Monday; it stayed on dock 4 all day."),
     "question": "Did the shipment leave the warehouse on Monday?",
     "candidates": [("c1", "Yes, it left on Monday"),
                    ("c2", "No, it did not leave on Monday"),
                    ("c3", "The log does not say")]},
    {"id": "neg-en-yes", "group": "neg-en", "family": CE.INFERENCE,
     "lang": "en", "gold": "c1",
     "state": ("Warehouse log: the shipment left the warehouse on Monday "
               "at 7am from dock 4."),
     "question": "Did the shipment leave the warehouse on Monday?",
     "candidates": [("c1", "Yes, it left on Monday"),
                    ("c2", "No, it did not leave on Monday"),
                    ("c3", "The log does not say")]},
    {"id": "neg-en-unstated", "group": "neg-en-unstated",
     "family": CE.INFERENCE, "lang": "en", "gold": "c3",
     "state": ("Warehouse log: dock 4 was repainted on Monday and the "
               "forklift was serviced."),
     "question": "Did the shipment leave the warehouse on Monday?",
     "candidates": [("c1", "Yes, it left on Monday"),
                    ("c2", "No, it did not leave on Monday"),
                    ("c3", "The log does not say")]},
    {"id": "neg-es-no", "group": "neg-es", "family": CE.INFERENCE,
     "lang": "es", "gold": "c2",
     "state": ("Registro del almacén: el envío no salió del almacén el "
               "lunes; se quedó en el muelle 4 todo el día."),
     "question": "¿Salió el envío del almacén el lunes?",
     "candidates": [("c1", "Sí, salió el lunes"),
                    ("c2", "No, no salió el lunes"),
                    ("c3", "El registro no lo dice")]},
    {"id": "neg-es-si", "group": "neg-es", "family": CE.INFERENCE,
     "lang": "es", "gold": "c1",
     "state": ("Registro del almacén: el envío salió del almacén el lunes "
               "a las 7 de la mañana desde el muelle 4."),
     "question": "¿Salió el envío del almacén el lunes?",
     "candidates": [("c1", "Sí, salió el lunes"),
                    ("c2", "No, no salió el lunes"),
                    ("c3", "El registro no lo dice")]},
    {"id": "neg-es-nodice", "group": "neg-es-nodice", "family": CE.INFERENCE,
     "lang": "es", "gold": "c3",
     "state": ("Registro del almacén: el lunes se repintó el muelle 4 y se "
               "revisó la carretilla elevadora."),
     "question": "¿Salió el envío del almacén el lunes?",
     "candidates": [("c1", "Sí, salió el lunes"),
                    ("c2", "No, no salió el lunes"),
                    ("c3", "El registro no lo dice")]},

    # -- decisiones con prioridades: el criterio ordena y desempata -----
    {"id": "prio-en-duration", "group": "prio-en", "family": CE.PRIORITY,
     "lang": "en", "gold": "c2",
     "state": ("Purchasing rule: prioritise duration; if two options last "
               "the same, take the lower price."),
     "question": "Which battery should purchasing buy?",
     "candidates": [("c1", "Battery Kestrel: 12 euros, lasts 4 months"),
                    ("c2", "Battery Lynx: 30 euros, lasts 9 months"),
                    ("c3", "Battery Marten: 18 euros, lasts 6 months")]},
    {"id": "prio-en-tiebreak", "group": "prio-en", "family": CE.PRIORITY,
     "lang": "en", "gold": "c3",
     "state": ("Purchasing rule: prioritise duration; if two options last "
               "the same, take the lower price."),
     "question": "Which battery should purchasing buy?",
     "candidates": [("c1", "Battery Kestrel: 12 euros, lasts 4 months"),
                    ("c2", "Battery Lynx: 30 euros, lasts 9 months"),
                    ("c3", "Battery Marten: 18 euros, lasts 9 months")]},
    {"id": "prio-es-duracion", "group": "prio-es", "family": CE.PRIORITY,
     "lang": "es", "gold": "c2",
     "state": ("Norma de compras: prioriza la duración; si dos opciones "
               "duran lo mismo, elige la de menor precio."),
     "question": "¿Qué pila debe comprar el departamento?",
     "candidates": [("c1", "Pila Cernícalo: 12 euros, dura 4 meses"),
                    ("c2", "Pila Lince: 30 euros, dura 9 meses"),
                    ("c3", "Pila Marta: 18 euros, dura 6 meses")]},
    {"id": "prio-es-desempate", "group": "prio-es", "family": CE.PRIORITY,
     "lang": "es", "gold": "c3",
     "state": ("Norma de compras: prioriza la duración; si dos opciones "
               "duran lo mismo, elige la de menor precio."),
     "question": "¿Qué pila debe comprar el departamento?",
     "candidates": [("c1", "Pila Cernícalo: 12 euros, dura 4 meses"),
                    ("c2", "Pila Lince: 30 euros, dura 9 meses"),
                    ("c3", "Pila Marta: 18 euros, dura 9 meses")]},
]


def decisions() -> list[CE.Decision]:
    """Los casos como `Decision`s, con el contrato de su familia."""
    out = []
    for case in CASES:
        out.append(CE.Decision(
            state=case["state"], question=case["question"],
            candidates=tuple(CE.Candidate(cid, text)
                             for cid, text in case["candidates"]),
            family=case["family"], lang=case["lang"], gold=case["gold"],
            meta={"id": case["id"], "group": case["group"]}))
    return out


def _bucket(rows: list[dict], key: str) -> dict:
    """accuracy por familia / idioma / K, con su n al lado."""
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(str(row[key]), []).append(row)
    out = {}
    for name, group in sorted(buckets.items()):
        hits = sum(1 for r in group if r["correct"])
        out[name] = {"n": len(group), "correct": hits,
                     "accuracy": hits / len(group),
                     "chance": sum(1.0 / r["k"] for r in group) / len(group)}
    return out


def _pairs(rows: list[dict]) -> dict:
    """Éxito CONJUNTO por grupo contrafactual: las dos o ninguna."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    multi = {g: rs for g, rs in groups.items() if len(rs) > 1}
    joint = {g: all(r["correct"] for r in rs) for g, rs in sorted(multi.items())}
    n = len(joint)
    hits = sum(1 for v in joint.values() if v)
    lo, hi = wilson_interval(hits, n) if n else (0.0, 1.0)
    return {"n_groups": n, "joint_correct": hits,
            "joint_rate": (hits / n) if n else None,
            "ci95": [lo, hi], "per_group": joint,
            "note": ("a counterfactual group counts only when EVERY variant "
                     "is right; one right out of two is what a fixed "
                     "preference for a text already achieves")}


def measure_one(weight_id: str, device: str = "cpu",
                batch_size: int = CE.DEFAULT_BATCH_SIZE) -> dict:
    """Un checkpoint, sin entrenar, en los dos modos de reducción."""
    import torch  # noqa: F401  (falla aquí y no a mitad de la medición)

    scorer = CE.NliPairScorer(weight_id, device=device, batch_size=batch_size)
    decs = decisions()
    flat: list[tuple[str, str]] = []
    spans = []
    for dec in decs:
        pairs = CE.render_pairs(dec)
        spans.append((len(flat), len(flat) + len(pairs)))
        flat.extend(pairs)
    t0 = time.perf_counter()
    class_logits = scorer.class_logits(flat)   # UNA pasada, dos lecturas
    wall = time.perf_counter() - t0

    modes = {}
    for mode in CE.SCORE_MODES:
        scores = CE.reduce_logits(class_logits, mode, scorer.entail_index)
        rows = []
        for dec, (lo, hi) in zip(decs, spans):
            z = scores[lo:hi]
            probs = CE.softmax(z)
            ids = [c.id for c in dec.candidates]
            best = max(range(len(ids)), key=lambda i: z[i])
            rows.append({
                "id": dec.meta["id"], "group": dec.meta["group"],
                "family": dec.family, "lang": dec.lang, "k": dec.k,
                "comparative_context": dec.wants_context(),
                "gold": dec.gold, "argmax_id": ids[best],
                "correct": ids[best] == dec.gold,
                "p_gold": probs[ids.index(dec.gold)],
                "z": [round(v, 6) for v in z]})
        hits = sum(1 for r in rows if r["correct"])
        n = len(rows)
        lo, hi = wilson_interval(hits, n)
        by_family = _bucket(rows, "family")
        modes[mode] = {
            "n": n, "correct": hits, "accuracy": hits / n, "ci95": [lo, hi],
            "chance": sum(1.0 / r["k"] for r in rows) / n,
            "macro_family": sum(v["accuracy"] for v in by_family.values())
                            / len(by_family),
            "mean_nll": sum(-math.log(max(r["p_gold"], 1e-12))
                            for r in rows) / n,
            "by_family": by_family,
            "by_lang": _bucket(rows, "lang"),
            "by_k": _bucket(rows, "k"),
            "counterfactual_pairs": _pairs(rows),
            "rows": rows,
        }

    # las tres comprobaciones del gate, ahora contra pesos REALES
    ce = CE.CrossEncoderScorer(scorer)
    contract = {
        "permutation_invariance": CE.check_permutation_invariance(
            ce, CE.CONTRACT_CASE),
        "text_follows_id": CE.check_text_follows_id(ce, CE.CONTRACT_CASE),
        "comparative_context": CE.check_comparative_context(
            ce, CE.CONTRACT_CASE, CE.CONTRACT_FOREIGN_ID,
            CE.CONTRACT_FOREIGN_TEXT),
    }
    return {
        "checkpoint": scorer.describe(),
        "manifest": load_manifest(weight_id),
        "pairs": len(flat),
        "forward_seconds": round(wall, 3),
        "ms_per_pair": round(1000.0 * wall / max(len(flat), 1), 3),
        "trained_here": False,
        "modes": modes,
        "contract_checks": contract,
        "contract_pass": all(c["pass"] for c in contract.values()),
    }


def measure(device: str = "cpu", batch_size: int = CE.DEFAULT_BATCH_SIZE,
            weight_ids: list[str] | None = None, write: bool = True) -> dict:
    ids = weight_ids or list(SCORERS)
    results = {wid: measure_one(wid, device=device, batch_size=batch_size)
               for wid in ids}
    report = {
        "task": TASK,
        "generated_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "scope": (f"{len(CASES)} hand-written probe cases, five families, "
                  "ES+EN, K=3. NOT the development battery (#T-battery-dev, "
                  "400 reviewed cases, does not exist yet) and NOT a "
                  "product accuracy: read the Wilson interval, not the "
                  "point. No training happened here (#T-ce-finetune)."),
        "device": device,
        "batch_size": batch_size,
        "hypothesis_format": CE.HYPOTHESIS,
        "premise_format": CE.PREMISE,
        "comparative_context_by_family": CE.COMPARATIVE_CONTEXT,
        "default_score_mode": CE.DEFAULT_SCORE_MODE,
        "score_mode_note": (
            "entail_logit is the published zero-shot recipe of these "
            "checkpoints; entail_logprob is published alongside because the "
            "two heads have 3 and 2 classes and their raw logits are not on "
            "the same scale"),
        "warnings": [
            f"{wid}: training mix includes "
            f"{list(SCORERS[wid]['contaminated_benchmarks'])} — not a clean "
            f"transfer claim for those benchmarks"
            for wid in ids if SCORERS[wid].get("contaminated_benchmarks")],
        "checkpoints": results,
    }
    if write:
        GATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(GATE_PATH, "w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True,
                      ensure_ascii=False)
            fh.write("\n")
    return report


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["measure"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=CE.DEFAULT_BATCH_SIZE)
    ap.add_argument("--checkpoint", action="append", dest="checkpoints")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv[1:])

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    report = measure(device=args.device, batch_size=args.batch_size,
                     weight_ids=args.checkpoints, write=not args.no_write)
    for wid, res in report["checkpoints"].items():
        for mode, m in res["modes"].items():
            print(f"[nograd] {wid} {mode}: {m['correct']}/{m['n']} = "
                  f"{m['accuracy']:.3f} "
                  f"CI95[{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}] "
                  f"chance={m['chance']:.3f} "
                  f"pairs={m['counterfactual_pairs']['joint_correct']}/"
                  f"{m['counterfactual_pairs']['n_groups']}")
        print(f"[nograd] {wid} contract: "
              f"{'PASS' if res['contract_pass'] else 'FAIL'} "
              f"({res['ms_per_pair']} ms/pair on {report['device']})")
    if not args.no_write:
        print(f"[nograd] wrote {GATE_PATH.relative_to(ROOT)}")
    return 0 if all(r["contract_pass"]
                    for r in report["checkpoints"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
