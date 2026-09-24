"""El brazo muestreado contra el brazo exacto — #T-fullspace-objective.

Puntúa el PAR de checkpoints y aplica la regla que
`artifacts/gates/T-fullspace-objective/verdict.md` pre-registró antes de que
ninguna de las dos cifras existiera. La mecánica de puntuación es la de
`tools.bigk_arm` — mismo corte de desarrollo, mismo barrido de K, misma
lectura de abstención — y se importa en vez de copiarse: lo que cambia es la
PREGUNTA, el par y el veredicto, no cómo se mide.

* **primaria** — accuracy a cardinalidad completa (BANKING77, 77 etiquetas,
  azar 0,012987) sobre el corte de DESARROLLO congelado (n=1 000). Los
  brazos se eligen aquí; el corte reservado se lee aparte, una vez, con su
  motivo (R7).
* **control** — el brazo de la vía exacta (`bigk-fullspace`), no el de
  K≤8: la variable declarada es que el denominador sale del espacio de la
  fila, y eso sólo se lee contra el que ya normalizaba sobre ese espacio.
* **abstención** en los dos regímenes, y la stage eval K≤8 de cada
  checkpoint como diagnóstico (R1).

CLI:
    PYTHONPATH=. .venv-train/bin/python -m tools.fullspace_arm report \\
        --arm artifacts/checkpoints/decision/<sampled run>/stage-000062464 \\
        --control artifacts/checkpoints/decision/<exact run>/stage-000062464 \\
        --device mps
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from eval import cuts as CUTS  # noqa: E402
from tools.bigk_arm import arm_report, rel, supervision, utcnow  # noqa: E402

TASK = "T-fullspace-objective"
GATE_DIR = os.path.join(ROOT, "artifacts", "gates", TASK)
GATE_PATH = os.path.join(GATE_DIR, "gate.json")
BIGK_GATE = os.path.join(ROOT, "artifacts", "gates", "T-bigk-optsets",
                         "gate.json")


def verdict(arm: dict, control: dict) -> dict:
    """La regla de `verdict.md`, aplicada — no reelegida."""
    p, c = arm["primary_dev"], control["primary_dev"]
    beats = bool(p.get("beats_chance"))
    over_control = bool(p.get("accuracy") is not None
                        and c.get("accuracy") is not None
                        and p["accuracy"] > c["accuracy"])
    collapsed = (p.get("abstain_rate") is not None
                 and p["abstain_rate"] > 0.95)
    regime = arm.get("cardinality_regime") or {}
    extended = bool(regime.get("in_batch_negatives"))
    go = bool(beats and over_control and not collapsed and extended)
    if not extended:
        reading = ("this checkpoint did NOT train with the extended "
                   "denominator: its `cardinality_regime` carries no "
                   "`in_batch_negatives` block. Rule 6 of the pre-registered "
                   "verdict — the result is not published as a verdict of "
                   "this hypothesis, because an in-batch denominator built "
                   "from single-dataset batches is the exact arm of "
                   "#T-bigk-optsets with noise")
    elif collapsed:
        reading = ("the head abstains on nearly every row at full "
                   "cardinality, so the primary is not interpretable as a "
                   "ranking result — rule 4 of the pre-registered verdict")
    elif go:
        reading = ("the corrected cross-space denominator moves the primary "
                   "above chance on the development cut AND above the exact "
                   "full-space arm: the next step is the SAME arm on another "
                   "seed (R9) before the reserved cut is read once")
    else:
        reading = ("the primary interval contains chance: the SAMPLED "
                   "EXTENSION of the denominator is discarded as the cause "
                   "of the transfer failure — at this budget, one seed, a "
                   "frozen backbone, and over an in-batch pool whose "
                   "measured ceiling is 175 distinct label texts. The next "
                   "suspect stays the REPRESENTATION: #T-encoder-finetune "
                   "first, then backbone capacity (68 M -> 149 M+)")
    return {
        "go": go,
        "decided_by": ("the arm trained with the extended denominator AND "
                       "accuracy_ci95[0] > chance on the development cut "
                       "AND accuracy above the exact full-space control AND "
                       "the head is not abstaining on everything"),
        "pre_registered": rel(os.path.join(GATE_DIR, "verdict.md")),
        "beats_chance": beats,
        "above_control": over_control,
        "abstention_collapse": collapsed,
        "denominator_was_extended": extended,
        "reading": reading,
        "train_loss_is_not_comparable": (
            "the sampled loss UNDERESTIMATES the full-space loss (Jensen; "
            "training/python/test_fullspace_loss.py::"
            "test_the_loss_estimator_underestimates), so a lower train loss "
            "than the control is the estimator and not an improvement. Rule "
            "5 of the pre-registered verdict"),
        "r9": ("a NO-GO at this budget LIMITS SPEND AND DOES NOT ESTABLISH "
               f"CAUSE. What is discarded is discarded at "
               f"{arm.get('samples_seen')} rows, one seed, frozen backbone; "
               "any causal claim needs the winning arm repeated on another "
               "seed and at a larger budget"),
        "r4": ("this verdict is valid inside the regime it was measured in; "
               "it does not transfer to another one"),
    }


def _side_artifact(name: str) -> dict:
    path = os.path.join(GATE_DIR, name)
    if not os.path.exists(path):
        return {"missing": rel(path)}
    with open(path, encoding="utf-8") as fh:
        return {"artifact": rel(path), "content": json.load(fh)}


def compose(arm: dict, control: dict, sup: dict, cut, write: bool = True
            ) -> dict:
    batches = _side_artifact("batch-composition.json")
    doc = {
        "format": "jev.gate.v1",
        "task": TASK,
        "artifact": "gate",
        "generated_utc": utcnow(),
        "question": ("does extending the denominator BEYOND the row's own "
                     "label space — corrected cross-space in-batch "
                     "negatives — move the number at full cardinality, "
                     "over an arm that already normalised over that space?"),
        "pair": {
            "arm": arm["checkpoint"], "control": control["checkpoint"],
            "only_variable": ("the denominator leaves the row's own space: "
                              "corrected cross-space in-batch negatives, "
                              "which requires the mixed-batch loader and "
                              "declares it as part of the same variable"),
            "shared": ("corpus (decision-mix-clean-1m, --fence-clean), "
                       "seed 20260922, mix-seed 20260922, prior penalty 1.0, "
                       "ettin-68m frozen, d_model 512, 2 layers, 8 heads, "
                       "batch 64, max_length 256, set_attention=True"),
            "architecture_changed": False,
            "architecture_note": ("the head is untouched. An arm with "
                                  "`set_attention=False` would be a CHANGE "
                                  "OF ARCHITECTURE and no verdict would "
                                  "transfer between the two (R4, R9)"),
        },
        "cut": {
            "name": cut.name, "dataset": cut.dataset, "split": cut.split,
            "reserved": cut.reserved, "rows": arm["primary_dev"]["n"],
            "cardinality": arm["primary_dev"]["cardinality"],
            "chance": arm["primary_dev"]["chance"],
            "rule": ("R7 — arms are chosen on the development cut; the "
                     "reserved cut is read separately and logged"),
        },
        "primary": {"arm": arm["primary_dev"],
                    "control": control["primary_dev"]},
        "verdict": verdict(arm, control),
        "arm": arm,
        "control": control,
        "unknown_supervision": sup,
        "loss_spec": _side_artifact("loss-spec.json"),
        "batch_composition": batches,
        "cost": {"see": rel(os.path.join(GATE_DIR, "cost.json")),
                 "decision": rel(os.path.join(GATE_DIR,
                                              "head-decision.md"))},
        "upstream_verdict": {
            "task": "T-bigk-optsets",
            "artifact": rel(BIGK_GATE),
            "reading": ("the cardinality of the objective was discarded as "
                        "the cause at 250 048 rows, one seed, frozen "
                        "backbone (R9: that limits spend, it does not "
                        "establish cause). This task is no longer the "
                        "continuation of that hypothesis"),
        },
        "rules_honoured": [
            "R1 — the training regime and the stage-eval regime are both "
            "declared in each checkpoint's `cardinality_regime`; the primary "
            "is measured at full cardinality",
            "R2 — every accuracy here carries its K, its chance and its "
            "95 % interval",
            "R3 — no external number is restated here as a comparison",
            "R4 — the verdict names the regime it is valid in, and the "
            "architecture it is valid for",
            "R5 — the slope is shown at 62 k before anything is scaled",
            "R7 — this artifact is the development cut; reads of the "
            "reserved cut are logged in "
            "artifacts/gates/T-eval-cardinality/test-queries.json",
            "R9 — a cheap NO-GO limits spend and does not establish cause",
        ],
    }
    if write:
        os.makedirs(GATE_DIR, exist_ok=True)
        with open(GATE_PATH, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, ensure_ascii=False, sort_keys=True)
            fh.write("\n")
    return doc


def report(arm_ckpt: str, control_ckpt: str, device: str = "auto",
           limit: int | None = None, batch_size: int = 16,
           write: bool = True, log=print) -> dict:
    cut = CUTS.DEV
    samples = CUTS.samples(cut, limit)
    log(f"[arm] development cut: {len(samples)} rows, "
        f"K={len(samples[0].options) if samples else 0}")
    arm = arm_report(arm_ckpt, samples, device, batch_size, log)
    control = arm_report(control_ckpt, samples, device, batch_size, log)
    doc = compose(arm, control, supervision(), cut, write)
    log(f"[arm] verdict: {'GO' if doc['verdict']['go'] else 'NO-GO'}")
    log(f"[arm] {doc['verdict']['reading']}")
    if write:
        log(f"[arm] artifact: {rel(GATE_PATH)}")
    return doc


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(prog="tools.fullspace_arm")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="score the pair and write the gate")
    r.add_argument("--arm", required=True)
    r.add_argument("--control", required=True)
    r.add_argument("--device", default="auto")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--batch-size", type=int, default=16)
    r.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv[1:])
    doc = report(args.arm, args.control, args.device, args.limit or None,
                 args.batch_size, write=not args.no_write)
    print(json.dumps({"primary": doc["primary"], "verdict": doc["verdict"]},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
