"""¿Capacidad o entreno? Sobreajustar las familias numéricas (#T-capacity-probe).

La conclusión de `#T-numeric-gen` («el cuello es el backbone») salió de un
smoke de 625 pasos y 1 época cuya pérdida de entreno bajó sólo de 1,07 a
1,02: el modelo no llegó a ajustar ni lo que vio. Eso no distingue «no
puede» de «no se le dejó». Esta sonda sí:

* **fit**: 512 episodios por regla de `attribute_comparison` y
  `priority_decision` (lote 1), muchas épocas. Se mide el acierto sobre
  ESOS MISMOS 512.
* **unseen**: 1 024 episodios de las mismas familias, de grupos que no
  están en fit. Mide si lo ajustado generaliza en distribución.

Umbrales escritos ANTES de medir (`PREDICTION`). Si fit supera 0,95, la
capacidad existe y el smoke estaba sub-entrenado. Si fit se queda por
debajo de 0,70, el MiniLM-L6 no puede representar la comparación en este
formato (pairwise): el siguiente brazo es formato/backbone.

CLI:
    PYTHONPATH=. .venv-train/bin/python -m training.python.ce_capacity \\
        run --device mps
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.python import ce_finetune as FT  # noqa: E402
from training.python import ce_overfit as OF  # noqa: E402

TASK = "T-capacity-probe"
GATE_DIR = ROOT / "artifacts" / "gates" / TASK
SOURCE = ROOT / "artifacts" / "episodes-rule" / "batch-0001" / "episodes.jsonl"
FAMILIES = ("attribute_comparison", "priority_decision")
N_FIT, N_UNSEEN = 512, 1024
SEED = 20260930

PREDICTION = {
    "written_utc": "2026-09-30",
    "written_before": "any figure of this probe",
    "capacity_ok_if": "fit accuracy > 0.95 at the end of the run",
    "capacity_limit_if": "fit accuracy < 0.70 at the end of the run",
    "in_between": "partial: capacity exists but is slow to use; read unseen",
    "expect": "fit > 0.95 (the 48-case mechanics overfit reached 0.958 on "
              "hand-made numeric cases); unseen well above the untuned "
              "control but below fit",
}


def pick(seed: int = SEED):
    """fit y unseen por GRUPO (los contrafactuales de un grupo no se parten)."""
    groups: dict[str, list[dict]] = {}
    with open(SOURCE, encoding="utf-8") as fh:
        for line in fh:
            ep = json.loads(line)
            if ep["family"] in FAMILIES:
                groups.setdefault(ep["variant_group"], []).append(ep)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    fit, unseen, i = [], [], 0
    while len(fit) < N_FIT:
        fit += groups[keys[i]]; i += 1
    while len(unseen) < N_UNSEEN:
        unseen += groups[keys[i]]; i += 1
    return fit[:N_FIT], unseen[:N_UNSEEN]


def run(args) -> dict:
    import torch

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    fit_eps, unseen_eps = pick(args.seed)
    fit = [FT.decision_of(e) for e in fit_eps]
    unseen = [FT.decision_of(e) for e in unseen_eps]
    tok, model, entail, source = FT.load_model(FT.CONTROL, args.weights,
                                               args.device)
    device = torch.device(args.device)
    opt = torch.optim.AdamW(FT.param_groups(model, args.lr_encoder,
                                            args.lr_head),
                            weight_decay=FT.WEIGHT_DECAY)
    rng = random.Random(args.seed)
    curve, steps, t0 = [], 0, time.perf_counter()

    def measure(epoch: int, loss: float | None) -> dict:
        f = OF.evaluate(model, tok, fit, device, entail, args.batch)
        u = OF.evaluate(model, tok, unseen, device, entail, args.batch)
        row = {"epoch": epoch, "steps": steps, "train_loss": loss,
               "fit": round(f["accuracy"], 4), "fit_nll": round(f["mean_nll"], 4),
               "unseen": round(u["accuracy"], 4),
               "unseen_ci95": [round(x, 4) for x in u["ci95"]],
               "fit_by_family": {k: v["accuracy"] for k, v in f["by_family"].items()},
               "unseen_by_family": {k: v["accuracy"]
                                    for k, v in u["by_family"].items()},
               "chance": round(u["chance"], 4),
               "seconds": round(time.perf_counter() - t0, 1)}
        curve.append(row)
        print(f"[capacity] epoch {epoch} loss {loss} fit {row['fit']} "
              f"unseen {row['unseen']} ({row['seconds']}s)", flush=True)
        return row

    measure(0, None)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(fit)))
        rng.shuffle(order)
        tot = n = 0
        for s in range(0, len(order), args.batch):
            chunk = [OF.shuffled_candidates(fit[i], rng)
                     for i in order[s:s + args.batch]]
            scores, _, _ = OF.forward_logits(model, tok, chunk, device, entail)
            loss = FT.listwise_loss(scores, FT.positive_mask(chunk,
                                                             scores.shape[1]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), FT.GRAD_CLIP)
            opt.step()
            opt.zero_grad(set_to_none=True)
            steps += 1
            tot += float(loss.detach()); n += 1
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            measure(epoch, round(tot / n, 4))

    final = curve[-1]
    verdict = ("capacity-ok" if final["fit"] > 0.95 else
               "capacity-limit" if final["fit"] < 0.70 else "partial")
    doc = {"task": TASK, "generated_utc": FT.utcnow(),
           "prediction": PREDICTION, "verdict": verdict,
           "model": source, "source": FT._rel(SOURCE),
           "families": FAMILIES, "n_fit": len(fit), "n_unseen": len(unseen),
           "hyperparams": {"epochs": args.epochs, "lr_encoder": args.lr_encoder,
                           "lr_head": args.lr_head, "decisions_per_batch": args.batch,
                           "seed": args.seed, "device": args.device},
           "curve": curve}
    out = Path(args.out)
    FT._dump(out, doc)
    print(f"[capacity] verdict {verdict}; wrote {FT._rel(out)}", flush=True)
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["run"])
    ap.add_argument("--weights", default=FT.DEFAULT_WEIGHTS)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--lr-encoder", type=float, default=3e-5)
    ap.add_argument("--lr-head", type=float, default=FT.DEFAULT_LR_HEAD)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=str(GATE_DIR / "fit.json"))
    args = ap.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
