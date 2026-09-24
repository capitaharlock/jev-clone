"""Cuántas etiquetas únicas trae DE VERDAD un batch — #T-fullspace-objective.

La redacción original de la task prometía «cientos de etiquetas por batch»
para los negativos in-batch. Eso no se sigue del cargador: `MixtureStream.
epoch()` (`train_decision.py:399-416`) sirve cada batch de **un solo
dataset** — pide el siguiente batch al sampler más atrasado y devuelve ése,
entero. En una tarea binaria la unión de etiquetas de 64 filas tiene dos
candidatos, no cientos.

Esto lo MIDE sobre la mezcla real del brazo con el que esta task se compara
(`decision-mix-clean-1m`, `--fence-clean`, mix-seed 20260922, B=64) y
publica tres cosas:

1. la distribución de etiquetas únicas, espacios y datasets por batch tal
   como el cargador los sirve hoy;
2. la misma distribución si los batches fueran **mixtos** — el mismo
   presupuesto de filas intercalando los datasets en vez de agotarlos de
   uno en uno — que es el contrafactual que decide si el cambio del
   cargador hace falta;
3. las **colisiones de texto de etiqueta entre espacios**, que es el pool
   de falsos negativos detectables que `training.python.fullspace_loss.
   filter_pool` tiene que quitar antes de muestrear.

No entrena nada, no escribe checkpoint y no lee ningún corte reservado.

CLI:
    PYTHONPATH=. .venv-train/bin/python -m tools.fullspace_batches --batches 400
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import mix as mixmod  # noqa: E402
from data.optset import PREFETCH_DIR, SamplerConfig, label_pool  # noqa: E402
from training.python.fullspace_loss import normalise_key  # noqa: E402
from training.python.train_decision import (MixtureStream,  # noqa: E402
                                            build_mix, train_samplers)

TASK = "T-fullspace-objective"
GATE_DIR = os.path.join(ROOT, "artifacts", "gates", TASK)

#: el brazo con el que esta task se compara (`#T-bigk-optsets`)
ARM_SEED = 20260922
ARM_BATCH = 64


def batch_shape(batch: list) -> dict:
    """Lo que una fila del informe mide sobre UN batch."""
    golds = {s.answer for s in batch if not s.is_unknown}
    offered = {(s.dataset, o["id"]) for s in batch for o in s.options}
    return {
        "rows": len(batch),
        "datasets": len({s.dataset for s in batch}),
        "spaces": len({s.space or s.dataset for s in batch}),
        "unique_golds": len(golds),
        "unique_offered": len(offered),
        "unique_offered_texts": len(
            {normalise_key(o["text"]) for s in batch for o in s.options}),
    }


def summarise(shapes: list, key: str) -> dict:
    vals = sorted(s[key] for s in shapes)
    if not vals:
        return {}
    return {
        "min": vals[0], "max": vals[-1],
        "mean": round(statistics.fmean(vals), 4),
        "median": statistics.median(vals),
        "p10": vals[max(0, int(0.10 * len(vals)) - 1)],
        "p90": vals[min(len(vals) - 1, int(0.90 * len(vals)))],
    }


def report(shapes: list) -> dict:
    out = {k: summarise(shapes, k) for k in
           ("rows", "datasets", "spaces", "unique_golds", "unique_offered",
            "unique_offered_texts")}
    out["batches"] = len(shapes)
    out["pct_single_dataset_batches"] = round(
        100.0 * sum(1 for s in shapes if s["datasets"] == 1)
        / max(len(shapes), 1), 4)
    hist: dict = {}
    for s in shapes:
        hist[str(s["unique_golds"])] = hist.get(str(s["unique_golds"]), 0) + 1
    out["unique_golds_histogram"] = dict(sorted(
        hist.items(), key=lambda kv: int(kv[0])))
    return out


def mixed_batches_of(batches: list, batch_size: int) -> list:
    """El contrafactual: los mismos batches, intercalados entre datasets.

    Se toman las filas tal como salieron y se reparten round-robin por
    dataset, de modo que cada batch mixto lleva filas de tantos datasets
    como haya vivos en ese punto del stream. Es el mismo presupuesto de
    filas: lo único que cambia es el AGRUPAMIENTO, que es exactamente la
    variable que decide si el cargador tiene que cambiar.
    """
    by_dataset: dict = {}
    for batch in batches:
        for sample in batch:
            by_dataset.setdefault(sample.dataset, []).append(sample)
    queues = [list(v) for v in by_dataset.values()]
    out, current = [], []
    while any(queues):
        for q in queues:
            if not q:
                continue
            current.append(q.pop())
            if len(current) == batch_size:
                out.append(current)
                current = []
    if current:
        out.append(current)
    return out


def false_negative_mass(batches: list) -> dict:
    """La tasa de falso negativo por MASA, no por tipo.

    Contar «4 textos de 175 aparecen en más de un espacio» subestima el
    problema: `yes`/`no` son la mitad de los oros de una tarea binaria, así
    que en un batch mixto entran en el pool de casi todas las filas. Esto
    recorre las filas tal cual y cuenta, para cada una, cuántas etiquetas
    del pool in-batch (las de las OTRAS filas) normalizan a su propio oro.
    Son los falsos negativos DETECTABLES: `filter_pool` los quita, y lo que
    esta cifra dice es cuánto trabajo hace ese filtro.
    """
    rows = hit_rows = hits = pool_sum = 0
    for batch in batches:
        texts = [(i, normalise_key(o["text"]))
                 for i, s in enumerate(batch) for o in s.options]
        for i, sample in enumerate(batch):
            if sample.is_unknown:
                continue
            gold = next((normalise_key(o["text"]) for o in sample.options
                         if o["id"] == sample.answer), None)
            if gold is None:
                continue
            foreign = {t for j, t in texts if j != i}
            own = {normalise_key(o["text"]) for o in sample.options}
            rows += 1
            pool_sum += len(foreign - own)
            # una etiqueta OFRECIDA POR OTRA FILA cuyo texto es el oro de
            # ésta. Si el denominador in-batch se arma sin deduplicar por
            # clave, esa columna extra es el oro presentado como negativo y
            # el gradiente empuja hacia abajo la respuesta correcta.
            if gold in foreign:
                hit_rows += 1
                hits += sum(1 for _j, t in texts if t == gold) - 1
    return {
        "rows_scored": rows,
        "mean_in_batch_pool_per_row": round(pool_sum / max(rows, 1), 3),
        "rows_with_a_gold_collision": hit_rows,
        "mean_duplicate_gold_columns_per_affected_row": round(
            hits / max(hit_rows, 1), 3),
        "pct_rows_with_a_gold_collision": round(
            100.0 * hit_rows / max(rows, 1), 4),
        "reading": ("the share of rows for which an UNFILTERED in-batch "
                    "denominator would push the gradient down on their own "
                    "correct answer, because another row offered the same "
                    "label text. `fullspace_loss.filter_pool` removes these "
                    "from the pool before sampling"),
    }


def cross_space_collisions(datasets: list, root: str) -> dict:
    """Textos de etiqueta que aparecen en MÁS de un espacio.

    Cada uno es un falso negativo detectable en cuanto un batch mezcle
    espacios: la misma etiqueta ofrecida como distractor de una fila cuyo
    oro es esa misma etiqueta en otra taxonomía. `filter_pool` los quita
    por clave normalizada; los semánticos (texto distinto, significado
    igual) no son detectables así y quedan declarados como sesgo.
    """
    where: dict = {}
    sizes = {}
    for dataset in datasets:
        pool = label_pool(dataset, root)
        sizes[dataset] = len(pool)
        for opt in pool:
            key = normalise_key(opt.get("text") or opt["id"])
            where.setdefault(key, set()).add(dataset)
    shared = {k: sorted(v) for k, v in where.items() if len(v) > 1}
    return {
        "space_sizes": sizes,
        "distinct_label_texts": len(where),
        "texts_in_more_than_one_space": len(shared),
        "examples": dict(sorted(shared.items())[:12]),
        "detectable_by_text_identity": True,
        "residual": ("a SEMANTIC false negative — another space's label "
                     "that would also be correct for the row but is written "
                     "differently — is not detectable this way and stays a "
                     "declared bias of the sampled estimator"),
    }


def measure(batches: int, batch_size: int, seed: int, root: str) -> dict:
    t0 = time.perf_counter()
    # la MISMA receta que `train(--fence-clean)` arma: mismo target, misma
    # semilla y mismo margen de cap que `tools.mix_1m.run_mix` publica.
    clean = mixmod.clean_datasets()
    spec, _scan, pools, holdout = build_mix(
        seed, root, mixmod.CLEAN_1M_TARGET, clean,
        mixmod.CLEAN_1M_CAP_MARGIN, mixmod.layer_weights(clean), None, None)
    config = SamplerConfig(seed=seed)
    tr = train_samplers(holdout, config, None, root, spec, pools)
    stream = MixtureStream(tr, batch_size)
    served, shapes = [], []
    for batch in stream.epoch(0):
        served.append(batch)
        shapes.append(batch_shape(batch))
        if len(shapes) >= batches:
            break
    mixed_batches = mixed_batches_of(served, batch_size)
    mixed = [batch_shape(b) for b in mixed_batches]
    return {
        "task": TASK,
        "artifact": "batch-composition",
        "format": "jev.gate.v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loader": {
            "source": "training/python/train_decision.py:399-416 "
                      "(`MixtureStream.epoch`)",
            "how": "the least-served sampler is asked for its next batch and "
                   "that batch is yielded whole: one dataset per batch",
            "mixture": "decision-mix-clean-1m (--fence-clean)",
            "mix_seed": seed, "sampler_seed": seed,
            "batch_size": batch_size,
            "epoch_rows": stream.total(),
            "datasets": sorted(tr),
        },
        "as_served": report(shapes),
        "if_batches_were_mixed": report(mixed),
        "false_negatives_if_mixed": false_negative_mass(mixed_batches),
        "false_negatives_as_served": false_negative_mass(served),
        "cross_space": cross_space_collisions(sorted(tr), root),
        "seconds": round(time.perf_counter() - t0, 1),
    }


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batches", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=ARM_BATCH)
    ap.add_argument("--seed", type=int, default=ARM_SEED)
    ap.add_argument("--root", default=PREFETCH_DIR)
    ap.add_argument("--out", default=os.path.join(GATE_DIR,
                                                  "batch-composition.json"))
    args = ap.parse_args(argv[1:])
    out = measure(args.batches, args.batch_size, args.seed, args.root)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    print(json.dumps({k: out[k] for k in ("as_served",
                                          "if_batches_were_mixed")},
                     indent=2, sort_keys=True)[:2400])
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
