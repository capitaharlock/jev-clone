"""Label-space inventory of a realised mixture (#T-labelspace-div, mitad 1).

A *label space* is the canonical option set a row chooses from:

* a **shared** space is a small label pool reused across rows (banking77's
  77 intents, dbpedia14's 14 classes, boolq's yes/no). Rows sample K options
  from it, so a text->label map learned on one row transfers to the next —
  the mechanism #T-antiscale-diag eje 1 measured at 100 % of the 1 M mix.
* a **per-row** space is an option set that never repeats (a swag row's
  four continuations, a prog-gold row's entities). Each one is its own
  space of ~1 row: nothing transfers, nothing is memorised.

The split is empirical, not per-dataset prejudice: `shared` iff the union
of option texts over the realised rows (`pool_size`) fits in
`SHARED_POOL_THRESHOLD`. A shared pool bigger than that is a contradiction
— a "taxonomy" nobody ever sees twice is not a taxonomy.

`inventory_of()` rebuilds the mix's own `MixSpec` from its manifest (quotas
+ keep fractions + seen labels, the same three numbers a training run
consumes) and walks the SAME selection `data.mix.realise()` counts, so the
space counts describe the corpus that trained, not the raw prefetch dir.
`members_sha256` is recomputed and compared: a mismatch means the walk
drifted from the manifest and the inventory is refused, not published.

Run:
    python3 -m data.labelspace --manifest \
        artifacts/mix-1m/decision-mix-clean-1m-seed20260922.manifest.json \
        --out artifacts/gates/T-labelspace-div/inventory.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data import mix as mixmod  # noqa: E402

#: a label pool at most this big, reused across rows, is ONE shared space.
#: banking77 (77 intents) is the largest real taxonomy; swag's smallest
#: honest pool (4 x 73 k unique continuations) is three orders of magnitude
#: above. Anything between would be a new kind of corpus, and the
#: methodology note in the output says which side it fell on.
SHARED_POOL_THRESHOLD = 256

#: a mix is "poor" when at least this share of its rows sits in shared
#: spaces — few taxonomies covering almost all rows, the shape the brief
#: asks the alternative mix to break out of.
POOR_SHARED_SHARE = 0.80

#: a space with at least this many rows counts as REUTILIZABLE (learnable
#: across rows). 10 is below every shared taxonomy's smallest row count
#: and above the singleton regime that dominates the per-row tail.
REUSABLE_MIN_ROWS = 10


def option_signature(question: dict) -> str:
    """One space identity per question: sha1 of its sorted option texts."""
    texts = sorted(o.get("text", "") for o in question.get("options", []))
    return hashlib.sha1(
        json.dumps(texts, ensure_ascii=False).encode()).hexdigest()


def classify_space(pool_size: int,
                   threshold: int = SHARED_POOL_THRESHOLD) -> str:
    """`shared` when the pool is small enough to be learned across rows."""
    return "shared" if pool_size <= threshold else "per-row"


def collect_spaces(spec: mixmod.MixSpec,
                   root: str = mixmod.PREFETCH_DIR,
                   threshold: int = SHARED_POOL_THRESHOLD) -> dict:
    """Walk the spec's selection; per-dataset pools + option-set signatures.

    Returns `{dataset: {"questions": n, "pool": set, "signatures": {sig: n,
    "k": k}}}``. `pool` holds option texts, `signatures` counts how many
    questions showed exactly that option set.
    """
    per_dataset: dict = {}

    def on_member(dataset: str, _index: int, _row: dict, q: dict) -> None:
        entry = per_dataset.setdefault(
            dataset, {"questions": 0, "pool": set(), "signatures": {},
                      "k_of_sig": {}})
        entry["questions"] += 1
        options = q.get("options", [])
        for o in options:
            entry["pool"].add(o.get("text", ""))
        sig = option_signature(q)
        entry["signatures"][sig] = entry["signatures"].get(sig, 0) + 1
        entry["k_of_sig"].setdefault(sig, len(options))

    counts = mixmod.realise(spec, root, on_member=on_member)
    return per_dataset, counts


def inventory_of(manifest_path: str,
                 root: str = mixmod.PREFETCH_DIR,
                 threshold: int = SHARED_POOL_THRESHOLD) -> dict:
    """Full inventory of the mixture a manifest describes."""
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    spec = mixmod.spec_from_manifest(manifest)
    # the manifest's own holdout is the fence the mix trained under: rows
    # whose gold is not in `seen` never entered the corpus.
    holdout = manifest.get("holdout", {})
    spec.seen_labels = {
        d: sorted(h["seen"]) for d, h in holdout.items()
        if h.get("unseen") and h.get("seen") and d in spec.quotas}
    per_dataset, counts = collect_spaces(spec, root)

    spaces = []
    for dataset in sorted(per_dataset):
        entry = per_dataset[dataset]
        pool = entry["pool"]
        kind = classify_space(len(pool), threshold)
        spaces.append({"dataset": dataset, "kind": kind,
                       "filas": entry["questions"],
                       "pool_size": len(pool),
                       "n_distinct_optionsets": len(entry["signatures"])})
    shared_rows = sum(s["filas"] for s in spaces if s["kind"] == "shared")
    total = sum(s["filas"] for s in spaces)
    n_per_row_spaces = sum(s["n_distinct_optionsets"] for s in spaces
                           if s["kind"] == "per-row")
    n_shared = sum(1 for s in spaces if s["kind"] == "shared")
    # per-row size histogram: how many spaces hold 1 row, 2 rows, ...
    # A space with >= REUSABLE_MIN_ROWS rows is REUTILIZABLE: seen often
    # enough to be learned, as opposed to a singleton nobody can memorise
    # OR generalise from. The 10x bar of the alternative mix is measured
    # on this count, not on singletons.
    size_hist: dict = {}
    n_reutilizables = n_shared
    for dataset in per_dataset:
        entry = per_dataset[dataset]
        if classify_space(len(entry["pool"]), threshold) != "per-row":
            continue
        for sig, n in entry["signatures"].items():
            size_hist[str(n)] = size_hist.get(str(n), 0) + 1
            if n >= REUSABLE_MIN_ROWS:
                n_reutilizables += 1
    by_dataset = dict(sorted(counts.get("dataset", {}).items(),
                             key=lambda kv: -kv[1]))
    top1 = next(iter(by_dataset.values())) / total if total else 0.0
    top3 = sum(list(by_dataset.values())[:3]) / total if total else 0.0

    unseen_by_cut = {
        f"{d}-holdout": h["unseen"] for d, h in sorted(holdout.items())
        if h.get("unseen")}
    overlap = _overlap_with_walk(spec, root, unseen_by_cut)

    return {
        "mix": manifest.get("corpus", manifest.get("task", "?")),
        "manifest": os.path.relpath(manifest_path, ROOT),
        "manifest_members_sha256": manifest.get("members_sha256"),
        "realised_members_sha256": counts.get("members_sha256"),
        "selection_match": (counts.get("members_sha256")
                            == manifest.get("members_sha256")),
        "method": {
            "shared_pool_threshold": threshold,
            "signature": "sha1(sorted(option texts))",
            "space_rule": ("one space per dataset while pool_size <= "
                           "threshold, else one space per distinct "
                           "option set"),
            "poor_rule": (f"pobre iff shared rows >= "
                          f"{POOR_SHARED_SHARE:.2f} of realised rows"),
        },
        "n_espacios": n_shared + n_per_row_spaces,
        "n_shared": n_shared,
        "n_per_row_spaces": n_per_row_spaces,
        "n_reutilizables": n_reutilizables,
        "reutilizable_min_filas": REUSABLE_MIN_ROWS,
        "filas_realizadas": total,
        "filas_en_shared": shared_rows,
        "fraccion_shared": round(shared_rows / total, 6) if total else 0.0,
        "filas_por_espacio_shared": {
            s["dataset"]: s["filas"] for s in spaces
            if s["kind"] == "shared"},
        "per_row_size_hist": dict(sorted(size_hist.items(),
                                         key=lambda kv: int(kv[0]))),
        "concentracion": {"top1": round(top1, 6), "top3": round(top3, 6)},
        "espacios": spaces,
        "solape_unseen": overlap,
        "veredicto": ("pobre" if total and shared_rows / total >=
                      POOR_SHARED_SHARE else "diverso"),
    }


def _overlap_with_walk(spec: mixmod.MixSpec, root: str,
                       unseen_by_cut: dict) -> dict:
    """Second walk: per-cut option/answer contact with unseen labels."""
    cuts = {c: set(v) for c, v in unseen_by_cut.items()}
    hits = {c: {"questions_with_unseen_option": 0,
                "questions_with_unseen_answer": 0,
                "n_unseen_labels": len(v)}
            for c, v in cuts.items()}

    def on_member(_dataset: str, _index: int, _row: dict, q: dict) -> None:
        texts = {o.get("text", "") for o in q.get("options", [])}
        answer_text = ""
        for o in q.get("options", []):
            if o.get("id") == q.get("answer"):
                answer_text = o.get("text", "")
                break
        for cut, labset in cuts.items():
            if texts & labset:
                hits[cut]["questions_with_unseen_option"] += 1
            if answer_text and answer_text in labset:
                hits[cut]["questions_with_unseen_answer"] += 1

    mixmod.realise(spec, root, on_member=on_member)
    return hits


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=mixmod.PREFETCH_DIR)
    args = ap.parse_args(argv)
    inv = inventory_of(args.manifest, args.root)
    if not inv["selection_match"]:
        print(f"REFUSED: realised {inv['realised_members_sha256']} != "
              f"manifest {inv['manifest_members_sha256']}")
        return 1
    with open(args.out, "w") as fh:
        json.dump(inv, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    print(f"espacios={inv['n_espacios']} shared={inv['n_shared']} "
          f"per_row={inv['n_per_row_spaces']} "
          f"fraccion_shared={inv['fraccion_shared']} "
          f"veredicto={inv['veredicto']} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
