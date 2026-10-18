"""How much of the decision model's accuracy is surface matching
(#T-data-eval).

The 1 M checkpoints score 0.94 on civil-comments and 0.97 on email-triage
while swag sits at 0.255 against a 0.25 chance and synth-v1 at 0.230. That
pairing only has two readings: either the reasoning sets are harder, or the
global number is carried by tasks a text -> label map already solves. This
module measures which, on the SAME eval rows the trainer's stage eval uses,
and publishes both halves next to each other.

Three predictors over one row, always the options-only cut (argmax over the
K real options; `unknown` is dropped, because a row where the model
abstains says nothing about whether the answer was findable):

* **model** — the trained pointer head of the checkpoint under test.
* **lexical** — parameter-free: pick the option whose character-3-gram set
  is most contained in the state's. Char 3-grams because that is the unit
  `data.hardneg` already hashes; it is the strongest predictor you can
  build out of surface overlap alone, with no training and no parameters.
* **chance** — 1/K averaged over the rows, the sampler's own floor.

The split of the corpus itself is stated, not inferred: `LABEL_MAPPING` are
the sets whose answer is a label out of a task-global vocabulary (a map
from text to label is enough); `COMPOSITIONAL` are the sets whose answer
only exists relative to THIS row's options — the continuation that follows
(swag), the hypothesis the premise entails (snli), the grounded answer
(synth-v1), the fact the graph holds (prog-gold). Accuracy is then
recomposed at the 1 M mixture's own shares, so "how much of the global
number" is a weighted sum and not an impression.

CLI:
    .venv-train/bin/python -m tools.diagnose.surface \
        --checkpoint artifacts/checkpoints/decision/<run>/stage-001000000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.optset import SamplerConfig, iter_rows, label_pool  # noqa: E402
from training.python import train_decision as T  # noqa: E402

TASK = "T-data-eval"
OUT_DIR = os.path.join(ROOT, "artifacts", "gates", TASK)
OUT_PATH = os.path.join(OUT_DIR, "surface_vs_reasoning.json")

#: a dataset is put in the "closed vocabulary" group when most of its
#: option TEXTS repeat across rows — the same string is offered to many
#: states, so a text -> label map is enough to answer. Below the threshold
#: the options are a per-row span (swag's four continuations) and no map
#: over the label space can be learned. The axis is MEASURED per dataset
#: (`option_text_reuse`), never declared in a list here.
REUSE_THRESHOLD = 0.5

DEFAULT_ROWS = 600

#: `train_decision.EVAL_SPLIT` has no entry for swag, so its eval sampler
#: defaults to "test" — and the converted swag corpus holds only `train`
#: and `calibration`. The consequence is measured, not argued: swag is
#: 7.35 % of the 1 M mixture and it is ABSENT from every `per_dataset`
#: block of every stage eval. It is scored here off its calibration split,
#: which the decision runs never train on, and the substitution is stated
#: in the report rather than hidden in a default.
SPLIT_OVERRIDE = {"swag": "calibration"}


def trigrams(text: str) -> set:
    t = " " + " ".join(text.lower().split()) + " "
    return {t[i:i + 3] for i in range(max(0, len(t) - 2))}


def lexical_choice(sample) -> int:
    """Containment of each option's char-3-grams in the state's."""
    hay = trigrams(sample.state + " " + sample.question)
    best, best_score = 0, -1.0
    for i, opt in enumerate(sample.options):
        g = trigrams(opt["text"])
        score = len(g & hay) / len(g) if g else 0.0
        if score > best_score:
            best, best_score = i, score
    return best


def option_text_integrity(dataset: str, split: str,
                          max_rows: int = 5000) -> dict:
    """Does the sampler's global pool still carry each ROW's option texts?

    `data.optset.label_pool` keys the pool on the option ID and keeps the
    FIRST text it sees for that id (`seen.setdefault`). That is correct
    for a semantic id — snli's `entailment` is the same string in every
    row — and wrong for a POSITIONAL one: swag's `o0..o3` name this row's
    four continuations, so the pool freezes row 0's and every later row is
    offered them instead of its own. This counts how often that happens,
    per dataset, on the rows an eval sampler reads.
    """
    pool = {o["id"]: o["text"] for o in label_pool(dataset)}
    rows = opts = mismatched = rows_hit = gold_hit = 0
    for row in iter_rows(dataset, split=split):
        rows += 1
        hit = False
        for q in row.get("questions", []):
            gold = q.get("answer")
            for opt in q.get("options", []):
                opts += 1
                if pool.get(opt["id"]) != opt.get("text", opt["id"]):
                    mismatched += 1
                    hit = True
                    gold_hit += int(opt["id"] == gold)
        rows_hit += int(hit)
        if rows >= max_rows:
            break
    if not opts:
        return {"rows": 0}
    return {
        "rows": rows,
        "options": opts,
        "options_whose_text_the_pool_replaces": mismatched,
        "option_text_mismatch_rate": round(mismatched / opts, 6),
        "rows_with_any_mismatch_rate": round(rows_hit / rows, 6),
        "gold_options_replaced": gold_hit,
        "gold_text_replaced_rate": round(gold_hit / rows, 6),
    }


def score_dataset(engine, sampler, dataset: str, rows: int,
                  batch_size: int, max_length: int) -> dict:
    """Model / lexical / chance on the same rows, options-only."""
    import torch

    n = model_ok = lex_ok = both_ok = model_only = 0
    chance = 0.0
    seen_texts: dict = {}
    engine.head.eval()
    with torch.no_grad():
        for batch in T.MixtureStream({dataset: sampler}, batch_size,
                                     verify=False).epoch(0):
            tokens, mask, _ = T.encode_states(
                engine.backbone, [s.state for s in batch], max_length)
            embs, spans = T.batch_embeddings(engine, batch)
            for i, sample in enumerate(batch):
                if sample.gold_index >= sample.k:   # an `unknown` row
                    continue
                logits = T.row_logits(engine, tokens, mask, embs, spans, i)
                pred = int(torch.argmax(logits[:sample.k]))
                lex = lexical_choice(sample)
                n += 1
                chance += 1.0 / sample.k
                for opt in sample.options:
                    seen_texts[opt["text"]] = seen_texts.get(
                        opt["text"], 0) + 1
                m_hit, l_hit = pred == sample.gold_index, \
                    lex == sample.gold_index
                model_ok += m_hit
                lex_ok += l_hit
                both_ok += m_hit and l_hit
                model_only += m_hit and not l_hit
            if n >= rows:
                break
    if not n:
        return {"n": 0}
    slots = sum(seen_texts.values())
    return {
        "n": n,
        "option_text_reuse": round(
            1.0 - len(seen_texts) / slots, 6) if slots else 0.0,
        "distinct_option_texts": len(seen_texts),
        "option_slots": slots,
        "chance": round(chance / n, 6),
        "model": round(model_ok / n, 6),
        "lexical": round(lex_ok / n, 6),
        "model_and_lexical": round(both_ok / n, 6),
        "model_not_lexical": round(model_only / n, 6),
        "model_lift_over_chance": round((model_ok - chance) / n, 6),
        "model_lift_over_lexical": round((model_ok - lex_ok) / n, 6),
    }


def group_of(report: dict) -> str:
    """Which side of the measured reuse threshold this dataset falls on."""
    return ("closed_vocabulary"
            if report.get("option_text_reuse", 0.0) >= REUSE_THRESHOLD
            else "per_row_options")


def recompose(per_dataset: dict, shares: dict) -> dict:
    """Weigh each dataset's accuracy at its share of the 1 M mixture."""
    total = sum(shares.get(d, 0) for d in per_dataset if per_dataset[d]["n"])
    out = {"covered_share_of_mixture": 0.0, "by_group": {}}
    if not total:
        return out
    groups: dict = {"closed_vocabulary": [], "per_row_options": []}
    for d, r in per_dataset.items():
        if r["n"]:
            groups[group_of(r)].append(d)
    for name, members in groups.items():
        w = sum(shares.get(d, 0) for d in members
                if per_dataset.get(d, {}).get("n"))
        if not w:
            continue
        def wsum(key):
            return sum(shares.get(d, 0) * per_dataset[d][key]
                       for d in members if per_dataset.get(d, {}).get("n"))
        out["by_group"][name] = {
            "datasets": [d for d in members
                         if per_dataset.get(d, {}).get("n")],
            "share_of_mixture": round(w / total, 6),
            "model": round(wsum("model") / w, 6),
            "lexical": round(wsum("lexical") / w, 6),
            "chance": round(wsum("chance") / w, 6),
            "contribution_to_global_model": round(wsum("model") / total, 6),
            "contribution_to_global_lift": round(
                (wsum("model") - wsum("chance")) / total, 6),
        }
    out["covered_share_of_mixture"] = round(total / sum(shares.values()), 6)
    out["global"] = {
        "model": round(sum(shares.get(d, 0) * r["model"]
                           for d, r in per_dataset.items() if r["n"]) / total, 6),
        "lexical": round(sum(shares.get(d, 0) * r["lexical"]
                             for d, r in per_dataset.items() if r["n"]) / total, 6),
        "chance": round(sum(shares.get(d, 0) * r["chance"]
                            for d, r in per_dataset.items() if r["n"]) / total, 6),
    }
    return out


def load_run_holdout(path: str) -> dict:
    """The carve the run ACTUALLY trained against, read back off disk.

    `T.build_holdout()` defaults to the four P0 datasets; the 1 M runs
    carry all twelve, and re-deriving the split here could disagree with
    the one the weights were trained under. `run.json` names its
    `holdout.json`, so that file is the source of truth.
    """
    with open(path) as fh:
        report = json.load(fh)
    return {d: T.DatasetHoldout(d, h["seen"], h["unseen"],
                                {i: i for i in h["seen"] + h["unseen"]},
                                exempt=h.get("exempt"))
            for d, h in report["per_dataset"].items()}


def run(ckpt_dir: str, rows: int = DEFAULT_ROWS, device: str = "auto",
        batch_size: int = 32, holdout_path: str | None = None,
        write: bool = True) -> dict:
    engine, manifest = T.load_checkpoint(ckpt_dir, device)
    cfg = SamplerConfig()
    holdout = (load_run_holdout(holdout_path) if holdout_path
               else T.build_holdout())
    ev_cfg = SamplerConfig(**{**cfg.__dict__, "unknown_fraction": 0.0})
    samplers = {}
    for d, h in sorted(holdout.items()):
        keep = set(h.seen)
        if len(keep) < 2:
            continue
        split = SPLIT_OVERRIDE.get(d, T.EVAL_SPLIT.get(d, "test"))
        samplers[d] = T.make_sampler(d, keep, split, ev_cfg,
                                     T.EVAL_ROWS_CAP)
    t0 = time.perf_counter()
    per_dataset = {d: score_dataset(engine, s, d, rows, batch_size,
                                    T.TRAIN_MAX_LENGTH)
                   for d, s in sorted(samplers.items())}
    integrity = {d: option_text_integrity(
        d, SPLIT_OVERRIDE.get(d, T.EVAL_SPLIT.get(d, "test")))
        for d in sorted(samplers)}
    mix_path = os.path.join(ROOT, "artifacts", "mix-1m",
                            "decision-mix-clean-1m-seed20260922.manifest.json")
    with open(mix_path) as fh:
        shares = json.load(fh)["composition"]["by_dataset"]
    report = {
        "format": 1,
        "task": TASK,
        "question": "how much of the global accuracy is surface matching",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": os.path.relpath(ckpt_dir, ROOT),
        "holdout": (os.path.relpath(holdout_path, ROOT) if holdout_path
                    else "training.python.train_decision.build_holdout()"),
        "model_version": manifest.get("model_version"),
        "backbone_frozen": manifest.get("backbone", {}).get("frozen"),
        "cut": "seen labels, eval split, options-only (unknown dropped)",
        "split_override": dict(SPLIT_OVERRIDE),
        "split_override_why": ("train_decision.EVAL_SPLIT has no swag entry "
                               "and the converted swag corpus has no test "
                               "split: swag is scored on `calibration`, "
                               "which no decision run trains on"),
        "predictors": {
            "model": "the trained pointer head of this checkpoint",
            "lexical": "argmax over options of char-3-gram containment in "
                       "the state — parameter-free, never trained",
            "chance": "mean 1/K over the scored rows",
        },
        "grouping": {
            "axis": "option_text_reuse = 1 - distinct option texts / option "
                    "slots, measured on the scored rows",
            "threshold": REUSE_THRESHOLD,
            "closed_vocabulary": "the same option strings recur across "
                                 "rows: a text -> label map answers them",
            "per_row_options": "the options are this row's own spans: no "
                               "map over a label space can answer them",
        },
        "rows_per_dataset": rows,
        "per_dataset": per_dataset,
        "option_text_integrity": integrity,
        "weighted_at_1m_shares": recompose(per_dataset, shares),
        "mix_shares": {d: shares[d] for d in sorted(shares)},
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    if write:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(OUT_PATH, "w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
            fh.write("\n")
    return report


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(prog="tools.diagnose.surface")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--holdout", default=None,
                    help="the run's holdout.json (all 12 mixture datasets); "
                         "defaults to the four-dataset P0 carve")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    rep = run(args.checkpoint, args.rows, args.device, args.batch_size,
              args.holdout, write=not args.no_write)
    print(json.dumps({"checkpoint": rep["checkpoint"],
                      "per_dataset": rep["per_dataset"],
                      "weighted": rep["weighted_at_1m_shares"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
