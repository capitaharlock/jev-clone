"""The label space the SAMPLER actually draws from (#T-labelspace-div).

`data.labelspace` counts the label spaces a mixture holds ON DISK, and on
that count `decision-mix-clean-1m` is poor (9 shared taxonomies over 83 %
of 1 M rows) while `decision-mix-labeldiv-1m` is rich (20 000 miniature
taxonomies). Neither number is what reaches the model.

`data.optset.OptionSetSampler.compose()` draws every distractor from
`self.pools[dataset]`, and `label_pool()` builds that pool keyed on the
option **id** over the whole corpus. So the label space a training sample
is actually scored against is the dataset's UNION pool, and a corpus whose
rows carry their own option set loses that structure on the way in:

* `swag` ids are positions (o0..o3): the pool is four texts taken from
  whichever row defined each id first, and every one of the 73 546
  selected members is shown those same four continuations — none of which
  belong to its own state. Same shape for `prog-gold` (10 ids over 7 588
  texts) and `synth-v1` (7 over 19 423).
* `episodic-div` mints globally-unique pseudoword labels precisely so its
  20 000 spaces cannot merge — and they merge anyway, into ONE pool of
  99 544 labels, because the pool is per dataset. Measured here: a
  sampled row is offered two distractors from two other taxonomies.

The second consequence is a cost, and it is the one that stops the curve:
`DistractorIndex.ranked()` scores the whole pool once per distinct gold
("a pool is at most a few hundred short strings", says its docstring). At
99 544 labels that is 15.7 s per distinct gold and ~99 544 distinct golds
— about 430 CPU-hours before a single epoch, and a cache of 10^10 ids.

The fix is `OptionSetSampler(space_scoped=...)`: the row's own option set
becomes its label space, distractors stay inside it and each index is six
labels wide. This module measures both sides of that switch and publishes
`artifacts/gates/T-labelspace-div/sampler-scope.json`.

CLI::

    python3 -m data.spacescope gate                # writes the gate json
    python3 -m data.spacescope gate --rows 500     # bigger leak probe
"""
from __future__ import annotations

import json
import os
import sys
import time

from data.mix import ROOT, SOURCES
from data.optset import (PREFETCH_DIR, DistractorIndex, OptionSetSampler,
                         SamplerConfig, iter_rows)

TASK = "T-labelspace-div"
GATE_DIR = os.path.join(ROOT, "artifacts", "gates", TASK)
GATE_PATH = os.path.join(GATE_DIR, "sampler-scope.json")

#: the corpora whose rows carry their own option set. Measured, not
#: asserted: `scan_dataset` reports `option_texts` against `pool_ids` and a
#: dataset is per-row exactly when the first exceeds the second.
PER_ROW_CANDIDATES = ("swag", "prog-gold", "synth-v1", "episodic-div")
#: the reference the task's 10x floor is measured against: the label spaces
#: `decision-mix-clean-1m` offers the sampler, which is one per dataset
CLEAN_SAMPLER_SPACES = 12
#: the task's acceptance for the alternative mixture
RATIO_FLOOR = 10.0
#: rows per dataset in the leak probe; the point is the leak RATE, and it
#: is visible in the first handful of rows
DEFAULT_PROBE_ROWS = 200
#: difficulty computations the UNSCOPED probe is allowed to spend. It has
#: to be a budget and not a row count: unscoped, one distinct gold costs a
#: pass over the whole pool, so 200 rows of `episodic-div` is 200 x 99 544
#: scorings — 52 minutes to measure a leak the first two rows already show.
#: That cost IS the finding, and `ranking_cost()` publishes it directly.
PROBE_SCORE_BUDGET = 400_000


def space_key(question: dict) -> str:
    """The identity of one question's own option set."""
    return "\x00".join(sorted(o.get("id", "")
                              for o in question.get("options", [])))


def scan_dataset(dataset: str, root: str = PREFETCH_DIR,
                 limit: int | None = None) -> dict:
    """The dataset's global pool and the row spaces it is made of.

    The pool is built exactly as `data.optset.label_pool` builds it —
    first text wins per id — because the point is to report what the
    sampler WOULD draw from, not a better pool this module invented.
    """
    pool: dict = {}
    texts: set = set()
    spaces: set = set()
    rows = 0
    questions = 0
    for row in iter_rows(dataset, split=None, limit=limit, root=root):
        rows += 1
        for question in row.get("questions", []):
            questions += 1
            spaces.add(space_key(question))
            for opt in question.get("options", []):
                pool.setdefault(opt["id"], opt.get("text", opt["id"]))
                texts.add(opt.get("text", opt["id"]))
    return {
        "dataset": dataset,
        "rows": rows,
        "questions": questions,
        "pool_ids": len(pool),
        "option_texts": len(texts),
        "row_spaces": len(spaces),
        "per_row": len(texts) > len(pool),
        "pool": [{"id": i, "text": pool[i]} for i in sorted(pool)],
    }


def leak_probe(dataset: str, pool: list, scoped: bool,
               rows: int = DEFAULT_PROBE_ROWS,
               root: str = PREFETCH_DIR, seed: int = 1) -> dict:
    """Do the emitted options stay inside the row's own option set?

    The one claim that matters about a mixture of many small spaces: a
    sample whose distractors come from another taxonomy is not a sample of
    that space. `pool` is passed in so the probe does not re-read the
    corpus to rediscover a pool the caller already scanned.
    """
    if not scoped:
        rows = max(2, min(rows, PROBE_SCORE_BUDGET // max(len(pool), 1)))
    sampler = OptionSetSampler(
        [dataset], SamplerConfig(seed=seed), rows_per_dataset=rows,
        root=root, pools={dataset: pool},
        space_scoped=(dataset,) if scoped else ())
    own = {}
    for n, row in enumerate(sampler.rows[dataset]):
        for question in row.get("questions", []):
            own[(f"{dataset}-{n}", question.get("id", ""))] = {
                o.get("id", "") for o in question.get("options", [])}
    t0 = time.perf_counter()
    samples = list(sampler.epoch(0))
    elapsed = time.perf_counter() - t0
    leaked = foreign = 0
    for sample in samples:
        inside = own.get((sample.row_id, sample.question_id), set())
        outside = [i for i in sample.option_ids() if i not in inside]
        if outside:
            leaked += 1
            foreign += len(outside)
    return {
        "scoped": scoped,
        "probe_rows": rows,
        "samples": len(samples),
        "samples_with_a_foreign_option": leaked,
        "foreign_options": foreign,
        "leak_rate": round(leaked / max(len(samples), 1), 4),
        "spaces_drawn_from": sampler.spaces_seen(dataset),
        "seconds": round(elapsed, 2),
        "seconds_per_1k_samples": round(
            1000 * elapsed / max(len(samples), 1), 2),
    }


def ranking_cost(pool: list, config: SamplerConfig | None = None) -> dict:
    """What one `DistractorIndex.ranked()` costs over a pool this size.

    The index caches per gold, so the corpus pays this once per DISTINCT
    gold — which for a pool of unique labels is once per label.
    """
    index = DistractorIndex(pool, config or SamplerConfig(seed=1))
    gold = pool[0]["id"] if pool else ""
    t0 = time.perf_counter()
    index.ranked(gold)
    seconds = time.perf_counter() - t0
    return {
        "pool": len(pool),
        "seconds_per_distinct_gold": round(seconds, 3),
        "distinct_golds_worst_case": len(pool),
        "hours_to_rank_every_gold": round(seconds * len(pool) / 3600, 1),
    }


def measure(datasets=PER_ROW_CANDIDATES, root: str = PREFETCH_DIR,
            rows: int = DEFAULT_PROBE_ROWS, time_ranking: bool = True) -> dict:
    """Both sides of the switch, per corpus that carries its own options."""
    out = {}
    for dataset in datasets:
        if dataset not in SOURCES or not any(
                os.path.exists(p) for p in SOURCES[dataset].paths(root)):
            continue
        scan = scan_dataset(dataset, root)
        pool = scan.pop("pool")
        entry = {"scan": scan,
                 "unscoped": leak_probe(dataset, pool, False, rows, root),
                 "scoped": leak_probe(dataset, pool, True, rows, root)}
        if time_ranking:
            entry["unscoped_ranking_cost"] = ranking_cost(pool)
        entry["sampler_spaces"] = {
            "unscoped": 1,
            "scoped": scan["row_spaces"],
            "on_disk": scan["row_spaces"],
        }
        out[dataset] = entry
    return out


def ratio_block(measured: dict, base: int = CLEAN_SAMPLER_SPACES) -> dict:
    """The task's 10x floor, measured where the trainer reads.

    A count of spaces on disk cannot satisfy it on its own: the mixture
    only offers what the sampler draws from, so the ratio is reported for
    both settings and only the scoped one is allowed to pass.
    """
    alt = max((m["sampler_spaces"]["scoped"] for m in measured.values()),
              default=0)
    unscoped = 1 if measured else 0
    return {
        "clean_1m_sampler_spaces": base,
        "alternative_sampler_spaces_scoped": alt,
        "alternative_sampler_spaces_unscoped": unscoped,
        "ratio_scoped": round(alt / max(base, 1), 2),
        "ratio_unscoped": round(unscoped / max(base, 1), 4),
        "floor": RATIO_FLOOR,
        "pass_scoped": alt >= RATIO_FLOOR * base,
        "pass_unscoped": unscoped >= RATIO_FLOOR * base,
        "why": ("the alternative mixture only carries its label spaces if "
                "the sampler is told to scope its pool per row; unscoped, "
                "every taxonomy it mints merges into one dataset pool and "
                "the mixture is LESS diverse than the corpus it replaces"),
    }


def gate(root: str = PREFETCH_DIR, rows: int = DEFAULT_PROBE_ROWS,
         write: bool = True, time_ranking: bool = True) -> dict:
    t0 = time.perf_counter()
    measured = measure(root=root, rows=rows, time_ranking=time_ranking)
    ratio = ratio_block(measured)
    checks = {
        "scoped_sampler_never_leaves_the_row_space": all(
            m["scoped"]["samples_with_a_foreign_option"] == 0
            for m in measured.values()),
        "unscoped_sampler_does_leave_it": any(
            m["unscoped"]["samples_with_a_foreign_option"] > 0
            for m in measured.values()),
        "ratio_at_or_over_floor_when_scoped": ratio["pass_scoped"],
    }
    out = {
        "task": TASK,
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "pass": all(checks.values()),
        "checks": checks,
        "ratio": ratio,
        "per_dataset": measured,
        "fix": {
            "what": "data.optset.OptionSetSampler(space_scoped=(...))",
            "default": "empty — no existing run changes",
            "wiring_for_the_curve": (
                "the 62 k / 250 k / 1 M half of this task must pass "
                "space_scoped=('episodic-div',) into its samplers, or it "
                "trains one 99 544-label space and measures the wrong "
                "corpus"),
        },
        "method": {
            "probe_rows_per_dataset": rows,
            "leak": ("a sample leaks when an option id it offers is not in "
                     "its own row's option set"),
            "reproduce": f"python3 -m data.spacescope gate --rows {rows}",
        },
    }
    if write:
        os.makedirs(GATE_DIR, exist_ok=True)
        with open(GATE_PATH, "w") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
            fh.write("\n")
    return out


def main(argv: list) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python3 -m data.spacescope")
    sub = ap.add_subparsers(dest="cmd")
    g = sub.add_parser("gate")
    g.add_argument("--rows", type=int, default=DEFAULT_PROBE_ROWS)
    g.add_argument("--no-write", action="store_true")
    g.add_argument("--no-timing", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd != "gate":
        ap.print_help()
        return 2
    out = gate(rows=args.rows, write=not args.no_write,
               time_ranking=not args.no_timing)
    for dataset, m in sorted(out["per_dataset"].items()):
        print(f"{dataset:14s} pool {m['scan']['pool_ids']:6d} ids / "
              f"{m['scan']['option_texts']:7d} texts · "
              f"{m['scan']['row_spaces']:7d} row spaces · "
              f"leak {m['unscoped']['leak_rate']:.0%} unscoped -> "
              f"{m['scoped']['leak_rate']:.0%} scoped")
    r = out["ratio"]
    print(f"ratio          {r['ratio_scoped']}x scoped (pass="
          f"{r['pass_scoped']}) vs {r['ratio_unscoped']}x unscoped "
          f"(pass={r['pass_unscoped']}), floor {r['floor']}x")
    print(f"gate           pass={out['pass']} -> "
          f"{os.path.relpath(GATE_PATH, ROOT)}")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
