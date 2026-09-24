"""Development cut and reserved test cut — rule R7 of #T-eval-cardinality.

`eval/fullspace.py` scores the 3 080 official BANKING77 test rows, and the
phase-2 plan wanted to pick K, log-Q, the encoder, the option text, the
mixture and the seed by looking at that same number. Nothing in that loop
touches a gradient and it still turns the test set into a development set:
the choice that survives is the one that scored best ON THOSE ROWS, and its
margin is then reported as if the rows were fresh. That is selection on the
test set, and the only defence is not to look.

So there are two cuts, and they are not interchangeable:

* **dev** — `banking77`, the TRAIN split, 1 000 rows drawn by a fixed seed,
  sealed by sha256. Arms are chosen here, as often as needed. These rows are
  not in the checkpoint's training corpus either (`fence.clean_1m` keeps the
  whole banking77 dataset out), so the cut measures the same transfer the
  final cut does — it is a development sample of the same population, not a
  seen-label cut.
* **test** — `banking77`, the official TEST split, all 3 080 rows. This is
  the cut the outside world's numbers are quoted on, and it is RESERVED:
  every query to it is written down in `test-queries.json` with its date, its
  checkpoint and its reason, including the ones that were made before this
  ledger existed. The ledger is not a permission system — it cannot stop
  anybody — it is the record that makes adaptive selection visible when
  somebody reads the numbers later.

Neither cut is "the hard one": they are the same task, the same 77-label
space and the same chance rate. The difference is only who is allowed to
look, and how often.

CLI (stdlib only, no torch):

    python3 -m eval.cuts seal              # seal (or re-verify) both cuts
    python3 -m eval.cuts show              # what each cut is
    python3 -m eval.cuts ledger            # the test-cut query record
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.optset import PREFETCH_DIR, iter_rows  # noqa: E402
from eval import splits as S  # noqa: E402

TASK = "T-eval-cardinality"
GATE_DIR = ROOT / "artifacts" / "gates" / TASK
LEDGER_PATH = GATE_DIR / "test-queries.json"
SPLITS_DIR = S.SPLITS_DIR

#: fixed before the first measurement on the dev cut; changing it re-seals
#: the split loudly instead of quietly moving the fence
DEV_SEED = 20260924
DEV_ROWS = 1000


@dataclass(frozen=True)
class Cut:
    """One named cut of one dataset, at the FULL cardinality of its space."""

    name: str
    dataset: str
    split: str
    rows: int | None          # None = every row of the split
    seed: int | None          # None = no sampling, the split as it ships
    reserved: bool
    answers: str

    @property
    def split_name(self) -> str:
        return f"{TASK}-{self.name}"

    @property
    def manifest_path(self) -> Path:
        return SPLITS_DIR / self.split_name / "manifest.json"


DEV = Cut(name="banking77-dev", dataset="banking77", split="train",
          rows=DEV_ROWS, seed=DEV_SEED, reserved=False,
          answers="which arm to keep: K, option text, objective, encoder, "
                  "mixture, seed — as many times as it takes")
TEST = Cut(name="banking77-test", dataset="banking77", split="test",
           rows=None, seed=None, reserved=True,
           answers="what the model scores on the rows the outside world's "
                   "numbers are quoted on — read rarely, logged always")
CUTS = {"dev": DEV, "test": TEST}


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------- rows

def split_ids(cut: Cut, root: str = PREFETCH_DIR) -> list[str]:
    """Every sample id of the cut's split, in file order.

    The id is `dataset:dataset-<n>:<question id>`, where `n` is the position
    of the row WITHIN the split — the same enumeration
    `eval.fullspace.full_samples` uses, so an id from this list names the
    same sample the scorer builds.
    """
    out = []
    for n, row in enumerate(iter_rows(cut.dataset, cut.split, root=root)):
        for question in row.get("questions", []):
            out.append(f"{cut.dataset}:{cut.dataset}-{n}:"
                       f"{question.get('id', '')}")
    return out


def cut_ids(cut: Cut, root: str = PREFETCH_DIR) -> list[str]:
    """The ids this cut is made of — sampled by seed, then sorted.

    Sorted, so the file is a set and not an order: the sampler decides WHICH
    rows, never in what sequence they are scored.
    """
    ids = split_ids(cut, root)
    if cut.rows is None or cut.rows >= len(ids):
        return sorted(ids)
    rng = random.Random(f"{cut.seed}\x00{cut.name}")
    return sorted(rng.sample(ids, cut.rows))


# ------------------------------------------------------------------ seals

def seal(cut: Cut, root: str = PREFETCH_DIR, force: bool = False,
         splits_dir: Path | None = None) -> dict:
    """Write the cut's manifest, or re-verify the one already on disk.

    Sealed bytes are never silently regenerated: an existing manifest is
    recomputed and compared, and a disagreement raises instead of
    overwriting. A cut that re-seals itself quietly is how a fence moves
    without anybody noticing.
    """
    base = (splits_dir or SPLITS_DIR) / cut.split_name
    ids = cut_ids(cut, root)
    payload = "".join(f"{i}\n" for i in ids)
    digest = S.sha256_text(payload)
    source = Path(root) / f"{cut.dataset}.jsonl"
    body = {
        "name": cut.split_name,
        "task": TASK,
        "rule": "R7 — arms are chosen on the dev cut; the test cut is "
                "reserved and every query to it is logged",
        "cut": cut.name,
        "reserved": cut.reserved,
        "answers": cut.answers,
        "dataset": cut.dataset,
        "split": cut.split,
        "cardinality": "the full label space of the dataset, every row",
        "seed": cut.seed,
        "created_at": utcnow(),
        "source": {
            "path": str(source.relative_to(ROOT))
            if str(source).startswith(str(ROOT)) else str(source),
            "sha256": S.sha256_file(source),
            "rows_in_split": len(split_ids(cut, root)),
        },
        "counts": {"rows": len(ids)},
        "files": {f"{cut.name}.ids": {"rows": len(ids), "sha256": digest}},
    }
    if (base / "manifest.json").exists() and not force:
        old = json.loads((base / "manifest.json").read_text())
        recorded = old["files"][f"{cut.name}.ids"]["sha256"]
        if recorded != digest:
            raise ValueError(
                f"sealed cut {cut.split_name!r} disagrees with the rows "
                f"recomputed now ({recorded} != {digest}): the corpus or the "
                "seed moved. Re-seal on purpose (--force) or find out what "
                "changed")
        return old
    body["manifest_sha256"] = S.sha256_text(json.dumps(body, sort_keys=True))
    base.mkdir(parents=True, exist_ok=True)
    (base / f"{cut.name}.ids").write_text(payload, encoding="utf-8")
    (base / "manifest.json").write_text(
        json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body


def sealed(cut: Cut, splits_dir: Path | None = None,
           root: str = PREFETCH_DIR) -> dict:
    """The seal as published: `{manifest, manifest_sha256, ids}`.

    Git carries the manifest, never the `.ids` list (`.gitignore`: they are
    megabytes of row ids). So in a fresh clone the ids are recomputed from
    the corpus and CHECKED against the digest the manifest recorded — which
    is the same guarantee, since a seal is a digest and not a file.
    """
    base = (splits_dir or SPLITS_DIR) / cut.split_name
    manifest = json.loads((base / "manifest.json").read_text())
    digest = manifest["files"][f"{cut.name}.ids"]["sha256"]
    path = base / f"{cut.name}.ids"
    if path.exists():
        ids = [ln for ln in path.read_text().splitlines() if ln]
    else:
        ids = cut_ids(cut, root)
        if S.sha256_text("".join(f"{i}\n" for i in ids)) != digest:
            raise ValueError(
                f"{cut.split_name}: the ids recomputed from the corpus do "
                f"not hash to the sealed digest {digest}. The corpus moved "
                "under a sealed cut; do not score against it")
    return {"manifest": f"artifacts/splits/{cut.split_name}/manifest.json",
            "manifest_sha256": manifest["manifest_sha256"],
            "split_sha256": digest,
            "rows": len(ids), "ids": ids}


# ----------------------------------------------------------------- samples

def samples(cut: Cut, limit: int | None = None, root: str = PREFETCH_DIR,
            splits_dir: Path | None = None) -> list:
    """The cut's rows as `Sample`s, every label of the space an option."""
    from eval import fullspace as F

    keep = set(sealed(cut, splits_dir)["ids"])
    return F.full_samples(cut.dataset, limit or len(keep), split=cut.split,
                          keep=keep)


# ------------------------------------------------------------- the ledger

def ledger() -> dict:
    """The record of every query made to the reserved cut."""
    if LEDGER_PATH.exists():
        return json.loads(LEDGER_PATH.read_text())
    return {"format": 1, "task": TASK,
            "rule": "R7 — every read of the reserved cut is logged with its "
                    "date, its checkpoint and its reason. The ledger does "
                    "not grant or refuse access; it makes adaptive "
                    "selection visible to whoever reads the numbers later",
            "cut": TEST.split_name, "queries": []}


def record_query(checkpoint: str, reason: str, artifact: str = "",
                 rows: int | None = None, by: str = "",
                 when: str = "", reconstructed: bool = False,
                 write: bool = True) -> dict:
    """Append one query to the reserved cut. Never rewrites an old entry."""
    doc = ledger()
    entry = {
        "when": when or utcnow(),
        "checkpoint": checkpoint,
        "reason": reason,
        "artifact": artifact,
        "rows": rows,
        "by": by or "eval.scoreboard",
    }
    if reconstructed:
        entry["recorded"] = (
            "RECONSTRUCTED from the artifact on disk — this query was made "
            "before the ledger existed, so its reason is read off the task "
            "that produced it, not off a note written at the time")
    doc["queries"].append(entry)
    doc["updated_utc"] = utcnow()
    doc["n_queries"] = len(doc["queries"])
    if write:
        GATE_DIR.mkdir(parents=True, exist_ok=True)
        LEDGER_PATH.write_text(json.dumps(doc, indent=2,
                                          ensure_ascii=False) + "\n")
    return doc


def report(splits_dir: Path | None = None) -> dict:
    """Both cuts as the scoreboard publishes them."""
    out = {}
    for key, cut in CUTS.items():
        row = {"cut": cut.name, "dataset": cut.dataset, "split": cut.split,
               "reserved": cut.reserved, "answers": cut.answers,
               "seed": cut.seed}
        try:
            seal_doc = sealed(cut, splits_dir)
            row.update({k: v for k, v in seal_doc.items() if k != "ids"})
        except FileNotFoundError:
            row["sealed"] = False
            row["how"] = "python3 -m eval.cuts seal"
        out[key] = row
    out["queries_to_the_reserved_cut"] = ledger().get("queries", [])
    return out


# --------------------------------------------------------------------- CLI

def main(argv: list) -> int:
    ap = argparse.ArgumentParser(prog="eval.cuts")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seal", help="seal or re-verify both cuts")
    s.add_argument("--force", action="store_true")
    sub.add_parser("show", help="what each cut is")
    sub.add_parser("ledger", help="the reserved cut's query record")
    log = sub.add_parser("log", help="record a query to the reserved cut")
    log.add_argument("--checkpoint", required=True)
    log.add_argument("--reason", required=True)
    log.add_argument("--artifact", default="")
    log.add_argument("--by", default="")
    args = ap.parse_args(argv)

    if args.cmd == "seal":
        for cut in CUTS.values():
            man = seal(cut, force=args.force)
            print(f"[cuts] {cut.split_name}: {man['counts']['rows']} rows, "
                  f"sha256 {man['files'][cut.name + '.ids']['sha256'][:16]}…")
        return 0
    if args.cmd == "log":
        doc = record_query(args.checkpoint, args.reason, args.artifact,
                           by=args.by)
        print(json.dumps(doc["queries"][-1], indent=2, ensure_ascii=False))
        return 0
    if args.cmd == "ledger":
        print(json.dumps(ledger(), indent=2, ensure_ascii=False))
        return 0
    print(json.dumps(report(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
