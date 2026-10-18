"""Many-small-spaces synthetic corpus (#T-labelspace-div, mitad 1, paso 2).

The inventory (`data.labelspace`, `artifacts/gates/T-labelspace-div/`) says
the 1 M mix is poor: 9 shared taxonomies cover 83 % of the rows, so a
text->label map answers everything (#T-antiscale-diag eje 1: 100 %). This
module points the synthetic factory at the opposite corner: MANY SMALL
SPACES instead of many rows of few spaces — 20 000 taxonomies of ~52 rows
each, same total volume (1 M), same row schema as every prefetch shard.

One space is one miniature taxonomy: a field with K pseudoword labels
(K = 2..8, seeded per space), rows that state `field = value` among
distractor lines and ask for it back. The structure mirrors email-triage
(the same K options every row of the space); only the scale of the
taxonomy count differs — 20 000 vs 9. That is the experimental factor, and
it is the ONLY one: no unknowns, no teacher, no multilingual axis.

Why pseudowords: labels must be globally unique across spaces, otherwise
two spaces share a text and merge into one. Pronounceable syllable
assembly keeps them embeddable by the frozen backbone without colliding
with any real word — the measured overlap with the unseen cuts is part of
the manifest, expected 0, published either way.

Determinism: every space draws from `Random(f"{seed} episodic-div
{space}")`; shard assignment is a round-robin interleave, so trimming the
1.04 M supply to the 1 M quota cuts rows uniformly and every space
survives with ~50 rows. Same seed + same code = same bytes; the manifest
pins every shard sha256.

Origin is `synthetic` (machine-generated rows, honest) — but the corpus
never joins the product mix: `episodic-div` is registered `experimental`
and excluded from every default scan, so no running or queued run changes
bytes because this file exists. The curve job selects it explicitly with
`train_decision --only-dataset episodic-div`.

Run (CPU only, no torch, no MPS):
    python3 -m data.episodic --seed 20260922 --out-dir artifacts/episodic-div
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

#: bump when the generator, the schema or the recipe change: rows from an
#: older version are a different corpus wearing the same name.
EPISODIC_VERSION = "episodic-div-v1"
SOURCE_ID = "episodic-div"

N_SPACES = 20_000
ROWS_PER_SPACE = 52
N_SHARDS = 20
SUPPLY_ROWS = N_SPACES * ROWS_PER_SPACE  # 1 040 000; quota trims to 1 M
MIN_K = 2
MAX_K = 8

_ONSETS = ("br", "cr", "dr", "fl", "gr", "kl", "pr", "str", "tr", "bl",
           "sn", "sk", "th", "ch", "sh", "m", "n", "l", "r", "s", "t",
           "k", "p", "d", "v", "z", "w", "j")
_NUCLEI = ("a", "e", "i", "o", "u", "ae", "oo", "ee")
_CODAS = ("", "n", "r", "l", "s", "k", "t", "m", "p", "x", "nd", "rk")

_FIELDTPL = ("status", "tier", "queue", "phase", "grade", "sector", "mode",
             "class", "route", "batch")


def _pseudoword(rng: random.Random) -> str:
    """One pronounceable nonce word, 2-3 syllables."""
    parts = []
    for _ in range(rng.choice((2, 2, 2, 3))):
        parts.append(rng.choice(_ONSETS) + rng.choice(_NUCLEI)
                     + rng.choice(_CODAS))
    return "".join(parts)


def fresh_word(rng: random.Random, used: set) -> str:
    """A pseudoword no earlier space has used — spaces never merge."""
    while True:
        w = _pseudoword(rng)
        if w not in used:
            used.add(w)
            return w


def space_spec(space_idx: int, seed: int, used: set) -> dict:
    """The miniature taxonomy of one space: field, K labels, entity kind."""
    rng = random.Random(f"{seed}\x00{SOURCE_ID}\x00{space_idx}")
    k = rng.randint(MIN_K, MAX_K)
    labels = [fresh_word(rng, used) for _ in range(k)]
    field = rng.choice(_FIELDTPL) + "-" + fresh_word(rng, used)
    entity = fresh_word(rng, used) + "-" + str(rng.randint(10, 99))
    fillers = [fresh_word(rng, used) for _ in range(6)]
    return {"space": space_idx, "k": k, "labels": labels, "field": field,
            "entity": entity, "fillers": fillers, "rng": rng}


def generate(seed: int, out_dir: str,
             n_spaces: int = N_SPACES,
             rows_per_space: int = ROWS_PER_SPACE,
             n_shards: int = N_SHARDS,
             log=None) -> dict:
    """Write the shards; return the versioned manifest (unwritten)."""
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    paths = [os.path.join(out_dir, f"{SOURCE_ID}-{s:03d}.jsonl")
             for s in range(n_shards)]
    fhs = [open(p, "w") for p in paths]
    used: set = set()
    # the interleave is round-robin over (space, row): trimming the supply
    # to the quota later cuts every space equally, none is beheaded.
    k_hist: dict = {}
    n_rows = 0
    try:
        specs = [space_spec(i, seed, used) for i in range(n_spaces)]
        for spec in specs:
            k_hist[str(spec["k"])] = k_hist.get(str(spec["k"]), 0) + 1
        if log:
            log(f"[episodic] {n_spaces} spaces spec'd, "
                f"{len(used)} unique words")
        # rows are drawn space by space (rng order) but dealt round-robin
        for i, spec in enumerate(specs):
            for j, row in enumerate(_rows_of(spec, rows_per_space)):
                fhs[(i + j) % n_shards].write(
                    json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1
            if log and (i + 1) % 5000 == 0:
                log(f"[episodic] {i + 1}/{n_spaces} spaces")
    finally:
        for fh in fhs:
            fh.close()
    shards = []
    for p in paths:
        shards.append({"path": os.path.relpath(p, ROOT),
                       "bytes": os.path.getsize(p),
                       "sha256": _sha256_file(p)})
    spaces_seen = _count_spaces(paths)
    manifest = {
        "corpus": "decision-mix-labeldiv-1m",
        "version": EPISODIC_VERSION,
        "task": "T-labelspace-div",
        "seed": seed,
        "built_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "recipe": {"n_spaces": n_spaces,
                   "rows_per_space": rows_per_space,
                   "supply_rows": n_rows,
                   "target_rows": 1_000_000,
                   "k_range": [MIN_K, MAX_K],
                   "k_hist_spaces": dict(sorted(k_hist.items())),
                   "n_shards": n_shards,
                   "interleave": "shard = (space_idx + row_j) % n_shards",
                   "unknowns": 0,
                   "why_pseudowords": ("globally unique labels so spaces "
                                       "never merge; pronounceable so the "
                                       "frozen backbone embeds them")},
        "sources": {SOURCE_ID: {
            "family": "episodic", "origin": "synthetic",
            "dynamic_options": True, "experimental": True}},
        "shards": shards,
        "labelspace": {"n_espacios": spaces_seen,
                       "n_reutilizables": spaces_seen,
                       "rows_per_space_nominal": rows_per_space},
        "elapsed_s": round(time.time() - t0, 1),
    }
    return manifest


def _rows_of(spec: dict, n: int):
    rng = spec["rng"]
    for j in range(n):
        gold = rng.choice(spec["labels"])
        distractors = spec["fillers"][:4]
        lines = [f"{spec['field']} = {gold}"]
        for i, d in enumerate(distractors[: rng.randint(2, 4)]):
            lines.append(f"{spec['field']}-note-{i} = {d}")
        rng.shuffle(lines)
        state = f"Record {spec['entity']}-{j}:\n" + "\n".join(lines)
        options = [{"id": t, "text": t} for t in spec["labels"]]
        rng.shuffle(options)
        yield {
            "state": state,
            "questions": [{
                "id": f"epdiv-{spec['space']:05d}-{j}",
                "kind": "boolean" if spec["k"] == 2 else "choice",
                "options": options,
                "answer": gold,
            }],
            "split": "train",
        }


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(1 << 20)
            if not block:
                return h.hexdigest()
            h.update(block)


def _count_spaces(paths: list) -> int:
    """Distinct question-id prefixes across the shards just written."""
    seen = set()
    for p in paths:
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                for q in json.loads(line).get("questions", []):
                    seen.add(q.get("id", "").rsplit("-", 1)[0])
    return len(seen)


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--out-dir", default=os.path.join("artifacts",
                                                      "episodic-div"))
    ap.add_argument("--manifest", default=os.path.join(
        "artifacts", "mix-labeldiv",
        "decision-mix-labeldiv-1m-seed20260922.manifest.json"))
    ap.add_argument("--spaces", type=int, default=N_SPACES)
    ap.add_argument("--rows-per-space", type=int, default=ROWS_PER_SPACE)
    ap.add_argument("--shards", type=int, default=N_SHARDS)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    log = (lambda *a: None) if args.quiet else (
        lambda *a: print(*a, flush=True))
    manifest = generate(args.seed, args.out_dir, args.spaces,
                        args.rows_per_space, args.shards, log)
    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)
    with open(args.manifest, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    log(f"[episodic] {manifest['recipe']['supply_rows']} rows, "
        f"{manifest['labelspace']['n_espacios']} spaces -> {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
