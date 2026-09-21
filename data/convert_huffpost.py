"""Convert HuffPost raw snapshot -> universal schema JSONL (stdlib only).

Source: HF hub snapshot of khalidalt/HuffPost (legacy script, datasets>=3
refuses it): News_Category_Dataset_v2.json, 200853 lines, no split column.
Assigns deterministic 90/10 train/test by GROUP (skeleton+domain+
language, #T-split-domain) so the baseline trainer has an eval split that
no template straddles. Reuses data.adapters.adapt_huffpost (seed=0).

Usage: python3 data/convert_huffpost.py
  in : $HF_HUB/datasets--khalidalt--HuffPost/snapshots/<rev>/News_Category_Dataset_v2.json
  out: artifacts/data-prefetch/huffpost.jsonl
"""

import dataclasses
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import data.adapters as A  # noqa: E402
from eval.splits import group_split_of  # noqa: E402

SNAP = Path(os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--khalidalt--HuffPost"
    "/snapshots/01020533529fc1cda0af7d99231eb96e7837f883"
    "/News_Category_Dataset_v2.json"))
OUT = ROOT / "artifacts" / "data-prefetch" / "huffpost.jsonl"


def split_of(headline: str) -> str:
    """90/10 by GROUP (skeleton+domain+language), never by row (#T-split-domain).

    The previous hash took `headline + str(i)`, i.e. the ROW INDEX, so two
    near-identical headlines could land on opposite sides of the fence.
    """
    return group_split_of(headline, domain="huffpost", test_frac=0.1)


def main():
    n_rows, n_ex, skipped = 0, 0, 0
    with open(SNAP) as fin, open(OUT, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                cat = str(r.get("category", "")).upper()
                if cat not in A.HUFFPOST_UNIVERSE:
                    skipped += 1
                    continue
                row = {"headline": r.get("headline", ""),
                       "description": r.get("short_description",
                                            r.get("description", "")),
                       "category": cat,
                       "split": split_of(r.get("headline", ""))}
                for ex in A.adapt_huffpost([row], seed=0):
                    ex.split = row["split"]
                    fout.write(json.dumps(dataclasses.asdict(ex),
                                          ensure_ascii=False) + "\n")
                    n_ex += 1
                n_rows += 1
            except Exception:
                skipped += 1
    print(json.dumps({"job": "huffpost", "status": "converted",
                      "rows": n_rows, "examples": n_ex, "skipped": skipped,
                      "out": str(OUT)}))


if __name__ == "__main__":
    main()
