"""Convert banking77 raw CSVs -> universal schema JSONL (stdlib only).

Source: artifacts/data-raw/banking77/{train,test}.csv from
PolyAI-LDN/task-specific-datasets (legacy script, datasets>=3 refuses it).
NOTE: data.adapters.adapt_banking77 builds the distractor universe from
the rows passed in ONE call, so all rows go in a single call (the
per-row call pattern in prefetch.convert_p0 would raise AdapterError and
skip everything).

Usage: python3 data/convert_banking77.py
  out: artifacts/data-prefetch/banking77.jsonl
"""

import csv
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import data.adapters as A  # noqa: E402

RAW = ROOT / "artifacts" / "data-raw" / "banking77"
OUT = ROOT / "artifacts" / "data-prefetch" / "banking77.jsonl"


def main():
    rows = []
    for split, name in (("train", "train.csv"), ("test", "test.csv")):
        with open(RAW / name, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("text") and r.get("category"):
                    rows.append({"text": r["text"], "label": r["category"],
                                 "split": split})
    n_ex, skipped = 0, 0
    with open(OUT, "w") as fout:
        try:
            examples = A.adapt_banking77(rows, seed=0)
        except A.AdapterError as e:
            print(json.dumps({"job": "banking77", "status": "error",
                              "detail": str(e)}))
            return
        for ex in examples:
            fout.write(json.dumps(dataclasses.asdict(ex),
                                  ensure_ascii=False) + "\n")
            n_ex += 1
    print(json.dumps({"job": "banking77", "status": "converted",
                      "rows": len(rows), "examples": n_ex,
                      "skipped": skipped, "out": str(OUT)}))


if __name__ == "__main__":
    main()
