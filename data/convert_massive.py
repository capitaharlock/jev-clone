"""Convert MASSIVE tarball -> universal schema JSONL (stdlib only).

Source: artifacts/data-raw/massive/amazon-massive-1.1.tar.gz (S3,
~1GB). Streams JSONL members for the requested locales (default
en-US/es-ES/fr-FR/de-DE/pt-PT/it-IT, T-massive-huff §26) from the
tarball without full extraction. v1 maps utt -> intent (choice,
n_options=4, seed=0);
slot spans (annot_utt) are preserved raw in state for a later slot
adapter. partition train/valid/test -> split train/calibration/test.

Usage: python3 data/convert_massive.py [--locales en-US,es-ES]
  out: artifacts/data-prefetch/massive.jsonl
"""

import argparse
import dataclasses
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import data.adapters as A  # noqa: E402
from data.adapters import Example, Option, Question, _checked, _rng, _sample_subset  # noqa: E402

TAR = ROOT / "artifacts" / "data-raw" / "massive" / "amazon-massive-1.1.tar.gz"
OUT = ROOT / "artifacts" / "data-prefetch" / "massive.jsonl"

PART2SPLIT = {"train": "train", "valid": "calibration", "validation": "calibration",
              "test": "test", "dev": "calibration"}


def adapt_massive(rows: list[dict], *, seed: int = 0, n_options: int = 4):
    rng = _rng(seed)
    universe = sorted({r["intent"] for r in rows if r.get("intent")})
    if len(universe) < 2:
        raise A.AdapterError("massive: need >= 2 distinct intents")
    out = []
    for i, r in enumerate(rows):
        utt, intent = r.get("utt"), r.get("intent")
        if not utt or not str(utt).strip():
            raise A.AdapterError(f"massive row {i}: empty utt")
        labels = _sample_subset(intent, universe, n_options, rng) if intent else universe[:n_options]
        rng.shuffle(labels)
        out.append(_checked(Example(
            state=f"[{r.get('locale', '?')}] {utt}", split=r.get("split", "train"),
            questions=[Question(id=f"massive-intent-{i}", kind="choice",
                                options=[Option(id=l, text=l) for l in labels],
                                answer=intent if intent else "unknown")],
        ), f"massive row {i}"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locales", default="en-US,es-ES,fr-FR,de-DE,pt-PT,it-IT")
    args = ap.parse_args()
    locales = set(args.locales.split(","))
    rows, skipped = [], 0
    with tarfile.open(TAR, "r:*") as tf:
        for m in tf.getmembers():
            name = m.name
            loc = next((L for L in locales if f"/{L}." in name or name.endswith(f"/{L}.jsonl")), None)
            if loc is None or not name.endswith(".jsonl"):
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            for line in f:
                line = line.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if r.get("locale") not in locales:
                        continue
                    rows.append({"utt": r.get("utt", ""),
                                 "intent": r.get("intent"),
                                 "locale": r.get("locale"),
                                 "split": PART2SPLIT.get(r.get("partition", "train"), "train")})
                except Exception:
                    skipped += 1
    n_ex = 0
    with open(OUT, "w") as fout:
        for ex in adapt_massive(rows, seed=0):
            fout.write(json.dumps(dataclasses.asdict(ex), ensure_ascii=False) + "\n")
            n_ex += 1
    print(json.dumps({"job": "massive", "status": "converted",
                      "rows": len(rows), "examples": n_ex,
                      "skipped": skipped, "locales": sorted(locales),
                      "out": str(OUT)}))


if __name__ == "__main__":
    main()
