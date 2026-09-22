"""Predict CLI for baseline model packs (#T-train-base v0).

Usage: .venv-train/bin/python tools/predict_v0.py --run <ts> --task massive "text…"
"""
import argparse
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run id under artifacts/runs/")
    ap.add_argument("--task", required=True)
    ap.add_argument("text", nargs="+")
    args = ap.parse_args()

    with open(ROOT / "artifacts" / "runs" / args.run / "models" / f"{args.task}.pkl", "rb") as fh:
        pack = pickle.load(fh)
    vec, clf = pack["vectorizer"], pack["clf"]
    text = " ".join(args.text)
    proba = clf.predict_proba(vec.transform([text]))[0]
    labels = list(clf.classes_)
    ranked = sorted(zip(labels, proba), key=lambda kv: -kv[1])[:5]
    print(f"task={args.task} pred={ranked[0][0]}")
    for lab, p in ranked:
        print(f"  {lab}: {p:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
