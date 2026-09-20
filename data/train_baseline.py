"""Baseline trainer v1: TF-IDF + LogisticRegression per task family.

Consumes universal-schema JSONL from artifacts/data-prefetch/:
  - boolean (boolq, civil-comments) -> binary classification on state text
  - choice  (helpsteer2)            -> multiclass classification on state text
Trains on split==train, evaluates on calibration/test (or held-out slice).
Writes metrics + manifest to artifacts/runs/<ts>/.

Usage: .venv-train/bin/python data/train_baseline.py [--sample N]
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "artifacts" / "data-prefetch"

JOBS = [
    {"name": "boolq", "file": "boolq.jsonl", "kind": "boolean", "sample": None},
    {"name": "helpsteer2", "file": "helpsteer2.jsonl", "kind": "choice", "sample": None},
    {"name": "civil-comments", "file": "civil-comments.jsonl", "kind": "boolean", "sample": 50000},
    {"name": "huffpost", "file": "huffpost.jsonl", "kind": "choice", "sample": None},
    {"name": "banking77", "file": "banking77.jsonl", "kind": "choice", "sample": None},
    {"name": "massive", "file": "massive.jsonl", "kind": "choice", "sample": None},
    {"name": "logiqa", "file": "logiqa.jsonl", "kind": "choice", "sample": None},
    {"name": "reclor", "file": "reclor.jsonl", "kind": "choice", "sample": None},
]

EVAL_SPLITS = {"calibration", "test", "valid", "validation"}


def load(job, sample):
    X_train, y_train, X_eval, y_eval = [], [], [], []
    rng_seed = 7
    seen = 0
    for line in open(DATA / job["file"]):
        r = json.loads(line)
        for q in r["questions"]:
            ans = q.get("answer")
            if ans is None or ans == "unknown":
                continue  # hidden-gold clean-room rows never train
            if job["sample"] is not None and r.get("split") == "train":
                # deterministic reservoir-ish subsample: hash on id
                if (hash(q["id"]) % (10**9)) % 100 >= int(
                    100 * job["sample"] / 2000000
                ) and seen >= (sample or job["sample"]):
                    continue
            if r.get("split") == "train":
                if job["sample"] is not None and len(X_train) >= job["sample"]:
                    continue
                X_train.append(r["state"])
                y_train.append(ans)
            elif r.get("split") in EVAL_SPLITS:
                X_eval.append(r["state"])
                y_eval.append(ans)
            seen += 1
    # cap eval for speed
    if len(X_eval) > 20000:
        X_eval, y_eval = X_eval[:20000], y_eval[:20000]
    return X_train, y_train, X_eval, y_eval


def run_job(job):
    Xtr, ytr, Xev, yev = load(job, job["sample"])
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    Xtr_v = vec.fit_transform(Xtr)
    clf = LogisticRegression(max_iter=200, C=1.0)
    clf.fit(Xtr_v, ytr)
    out = {"task": job["name"], "n_train": len(ytr), "n_eval": len(yev)}
    if yev:
        pred = clf.predict(vec.transform(Xev))
        proba = clf.predict_proba(vec.transform(Xev))
        out["accuracy"] = accuracy_score(yev, pred)
        try:
            out["logloss"] = log_loss(yev, proba, labels=list(clf.classes_))
        except ValueError:
            out["logloss"] = None
    else:
        out["accuracy"] = out["logloss"] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None)
    args = ap.parse_args()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rundir = ROOT / "artifacts" / "runs" / ts
    rundir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run": ts,
        "model": "tfidf-logreg-v1",
        "seeds": {"subsample": 7},
        "data": {j["name"]: (DATA / j["file"]).stat().st_size for j in JOBS},
    }
    results = []
    for job in JOBS:
        t0 = time.time()
        res = run_job(job)
        res["seconds"] = round(time.time() - t0, 1)
        results.append(res)
        print(json.dumps(res), flush=True)
    manifest["jobs"] = results
    (rundir / "metrics.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {rundir / 'metrics.json'}")


if __name__ == "__main__":
    main()
