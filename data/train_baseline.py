"""HISTORICAL BASELINE — not the product path (#T-train-real).

    STATUS: superseded on 2026-09-21. Kept for comparison, never deleted,
    and it feeds NO green gate.

This is the 24 h loop the 2026-09-21 audit took apart (findings A and B):
`load()` builds `X = r["state"]`, `y = q["answer"]` and THROWS AWAY the
question and the option set, so the model is fitted to a fixed, global
label space per dataset — one pickle each, ~1.8 M bag-of-words weights.
That is why ReClor scored 0.254 (exact chance at K=4) and LogiQA 0.435
(a positional prior): the option TEXTS are never seen.

The product path is `training/python/train_decision.py`: one multi-dataset
model, listwise cross-entropy over the row's own `K + 1` option logits,
scored by `model.decision_head` over each option's text embedding. Its
headline metric is accuracy on labels never seen in training; this file
cannot even express that question.

Use this module only to reproduce the historical numbers or to compare
against them. Anything that promotes its accuracy to a product claim is a
regression — the `:8794` dashboard labels it as history for that reason.

Original docstring follows.

Baseline trainer v1: TF-IDF + LogisticRegression per task family.

Consumes universal-schema JSONL from artifacts/data-prefetch/:
  - boolean (boolq, civil-comments) -> binary classification on state text
  - choice  (helpsteer2)            -> multiclass classification on state text
Trains on split==train, evaluates on calibration/test (or held-out slice).
Writes metrics + manifest to artifacts/runs/<ts>/.

Usage: .venv-train/bin/python data/train_baseline.py [--sample N]
"""

import argparse
import hashlib
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

# Read by the monitor and by anything tempted to call this a result.
HISTORICAL_BASELINE = True
SUPERSEDED_BY = "training/python/train_decision.py (#T-train-real)"
SUPERSEDED_ON = "2026-09-21"

sys.path.insert(0, str(ROOT))
from data.firewall import (  # noqa: E402  (trainer runs as script or -m)
    BenchmarkRegistry,
    blocked_hash_sets,
    check_job_allowed,
)
from data.leakage import text_hash  # noqa: E402
from data.mix import keep_row, scan_supply  # noqa: E402

#: seed of the per-job downsample; the manifest already published `7`
SUBSAMPLE_SEED = 7

# Eval-only firewall (#T-halt-contam): synth-loop is quarantined (finding C),
# logiqa + reclor are reasoning benchmarks, never train (finding G). They must
# not reappear here — main() refuses them as an error via check_job_allowed.
#
# #T-corpus-rebalance: every `sample` below is the quota `data.mix` grants
# that source under the 15 %-per-dataset / 30 %-per-family caps. SAMPLING IS
# THE RULE, not the exception civil-comments used to be — `None` here used to
# mean "take all of it", which is how one binary dataset became 74.6 % of the
# corpus (finding E). `data/test_mix.py::JobsRegistry` fails if any of them
# grows past what the descriptor grants, so they cannot drift back.
#
# This trainer reads ONE file per job and so has no `synth-v1` entry: seven
# sources cover only 90 % of any mixture under the caps (six cap units), so
# the JOBS list itself cannot be cap-clean. The product path
# (`training/python/train_decision.py`) is the one the caps are enforced on.
JOBS = [
    {"name": "boolq", "file": "boolq.jsonl", "kind": "boolean", "sample": 9427},
    {"name": "helpsteer2", "file": "helpsteer2.jsonl", "kind": "choice", "sample": 9767},
    {"name": "civil-comments", "file": "civil-comments.jsonl", "kind": "boolean", "sample": 9767},
    {"name": "huffpost", "file": "huffpost.jsonl", "kind": "choice", "sample": 9767},
    {"name": "banking77", "file": "banking77.jsonl", "kind": "choice", "sample": 9767},
    {"name": "massive", "file": "massive.jsonl", "kind": "choice", "sample": 9767},
    {"name": "email-triage", "file": "email-triage.jsonl", "kind": "choice", "sample": 515},
]

_BLOCKED_RAW, _BLOCKED_NORM = None, None


def _blocked_sets():
    """Canary hash sets, loaded once per process (fast O(1) row check)."""
    global _BLOCKED_RAW, _BLOCKED_NORM
    if _BLOCKED_RAW is None:
        _BLOCKED_RAW, _BLOCKED_NORM = blocked_hash_sets()
    return _BLOCKED_RAW, _BLOCKED_NORM

EVAL_SPLITS = {"calibration", "test", "valid", "validation"}


def keep_fraction(job) -> float:
    """`job["sample"] / this file's train supply`, from the cached scan.

    The quota is a row COUNT; the selector needs a rate. `data.mix` already
    counted the supply of every registered source (and cached it), so this
    costs nothing and stays true when a corpus grows.
    """
    if job["sample"] is None:
        return 1.0
    supply = scan_supply([job["name"]]).get(job["name"], {})
    return min(1.0, job["sample"] / max(supply.get("train_questions", 0), 1))


def load(job, sample):
    X_train, y_train, X_eval, y_eval = [], [], [], []
    blocked_raw, blocked_norm = _blocked_sets()
    budget = sample or job["sample"]
    fraction = keep_fraction({**job, "sample": budget})
    index = -1
    for line in open(DATA / job["file"]):
        r = json.loads(line)
        if r.get("split") == "train":
            # Firewall at row level (#T-halt-contam): benchmark content in a
            # training row is a hard error, never a warning.
            state = r.get("state", "")
            if hashlib.sha256(state.encode()).hexdigest() in blocked_raw:
                raise ValueError(
                    f"job {job['name']}: training row matches blocked "
                    "content (exact layer)"
                )
            if text_hash(state) in blocked_norm:
                raise ValueError(
                    f"job {job['name']}: training row matches blocked "
                    "content (normalized layer)"
                )
        if r.get("split") == "train":
            index += 1
        for q in r["questions"]:
            ans = q.get("answer")
            if ans is None or ans == "unknown":
                continue  # hidden-gold clean-room rows never train
            if r.get("split") == "train":
                # Seeded, reproducible downsample keyed on the ROW. The old
                # key was `hash(q["id"])`: salted per process, and constant
                # across all 1 999 514 civil-comments rows (they share the
                # id `civil-toxicity-all-0`), so it kept everything.
                if not keep_row(SUBSAMPLE_SEED, job["name"], index,
                                q["id"], fraction):
                    continue
                if budget is not None and len(X_train) >= budget:
                    continue
                X_train.append(r["state"])
                y_train.append(ans)
            elif r.get("split") in EVAL_SPLITS:
                X_eval.append(r["state"])
                y_eval.append(ans)
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
    return out, vec, clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None)
    ap.parse_args()  # validated for argv errors; per-job `sample` rules apply
    print(f"[historical-baseline] superseded {SUPERSEDED_ON} by "
          f"{SUPERSEDED_BY}: this trainer discards the question and the "
          f"option set and fits a fixed global label space. Its numbers are "
          f"history, not a product claim.", file=sys.stderr, flush=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rundir = ROOT / "artifacts" / "runs" / ts
    rundir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run": ts,
        "model": "tfidf-logreg-v1",
        "seeds": {"subsample": 7},
        "data": {j["name"]: (DATA / j["file"]).stat().st_size
                 for j in JOBS if (DATA / j["file"]).exists()},
    }
    results = []
    modeldir = rundir / "models"
    modeldir.mkdir(exist_ok=True)
    import pickle

    jobs = [j for j in JOBS if (DATA / j["file"]).exists()]
    # Job-level barrier (#T-halt-contam): an eval-only benchmark in JOBS is a
    # hard error, never a warning. Fail before spending hours training.
    registry = BenchmarkRegistry()
    for job in jobs:
        check_job_allowed(job["name"], registry)
    for job in jobs:
        t0 = time.time()
        res, vec, clf = run_job(job)
        res["seconds"] = round(time.time() - t0, 1)
        with open(modeldir / f"{job['name']}.pkl", "wb") as fh:
            pickle.dump({"vectorizer": vec, "clf": clf, "task": job["name"]}, fh)
        res["model"] = f"models/{job['name']}.pkl"
        results.append(res)
        print(json.dumps(res), flush=True)
    manifest["jobs"] = results
    (rundir / "metrics.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {rundir / 'metrics.json'}")


if __name__ == "__main__":
    main()
