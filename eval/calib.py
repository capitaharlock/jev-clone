"""Calibration and abstention OF THE MODEL (#T-calib, rewritten by
#T-unseen-labels).

What this module used to be
---------------------------
Until 2026-09-21 the predictor calibrated here was a fixed, parameter-free
char-3-gram cosine scorer. Temperature scaling a scorer with no parameters
is a measurement of nothing the product ships: finding F of the audit. The
old artifact (``artifacts/gates/T-calib/calibration.json``) is kept on disk
untouched as frozen evidence of the #T-release bundle, and it is superseded:
``LEGACY_COSINE_CALIB`` names it, ``MODEL_CALIB`` names the real one.

What it is now
--------------
The predictor is the trained pointer decision head of #T-train-real, loaded
from its safetensors checkpoint through
``training.python.train_decision.load_checkpoint``. Every entry carries the
checkpoint's ``model_version``, and ``fit_calibrator`` REFUSES a predictor
card without one — a calibration artifact that cannot name the weights it
calibrates is exactly the artifact this task was opened to delete.

Split discipline is unchanged and still enforced: entries carry their
``split`` tag, the fit functions refuse anything that is not
``split == "calibration"``, and test/unseen are scored once, in the final
report (``eval/unseen.py``).

Temperature is fitted on the CALIBRATION split of the SEEN labels only.
There is deliberately no unseen-label calibration set: holding out a label
and then fitting a temperature on it would put the held-out text back in
the fitted surface. The same temperature is applied to the unseen cut and
the transfer is reported (``applied_to``), because that is the honest
deployment story: you calibrate on what you have.

The `unknown` output is a first-class abstention strategy here
(``confidence_scores(..., "unknown")`` = ``1 - p[unknown]``), so the
risk-coverage curve the product would actually run is the one published.

CLI:
    CKPT=artifacts/checkpoints/decision/train-real-v1/stage-000250000
    .venv-train/bin/python -m eval.calib fit --checkpoint $CKPT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CALIB_SEED = 20260920
CALIB_VERSION = 2
N_BINS = 15
BOOTSTRAP_B = 1000
ECE_TARGET = 0.05

#: the parameter-free cosine calibrator this module used to fit. Frozen:
#: `training/python/test_release.py` hashes it as release evidence.
LEGACY_COSINE_CALIB = os.path.join(ROOT, "artifacts", "gates", "T-calib",
                                   "calibration.json")
#: the calibrator of the real head, written next to the primary metric
MODEL_CALIB = os.path.join(ROOT, "artifacts", "gates", "T-unseen-labels",
                           "calibration.json")

STRATEGIES = ("unknown", "maxprob", "margin", "energy")


# -- metric primitives (unchanged; pure, stdlib) ---------------------------

def softmax(logits: list[float], temp: float) -> list[float]:
    m = max(logits)
    exps = [math.exp((l - m) / temp) for l in logits]
    s = sum(exps)
    return [e / s for e in exps]


def nll(probs: list[list[float]], labels: list[int]) -> float:
    return -sum(math.log(max(p[y], 1e-12))
                for p, y in zip(probs, labels)) / len(probs)


def brier(probs: list[list[float]], labels: list[int]) -> float:
    tot = 0.0
    for p, y in zip(probs, labels):
        tot += sum((v - (1.0 if i == y else 0.0)) ** 2
                   for i, v in enumerate(p))
    return tot / len(probs)


def ece(probs: list[list[float]], labels: list[int],
        n_bins: int = N_BINS) -> dict:
    """Expected calibration error over confidence bins.

    Empty bins are kept with n=0 (no crash, no silent rebinning); the
    reliability plot regenerates from these raw bins. Computed over the
    WHOLE cut, not over a rolling window: the window in the trainer is a
    live gauge, this is the published number.
    """
    bins = [{"lo": i / n_bins, "hi": (i + 1) / n_bins,
             "n": 0, "acc": 0.0, "conf": 0.0} for i in range(n_bins)]
    for p, y in zip(probs, labels):
        conf = max(p)
        pred = max(range(len(p)), key=lambda i: p[i])
        b = min(int(conf * n_bins), n_bins - 1)
        bins[b]["n"] += 1
        bins[b]["acc"] += 1.0 if pred == y else 0.0
        bins[b]["conf"] += conf
    err = 0.0
    n = len(probs)
    for b in bins:
        if b["n"]:
            b["acc"] /= b["n"]
            b["conf"] /= b["n"]
            err += (b["n"] / n) * abs(b["acc"] - b["conf"])
    return {"ece": err, "bins": bins, "n": n, "n_bins": n_bins}


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple:
    """95 % Wilson score interval — the honest CI for a small cut."""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


# -- temperature fitting ---------------------------------------------------

def _fit_temp_grid(probs_logits: list[list[float]],
                   labels: list[int]) -> dict:
    """Grid search on temperature minimizing NLL (deterministic)."""
    before = nll([softmax(l, 1.0) for l in probs_logits], labels)
    best_t, best_v = 1.0, before
    for i in range(61):
        t = 0.05 * (200.0 ** (i / 60.0))
        v = nll([softmax(l, t) for l in probs_logits], labels)
        if v < best_v:
            best_v, best_t = v, t
    return {"temperature": best_t, "nll_before": before,
            "nll_after": best_v, "n": len(labels)}


def _require_calibration(entries: list[dict]) -> None:
    bad = sorted({e["split"] for e in entries} - {"calibration"})
    if bad:
        raise ValueError(
            f"refusing to fit on non-calibration splits: {bad} "
            "(test/unseen stay untouched until the final report)")


def _require_model(predictor: dict) -> dict:
    """A calibrator must name the weights it calibrates."""
    mv = (predictor or {}).get("model_version")
    if not mv:
        raise ValueError(
            "predictor card carries no model_version: refusing to write a "
            "calibration artifact that cannot name its checkpoint "
            "(#T-unseen-labels gate criterion 2)")
    return predictor


def fit_temperature(entries: list[dict], group: str | None = None) -> dict:
    """Fit temperature on calibration entries only.

    group: None (global), "dataset" or "cardinality". Groups absent from
    the fit data are simply not emitted; application falls back to the
    global temperature and records the fallback.
    """
    _require_calibration(entries)
    if group is None:
        out = _fit_temp_grid([e["logits"] for e in entries],
                             [e["label"] for e in entries])
        out["group"] = "global"
        return out
    key = (lambda e: e["dataset"]) if group == "dataset" else (
        lambda e: e["cardinality"])
    groups: dict[str, list[dict]] = {}
    for e in entries:
        groups.setdefault(str(key(e)), []).append(e)
    return {"group": group,
            "by_group": {g: _fit_temp_grid([e["logits"] for e in es],
                                           [e["label"] for e in es])
                         for g, es in sorted(groups.items())}}


def fit_calibrator(calib_entries: list[dict], predictor: dict) -> dict:
    """Full hierarchy: global -> per-dataset -> per-cardinality."""
    _require_model(predictor)
    glob = fit_temperature(calib_entries)
    ds = fit_temperature(calib_entries, group="dataset")
    card = fit_temperature(calib_entries, group="cardinality")
    digest = hashlib.sha256(json.dumps(
        sorted(e["id"] for e in calib_entries)).encode()).hexdigest()
    return {"format": 2, "calib_version": CALIB_VERSION,
            "predictor": predictor,
            "model_version": predictor["model_version"],
            "fit_split": "calibration",
            "fit_cut": "seen labels only — a held-out label may not be "
                       "fitted on, not even a temperature",
            "applied_to": ["seen", "unseen", "eval-only"],
            "split_sha256": digest,
            "n_fit": len(calib_entries),
            "global": glob,
            "by_dataset": ds["by_group"],
            "by_cardinality": card["by_group"],
            "thresholds": fit_thresholds(calib_entries, glob)}


def _temp_for(cal: dict, entry: dict) -> tuple[float, str]:
    ds = cal.get("by_dataset", {}).get(entry.get("dataset"))
    if ds is not None:
        return ds["temperature"], "dataset"
    card = cal.get("by_cardinality", {}).get(str(entry.get("cardinality")))
    if card is not None:
        return card["temperature"], "cardinality"
    return cal["global"]["temperature"], "global-fallback"


def apply_calibrator(cal: dict, entry: dict) -> tuple[list[float], str]:
    """Scale entry logits; returns (probs, which_temperature_was_used)."""
    t, which = _temp_for(cal, entry)
    return softmax(entry["logits"], t), which


def calib_path() -> str:
    return MODEL_CALIB


def save_calibration(cal: dict, path: str | None = None) -> str:
    path = path or calib_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cal, f, indent=2)
        f.write("\n")
    return path


def load_calibration(path: str | None = None) -> dict:
    with open(path or calib_path()) as f:
        return json.load(f)


# -- abstention ------------------------------------------------------------

def confidence_scores(logits: list[float], probs: list[float],
                      strategy: str) -> float:
    """Higher = keep. `unknown` is the product's own abstention output."""
    if strategy == "unknown":
        # the head's last logit IS the learned `unknown`: keeping a row
        # means the model did not ask to abstain on it
        return 1.0 - probs[-1]
    if strategy == "maxprob":
        return max(probs)
    if strategy == "energy":
        m = max(logits)
        return m + math.log(sum(math.exp(l - m) for l in logits))
    if strategy == "margin":
        s = sorted(probs, reverse=True)
        return s[0] - s[1] if len(s) > 1 else s[0]
    raise ValueError(f"unknown strategy: {strategy}")


def risk_coverage(entries: list[dict], strategy: str,
                  n_points: int = 21) -> dict:
    """Selective risk at coverage levels + random-abstention baseline."""
    scored = [(confidence_scores(e["logits"], e["probs"], strategy),
               e["pred"] == e["label"]) for e in entries]
    scored.sort(key=lambda t: t[0], reverse=True)
    n = len(scored)
    overall_err = 1.0 - sum(ok for _, ok in scored) / n
    curve = []
    for i in range(n_points):
        k = max(1, int(n * (i + 1) / n_points))
        kept = scored[:k]
        risk = 1.0 - sum(ok for _, ok in kept) / k
        curve.append({"coverage": round(k / n, 6), "risk": round(risk, 6),
                      "k": k})
    avg_selective = sum(c["risk"] for c in curve) / len(curve)
    return {"strategy": strategy, "curve": curve,
            "overall_error": round(overall_err, 6),
            "avg_selective_risk": round(avg_selective, 6),
            "beats_random": avg_selective < overall_err,
            "random_risk": round(overall_err, 6), "n": n}


def bootstrap_risk_ci(entries: list[dict], strategy: str,
                      coverages: tuple[float, ...] = (0.5, 0.8, 0.9),
                      seed: int = CALIB_SEED, b: int = BOOTSTRAP_B) -> dict:
    """95% bootstrap CI of selective risk at fixed coverages."""
    rng = random.Random(seed)
    n = len(entries)
    base = [(confidence_scores(e["logits"], e["probs"], strategy),
             e["pred"] == e["label"]) for e in entries]
    out: dict[str, dict] = {}
    for cov in coverages:
        k = max(1, int(n * cov))
        risks = []
        for _ in range(b):
            samp = [base[rng.randrange(n)] for _ in range(n)]
            samp.sort(key=lambda t: t[0], reverse=True)
            kept = samp[:k]
            risks.append(1.0 - sum(ok for _, ok in kept) / k)
        risks.sort()
        ranked = sorted(base, key=lambda t: t[0], reverse=True)
        kept = ranked[:k]
        point = 1.0 - sum(ok for _, ok in kept) / k
        out[str(cov)] = {"coverage": cov, "k": k, "risk": round(point, 6),
                         "ci95": [round(risks[int(0.025 * b)], 6),
                                  round(risks[int(0.975 * b) - 1], 6)]}
    return out


def fit_thresholds(calib_entries: list[dict], glob: dict,
                   target_coverage: float = 0.9) -> dict:
    """Abstain-below thresholds fixed on calibration data only."""
    _require_calibration(calib_entries)
    t = glob["temperature"]
    out = {}
    for strategy in STRATEGIES:
        scored = sorted(
            (confidence_scores(e["logits"], softmax(e["logits"], t),
                               strategy) for e in calib_entries))
        idx = min(int(len(scored) * (1.0 - target_coverage)),
                  len(scored) - 1)
        out[strategy] = {"threshold": scored[idx],
                         "target_coverage": target_coverage}
    return out


# -- the model as the predictor -------------------------------------------

def predictor_card(manifest: dict, ckpt_dir: str) -> dict:
    """Identity of the weights being calibrated. No model_version, no fit."""
    arch = manifest.get("architecture", {})
    return {"name": "pointer-decision-head",
            "model_version": manifest.get("model_version"),
            "checkpoint": os.path.relpath(ckpt_dir, ROOT),
            "backbone": (manifest.get("backbone") or {}).get("id"),
            "tokenizer_hash": manifest.get("tokenizer_hash"),
            "samples_seen": manifest.get("samples_seen"),
            "architecture": arch,
            "version": CALIB_VERSION,
            "supersedes": {"name": "cosine-char3-softmax",
                           "artifact": os.path.relpath(LEGACY_COSINE_CALIB,
                                                       ROOT),
                           "why": "parameter-free scorer, not the product"}}


def entries_from_samples(engine, samples, split: str, cut: str,
                         dataset: str = "", batch_size: int = 32) -> list:
    """Real `[K + 1]` logits for a list of `data.optset.Sample`.

    The only predictor in this module. Requires the torch stack
    (`.venv-train`); nothing here is stubbed.
    """
    from training.python import train_decision as T

    T._require_torch()
    import torch

    out = []
    engine.head.eval()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            tokens, mask, _ = T.encode_states(
                engine.backbone, [s.state for s in batch],
                T.TRAIN_MAX_LENGTH)
            embs, spans = T.batch_embeddings(engine, batch)
            # one head call per batch, not per row (#T-metal-throughput);
            # each row is read back as its own K_i options + `unknown`.
            q_emb, opt_embs, opt_mask = T.pack_options(embs, spans,
                                                       engine.device)
            kmax = opt_embs.shape[1]
            rows = engine.head.forward_batch(tokens, mask, q_emb, opt_embs,
                                             opt_mask).float().cpu().tolist()
            for i, sample in enumerate(batch):
                k = len(sample.options)
                logits = [float(x) for x in rows[i][:k] + [rows[i][kmax]]]
                probs = softmax(logits, 1.0)
                pred = max(range(len(probs)), key=lambda j: probs[j])
                out.append({
                    "id": f"{sample.dataset}:{sample.row_id}:"
                          f"{sample.question_id}",
                    "split": split, "cut": cut,
                    "dataset": dataset or sample.dataset,
                    "cardinality": sample.k,
                    "logits": logits, "probs": probs,
                    "label": sample.gold_index, "pred": pred,
                    "locale": sample.state[1:6]
                    if sample.state[:1] == "[" else "",
                })
    return out


def metrics_of(entries: list[dict], probs_key: str = "probs") -> dict:
    """accuracy / ECE / Brier / NLL over a list of scored entries."""
    if not entries:
        return {"n": 0, "skipped": "empty cut"}
    probs = [e[probs_key] for e in entries]
    labels = [e["label"] for e in entries]
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in probs]
    correct = sum(p == y for p, y in zip(preds, labels))
    # argmax restricted to the K real options: does the pointer RANK the
    # right option first with `unknown` taken out of the race. Diagnostic.
    opt_correct = sum(
        (max(range(len(p) - 1), key=lambda i: p[i]) if len(p) > 1 else 0) == y
        for p, y in zip(probs, labels))
    n = len(entries)
    lo, hi = wilson_interval(correct, n)
    olo, ohi = wilson_interval(opt_correct, n)
    return {
        "n": n,
        "accuracy": round(correct / n, 6),
        "accuracy_ci95": [round(lo, 6), round(hi, 6)],
        "chance": round(sum(1.0 / len(p) for p in probs) / n, 6),
        "accuracy_options_only": round(opt_correct / n, 6),
        "accuracy_options_only_ci95": [round(olo, 6), round(ohi, 6)],
        "chance_options_only": round(
            sum(1.0 / max(len(p) - 1, 1) for p in probs) / n, 6),
        "ece": round(ece(probs, labels)["ece"], 6),
        "brier": round(brier(probs, labels), 6),
        "nll": round(nll(probs, labels), 6),
        "mean_k": round(sum(len(p) - 1 for p in probs) / n, 4),
        "abstain_rate": round(
            sum(p == len(pr) - 1 for p, pr in zip(preds, probs)) / n, 6),
    }


def calibrated(cal: dict, entries: list[dict]) -> list[dict]:
    """Copy of `entries` with temperature-scaled probs (and preds)."""
    out = []
    for e in entries:
        p, which = apply_calibrator(cal, e)
        out.append(dict(e, probs=p, temp_source=which,
                        pred=max(range(len(p)), key=lambda i: p[i])))
    return out


def run_calib(ckpt_dir: str, device: str = "auto",
              max_samples: int = 3000, out_path: str | None = None) -> dict:
    """Fit + save the calibrator of a checkpoint. Seen labels only."""
    from data.optset import SamplerConfig

    from training.python import train_decision as T

    from eval import unseen as U

    engine, manifest = T.load_checkpoint(ckpt_dir, device)
    holdout = T.build_holdout()
    entries = U.calibration_entries(engine, holdout, SamplerConfig(),
                                    max_samples=max_samples)
    cal = fit_calibrator(entries, predictor_card(manifest, ckpt_dir))
    path = save_calibration(cal, out_path)
    cal["path"] = path
    return cal


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(prog="eval.calib")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="fit + save the model calibrator")
    f.add_argument("--checkpoint", required=True)
    f.add_argument("--device", default="auto")
    f.add_argument("--max-samples", type=int, default=3000)
    f.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cal = run_calib(args.checkpoint, args.device, args.max_samples, args.out)
    print(json.dumps({"path": cal["path"],
                      "model_version": cal["model_version"],
                      "temperature": cal["global"]["temperature"],
                      "nll_before": cal["global"]["nll_before"],
                      "nll_after": cal["global"]["nll_after"],
                      "n_fit": cal["n_fit"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
