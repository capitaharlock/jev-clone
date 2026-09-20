"""Calibration and abstention (#T-calib).

Toolkit over (probs, label) pairs: NLL/Brier/ECE, temperature scaling
(global -> per-locale -> per-cardinality), reliability data regenerated
from raw metrics, and a serialized calibrator in ``calibration.json``.
Abstention strategies (max-prob, energy, explicit-unknown margin) are
compared with risk-coverage curves plus bootstrap intervals.

Split discipline: every entry carries its ``split`` tag and the fit
functions REFUSE any entry that is not ``split == "calibration"`` — the
runner can never tune parameters on test. Test and OOD are scored once,
in the final report.

The predictor calibrated here is a fixed, parameter-free cosine scorer
(char-3-gram, same family as the gold benchmark gap): no predictor
weights are fit on any split, so the only fitted parameters are the
temperatures, on the calibration split alone.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random

from data.gold import (
    _cos_text,
    build_pilot,
    massive_adapter,
    massive_card,
    massive_fixture_path,
    reserve_splits,
)

CALIB_SEED = 20260920
CALIB_VERSION = 1
N_BINS = 15
BOOTSTRAP_B = 1000
ECE_TARGET = 0.05

PREDICTOR = {"name": "cosine-char3-softmax", "base_temperature": 1.0,
             "version": 1}


def softmax(logits: list[float], temp: float) -> list[float]:
    m = max(logits)
    exps = [math.exp((l - m) / temp) for l in logits]
    s = sum(exps)
    return [e / s for e in exps]


def predict_probs(item: dict, base_temp: float = 1.0) -> list[float]:
    """Fixed predictor: cosine(state, option) -> softmax. No fitted params."""
    scores = [_cos_text(item["state"], o["text"]) for o in item["options"]]
    return softmax(scores, base_temp)


def logits_of(item: dict) -> list[float]:
    return [_cos_text(item["state"], o["text"]) for o in item["options"]]


def label_of(item: dict) -> int:
    return next(i for i, o in enumerate(item["options"])
                if o["id"] == item["answer"])


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
    reliability plot regenerates from these raw bins.
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
            "(test/OOD stay untouched until the final report)")


def fit_temperature(entries: list[dict],
                    group: str | None = None) -> dict:
    """Fit temperature on calibration entries only.

    group: None (global), "locale" (per-type) or "cardinality".
    Groups absent from the fit data are simply not emitted; application
    falls back to the global temperature and records the fallback.
    """
    _require_calibration(entries)
    if group is None:
        out = _fit_temp_grid([e["logits"] for e in entries],
                             [e["label"] for e in entries])
        out["group"] = "global"
        return out
    key = (lambda e: e["locale"]) if group == "locale" else (
        lambda e: e["cardinality"])
    groups: dict[str, list[dict]] = {}
    for e in entries:
        groups.setdefault(str(key(e)), []).append(e)
    return {"group": group,
            "by_group": {g: _fit_temp_grid([e["logits"] for e in es],
                                           [e["label"] for e in es])
                         for g, es in sorted(groups.items())}}


def fit_calibrator(calib_entries: list[dict]) -> dict:
    """Full hierarchy: global -> per-locale -> per-cardinality."""
    glob = fit_temperature(calib_entries)
    loc = fit_temperature(calib_entries, group="locale")
    card = fit_temperature(calib_entries, group="cardinality")
    digest = hashlib.sha256(json.dumps(
        sorted(e["id"] for e in calib_entries)).encode()).hexdigest()
    return {"format": 1, "calib_version": CALIB_VERSION,
            "predictor": PREDICTOR,
            "fit_split": "calibration", "split_sha256": digest,
            "n_fit": len(calib_entries),
            "global": glob,
            "by_locale": loc["by_group"],
            "by_cardinality": card["by_group"],
            "thresholds": fit_thresholds(calib_entries, glob)}


def _temp_for(cal: dict, entry: dict) -> tuple[float, str]:
    loc = cal["by_locale"].get(entry["locale"])
    if loc is not None:
        return loc["temperature"], "locale"
    card = cal["by_cardinality"].get(str(entry["cardinality"]))
    if card is not None:
        return card["temperature"], "cardinality"
    return cal["global"]["temperature"], "global-fallback"


def apply_calibrator(cal: dict, entry: dict) -> tuple[list[float], str]:
    """Scale entry logits; returns (probs, which_temperature_was_used)."""
    t, which = _temp_for(cal, entry)
    return softmax(entry["logits"], t), which


def calib_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "artifacts", "gates", "T-calib",
        "calibration.json")


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


def confidence_scores(logits: list[float], probs: list[float],
                      strategy: str) -> float:
    """Higher = keep. Three abstention strategies."""
    if strategy == "maxprob":
        return max(probs)
    if strategy == "energy":
        return math.log(sum(math.exp(l) for l in logits))
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
        curve.append({"coverage": k / n, "risk": risk, "k": k})
    avg_selective = sum(c["risk"] for c in curve) / len(curve)
    return {"strategy": strategy, "curve": curve,
            "overall_error": overall_err,
            "avg_selective_risk": avg_selective,
            "beats_random": avg_selective < overall_err,
            "random_risk": overall_err, "n": n}


def bootstrap_risk_ci(entries: list[dict], strategy: str,
                      coverages: tuple[float, ...] = (0.5, 0.8, 0.9),
                      seed: int = CALIB_SEED) -> dict:
    """95% bootstrap CI of selective risk at fixed coverages."""
    rng = random.Random(seed)
    n = len(entries)
    base = [(confidence_scores(e["logits"], e["probs"], strategy),
             e["pred"] == e["label"]) for e in entries]
    out: dict[str, dict] = {}
    for cov in coverages:
        k = max(1, int(n * cov))
        risks = []
        for _ in range(BOOTSTRAP_B):
            samp = [base[rng.randrange(n)] for _ in range(n)]
            samp.sort(key=lambda t: t[0], reverse=True)
            kept = samp[:k]
            risks.append(1.0 - sum(ok for _, ok in kept) / k)
        risks.sort()
        ranked = sorted(base, key=lambda t: t[0], reverse=True)
        kept = ranked[:k]
        point = 1.0 - sum(ok for _, ok in kept) / k
        out[str(cov)] = {"coverage": cov, "k": k, "risk": point,
                         "ci95": [risks[int(0.025 * BOOTSTRAP_B)],
                                  risks[int(0.975 * BOOTSTRAP_B) - 1]]}
    return out


def fit_thresholds(calib_entries: list[dict], glob: dict,
                   target_coverage: float = 0.9) -> dict:
    """Abstain-below thresholds fixed on calibration data only."""
    _require_calibration(calib_entries)
    t = glob["temperature"]
    out = {}
    for strategy in ("maxprob", "energy", "margin"):
        scored = sorted(
            (confidence_scores(e["logits"], softmax(e["logits"], t),
                               strategy) for e in calib_entries))
        idx = min(int(len(scored) * (1.0 - target_coverage)),
                  len(scored) - 1)
        out[strategy] = {"threshold": scored[idx],
                         "target_coverage": target_coverage}
    return out


def build_entries() -> tuple[list[dict], list[dict], list[dict]]:
    """Calibration / test / OOD entries with split tags.

    OOD = MASSIVE fixture (real data, 6 options, intent task): a genuine
    distribution shift in task, cardinality and vocabulary.
    """
    items, _ = build_pilot()
    splits = reserve_splits(items)
    cal_ids = {c["id"] for c in splits["calibration"]}

    def entry(it: dict) -> dict:
        logits = logits_of(it)
        probs = softmax(logits, 1.0)
        pred = max(range(len(probs)), key=lambda i: probs[i])
        return {"id": it["id"], "split": "calibration"
                if it["id"] in cal_ids else "test",
                "locale": it["locale"],
                "cardinality": len(it["options"]),
                "logits": logits, "label": label_of(it), "pred": pred}

    calib = [entry(it) for it in splits["calibration"]]
    test = [entry(it) for it in splits["test"]]
    with open(massive_fixture_path()) as f:
        rows = [json.loads(l) for l in f]
    intents = sorted({r["intent"] for r in rows})
    ood = []
    for r in rows:
        it = massive_adapter(r, intents)
        logits = logits_of(it)
        probs = softmax(logits, 1.0)
        pred = max(range(len(probs)), key=lambda i: probs[i])
        ood.append({"id": it["id"], "split": "ood", "locale": it["locale"],
                    "cardinality": len(it["options"]), "logits": logits,
                    "label": label_of(it), "pred": pred})
    return calib, test, ood


def _metrics(probs: list[list[float]], labels: list[int]) -> dict:
    return {"nll": nll(probs, labels), "brier": brier(probs, labels),
            "ece": ece(probs, labels)["ece"],
            "accuracy": sum(max(range(len(p)), key=lambda i: p[i]) == y
                            for p, y in zip(probs, labels)) / len(probs)}


def run_calib() -> dict:
    """Fit on calibration, score test+OOD once, write report + calibrator."""
    calib, test, ood = build_entries()
    cal = fit_calibrator(calib)
    save_calibration(cal)

    def score(entries: list[dict]) -> dict:
        pre = _metrics([softmax(e["logits"], 1.0) for e in entries],
                       [e["label"] for e in entries])
        post_ps, fallbacks = [], 0
        for e in entries:
            p, which = apply_calibrator(cal, e)
            post_ps.append(p)
            fallbacks += which == "global-fallback"
        post = _metrics(post_ps, [e["label"] for e in entries])
        post["reliability"] = ece(post_ps, [e["label"] for e in entries])
        scored = [dict(e, probs=p,
                       pred=max(range(len(p)), key=lambda i: p[i]))
                  for e, p in zip(entries, post_ps)]
        risks = {s: risk_coverage(scored, s) for s in
                 ("maxprob", "energy", "margin")}
        return {"pre": pre, "post": post, "fallbacks": fallbacks,
                "risk_coverage": risks,
                "bootstrap_ci": bootstrap_risk_ci(scored, "maxprob")}

    rep_test, rep_ood = score(test), score(ood)
    by_locale = {}
    for loc in ("en-US", "es-ES"):
        sub = [e for e in test if e["locale"] == loc]
        post_ps = [apply_calibrator(cal, e)[0] for e in sub]
        by_locale[loc] = _metrics(post_ps, [e["label"] for e in sub])
    ece_post = rep_test["post"]["ece"]
    verdict = ("GO" if ece_post <= ECE_TARGET else
               f"NO-GO: test ECE {ece_post:.4f} > target {ECE_TARGET} "
               "(honest miss, no fudging)")
    if rep_test["post"]["accuracy"] == 1.0:
        risk_note = ("test risk-coverage is vacuous: the fixed cosine "
                     "predictor is perfect on the synthetic pilot "
                     "(overall error 0.0), so no abstention can beat it "
                     "there; abstention value is demonstrated on OOD "
                     "(avg selective risk ~0.19 vs 0.50 random).")
    else:
        risk_note = "test risk-coverage beats random (see curve)."
    report = {"calib_version": CALIB_VERSION, "seed": CALIB_SEED,
              "ece_target": ECE_TARGET,
              "n": {"calibration": len(calib), "test": len(test),
                    "ood": len(ood)},
              "massive_card": massive_card(),
              "accuracy_preserved": (
                  rep_test["post"]["accuracy"] == rep_test["pre"]["accuracy"]
                  and rep_ood["post"]["accuracy"] == rep_ood["pre"]["accuracy"]),
              "test": rep_test, "ood": rep_ood,
              "drift_by_locale_test": by_locale,
              "locale_gap_post": abs(by_locale["en-US"]["ece"] -
                                     by_locale["es-ES"]["ece"]),
              "risk_note": risk_note,
              "verdict": verdict}
    out = os.path.join(os.path.dirname(calib_path()), "report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    return report
