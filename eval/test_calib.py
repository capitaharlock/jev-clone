"""Unit tests for eval.calib (#T-calib)."""
import json
import math
import os
import tempfile
import unittest

from .calib import (
    apply_calibrator,
    brier,
    bootstrap_risk_ci,
    build_entries,
    confidence_scores,
    ece,
    fit_calibrator,
    fit_temperature,
    load_calibration,
    nll,
    risk_coverage,
    save_calibration,
    softmax,
)


def _entries(split="calibration", n=40, seed=1):
    import random
    rng = random.Random(seed)
    out = []
    for i in range(n):
        # Informative confidence on every strategy: right 70% with
        # high-norm logits, wrong 30% with low-norm logits.
        y = 0 if rng.random() < 0.7 else 1
        logits = [2.0, 0.0] if y == 0 else [0.1, 0.0]
        m = max(logits)
        exps = [math.exp(l - m) for l in logits]
        s = sum(exps)
        p = [e / s for e in exps]
        out.append({"id": f"e{i}", "split": split, "locale": "en-US",
                    "cardinality": 2, "logits": logits, "label": y,
                    "pred": 0, "probs": p})
    return out


class TestMetrics(unittest.TestCase):
    def test_nll_known_value(self):
        self.assertAlmostEqual(nll([[[0.7, 0.3]][0]], [0]),
                               -math.log(0.7), places=9)

    def test_brier_known_value(self):
        # probs [0.7,0.3], label 0: .09+.09=0.18
        self.assertAlmostEqual(brier([[0.7, 0.3]], [0]), 0.18, places=9)

    def test_ece_perfect_is_zero(self):
        probs = [[1.0, 0.0]] * 10 + [[0.0, 1.0]] * 10
        labels = [0] * 10 + [1] * 10
        self.assertAlmostEqual(ece(probs, labels)["ece"], 0.0, places=9)

    def test_ece_empty_bins_kept(self):
        r = ece([[0.9, 0.1]], [0], n_bins=15)
        self.assertEqual(len(r["bins"]), 15)
        self.assertEqual(sum(b["n"] for b in r["bins"]), 1)
        empty = [b for b in r["bins"] if b["n"] == 0]
        self.assertTrue(empty)
        for b in empty:
            self.assertEqual(b["acc"], 0.0)

    def test_softmax_sums_to_one(self):
        self.assertAlmostEqual(sum(softmax([2.0, 1.0, 0.1], 0.5)), 1.0)


class TestFitting(unittest.TestCase):
    def test_fit_reduces_nll(self):
        es = _entries()
        res = fit_temperature(es)
        self.assertLess(res["nll_after"], res["nll_before"])
        self.assertGreater(res["temperature"], 0.0)

    def test_fit_refuses_test_split(self):
        es = _entries() + _entries(split="test", n=5)
        with self.assertRaises(ValueError):
            fit_temperature(es)

    def test_fit_refuses_ood_split(self):
        with self.assertRaises(ValueError):
            fit_temperature(_entries(split="ood", n=5))

    def test_temperature_preserves_argmax(self):
        es = _entries()
        res = fit_temperature(es)
        t = res["temperature"]
        for e in es:
            pre = max(range(2), key=lambda i: softmax(e["logits"], 1.0)[i])
            post = max(range(2), key=lambda i: softmax(e["logits"], t)[i])
            self.assertEqual(pre, post)


class TestSerialization(unittest.TestCase):
    def test_roundtrip_and_apply(self):
        es = _entries()
        cal = fit_calibrator(es)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "calibration.json")
            save_calibration(cal, p)
            back = load_calibration(p)
        self.assertEqual(back["global"]["temperature"],
                         cal["global"]["temperature"])
        for e in es:
            p1, _ = apply_calibrator(cal, e)
            p2, _ = apply_calibrator(back, e)
            self.assertAlmostEqual(p1[0], p2[0], places=12)

    def test_unseen_locale_uses_cardinality_then_global(self):
        es = _entries()
        cal = fit_calibrator(es)
        other = dict(es[0], locale="xx-YY")
        _, which = apply_calibrator(cal, other)
        self.assertEqual(which, "cardinality")
        alien = dict(es[0], locale="xx-YY", cardinality=5)
        _, which = apply_calibrator(cal, alien)
        self.assertEqual(which, "global-fallback")


class TestAbstention(unittest.TestCase):
    def test_risk_coverage_beats_random(self):
        es = _entries(n=200)
        for s in ("maxprob", "energy", "margin"):
            r = risk_coverage(es, s)
            self.assertTrue(r["beats_random"], s)
            self.assertAlmostEqual(r["random_risk"], r["overall_error"])

    def test_bootstrap_ci_shape(self):
        es = _entries(n=100)
        ci = bootstrap_risk_ci(es, "maxprob")
        for cov, v in ci.items():
            lo, hi = v["ci95"]
            self.assertLessEqual(lo, v["risk"])
            self.assertLessEqual(v["risk"], hi)
            self.assertLessEqual(lo, hi)

    def test_unknown_strategy_rejected(self):
        with self.assertRaises(ValueError):
            confidence_scores([1.0, 0.0], [0.9, 0.1], "nope")


class TestEntries(unittest.TestCase):
    def test_split_sizes_and_tags(self):
        calib, test, ood = build_entries()
        self.assertEqual(len(calib), 100)
        self.assertEqual(len(test), 400)
        self.assertTrue(all(e["split"] == "calibration" for e in calib))
        self.assertTrue(all(e["split"] == "test" for e in test))
        self.assertTrue(all(e["split"] == "ood" for e in ood))
        self.assertEqual(len(ood), 24)
        ids = [e["id"] for e in calib + test]
        self.assertEqual(len(set(ids)), len(ids))

    def test_end_to_end_fit_only_calibration(self):
        calib, test, _ = build_entries()
        cal = fit_calibrator(calib)
        self.assertEqual(cal["fit_split"], "calibration")
        self.assertEqual(cal["n_fit"], 100)
        with self.assertRaises(ValueError):
            fit_calibrator(calib + test)


if __name__ == "__main__":
    unittest.main()
