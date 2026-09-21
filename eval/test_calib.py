"""Unit tests for eval.calib — the calibrator OF THE MODEL (#T-calib,
rewritten by #T-unseen-labels).

The point of the rewrite is that the predictor is the trained pointer head,
not a parameter-free cosine scorer, so these tests police two things the
old suite could not: that an artifact without a `model_version` is refused,
and that the cosine predictor is really gone from the module.
"""
import json
import math
import os
import tempfile
import unittest

from . import calib as C


def _entries(split="calibration", n=40, seed=1, dataset="banking77", k=2):
    """Informative-confidence synthetic entries (no torch, no corpus).

    Right 70 % of the time with high-norm logits, wrong 30 % with low-norm
    ones, so every abstention strategy has signal to rank on.
    """
    import random
    rng = random.Random(seed)
    out = []
    for i in range(n):
        y = 0 if rng.random() < 0.7 else 1
        logits = ([2.0, 0.0] + [0.0] * (k - 2) if y == 0
                  else [0.1, 0.0] + [0.0] * (k - 2))
        probs = C.softmax(logits, 1.0)
        out.append({"id": f"e{i}", "split": split, "cut": "seen",
                    "dataset": dataset, "cardinality": k,
                    "logits": logits, "probs": probs, "label": y,
                    "pred": max(range(k), key=lambda j: probs[j])})
    return out


PREDICTOR = {"name": "pointer-decision-head",
             "model_version": "jev-dec-test-0000", "checkpoint": "x"}


class TestMetrics(unittest.TestCase):
    def test_nll_known_value(self):
        self.assertAlmostEqual(C.nll([[0.7, 0.3]], [0]),
                               -math.log(0.7), places=9)

    def test_brier_known_value(self):
        self.assertAlmostEqual(C.brier([[0.7, 0.3]], [0]), 0.18, places=9)

    def test_ece_perfect_is_zero(self):
        probs = [[1.0, 0.0]] * 10 + [[0.0, 1.0]] * 10
        labels = [0] * 10 + [1] * 10
        self.assertAlmostEqual(C.ece(probs, labels)["ece"], 0.0, places=9)

    def test_ece_empty_bins_kept(self):
        r = C.ece([[0.9, 0.1]], [0], n_bins=15)
        self.assertEqual(len(r["bins"]), 15)
        self.assertEqual(sum(b["n"] for b in r["bins"]), 1)
        self.assertTrue([b for b in r["bins"] if b["n"] == 0])

    def test_ece_maximally_overconfident_is_one(self):
        probs = [[1.0, 0.0]] * 10
        self.assertAlmostEqual(C.ece(probs, [1] * 10)["ece"], 1.0, places=9)

    def test_softmax_sums_to_one(self):
        self.assertAlmostEqual(sum(C.softmax([2.0, 1.0, 0.1], 0.5)), 1.0)

    def test_wilson_interval_brackets_the_point(self):
        lo, hi = C.wilson_interval(19, 100)
        self.assertLess(lo, 0.19)
        self.assertGreater(hi, 0.19)
        self.assertEqual(C.wilson_interval(0, 0), (0.0, 1.0))

    def test_metrics_of_counts_unknown_as_an_option(self):
        # two rows, K = 2 real options + unknown; row 0 right, row 1 abstains
        entries = [
            {"probs": [0.8, 0.1, 0.1], "label": 0, "cardinality": 2},
            {"probs": [0.2, 0.1, 0.7], "label": 0, "cardinality": 2},
        ]
        m = C.metrics_of(entries)
        self.assertEqual(m["n"], 2)
        self.assertEqual(m["accuracy"], 0.5)
        # with `unknown` out of the race the pointer ranks both right
        self.assertEqual(m["accuracy_options_only"], 1.0)
        self.assertEqual(m["abstain_rate"], 0.5)
        self.assertAlmostEqual(m["chance"], 1 / 3, places=6)
        self.assertAlmostEqual(m["chance_options_only"], 0.5, places=6)


class TestFitting(unittest.TestCase):
    def test_fit_reduces_nll(self):
        res = C.fit_temperature(_entries())
        self.assertLess(res["nll_after"], res["nll_before"])
        self.assertGreater(res["temperature"], 0.0)

    def test_fit_refuses_test_split(self):
        with self.assertRaises(ValueError):
            C.fit_temperature(_entries() + _entries(split="test", n=5))

    def test_fit_refuses_unseen_split(self):
        with self.assertRaises(ValueError):
            C.fit_temperature(_entries(split="unseen", n=5))

    def test_temperature_preserves_argmax(self):
        t = C.fit_temperature(_entries())["temperature"]
        for e in _entries():
            pre = max(range(2), key=lambda i: C.softmax(e["logits"], 1.0)[i])
            post = max(range(2), key=lambda i: C.softmax(e["logits"], t)[i])
            self.assertEqual(pre, post)

    def test_calibrator_refuses_predictor_without_model_version(self):
        for bad in ({}, {"name": "x"}, {"name": "x", "model_version": ""},
                    None):
            with self.assertRaises(ValueError):
                C.fit_calibrator(_entries(), bad)

    def test_calibrator_carries_model_version(self):
        cal = C.fit_calibrator(_entries(), PREDICTOR)
        self.assertEqual(cal["model_version"], PREDICTOR["model_version"])
        self.assertEqual(cal["predictor"]["name"], "pointer-decision-head")
        self.assertEqual(cal["fit_split"], "calibration")
        self.assertIn("seen labels only", cal["fit_cut"])


class TestSerialization(unittest.TestCase):
    def test_roundtrip_and_apply(self):
        es = _entries()
        cal = C.fit_calibrator(es, PREDICTOR)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "calibration.json")
            C.save_calibration(cal, p)
            back = C.load_calibration(p)
        self.assertEqual(back["global"]["temperature"],
                         cal["global"]["temperature"])
        for e in es:
            self.assertAlmostEqual(C.apply_calibrator(cal, e)[0][0],
                                   C.apply_calibrator(back, e)[0][0],
                                   places=12)

    def test_hierarchy_falls_back_dataset_then_cardinality_then_global(self):
        cal = C.fit_calibrator(_entries(), PREDICTOR)
        self.assertEqual(C.apply_calibrator(cal, _entries()[0])[1], "dataset")
        other = dict(_entries()[0], dataset="logiqa-mc")
        self.assertEqual(C.apply_calibrator(cal, other)[1], "cardinality")
        alien = dict(other, cardinality=7)
        self.assertEqual(C.apply_calibrator(cal, alien)[1], "global-fallback")

    def test_default_path_is_the_model_calibrator_not_the_legacy_one(self):
        self.assertEqual(C.calib_path(), C.MODEL_CALIB)
        self.assertNotEqual(C.MODEL_CALIB, C.LEGACY_COSINE_CALIB)
        self.assertIn("T-unseen-labels", C.MODEL_CALIB)


class TestCosineScorerIsGone(unittest.TestCase):
    """Finding F: the old module calibrated a parameterless scorer."""

    def test_no_cosine_predictor_left_in_the_module(self):
        for gone in ("predict_probs", "logits_of", "_cos_text", "PREDICTOR"):
            self.assertFalse(hasattr(C, gone),
                             f"{gone} is the cosine scorer path, it must "
                             "not come back")

    def test_source_does_not_import_the_cosine_scorer(self):
        src = open(os.path.join(os.path.dirname(__file__), "calib.py")).read()
        self.assertNotIn("_cos_text", src)
        self.assertNotIn("from data.gold import", src)
        # the only mention left may be the `supersedes` card
        self.assertEqual(src.count("cosine-char3-softmax"), 1)

    def test_predictor_card_names_the_checkpoint(self):
        manifest = {"model_version": "mv-1", "backbone": {"id": "ettin-68m"},
                    "architecture": {"d_model": 256}, "samples_seen": 10,
                    "tokenizer_hash": "abc"}
        card = C.predictor_card(manifest, C.ROOT + "/artifacts/ckpt")
        self.assertEqual(card["name"], "pointer-decision-head")
        self.assertEqual(card["model_version"], "mv-1")
        self.assertEqual(card["supersedes"]["name"], "cosine-char3-softmax")
        C._require_model(card)


class TestAbstention(unittest.TestCase):
    def test_unknown_strategy_is_one_minus_p_unknown(self):
        probs = [0.2, 0.1, 0.7]
        self.assertAlmostEqual(
            C.confidence_scores([0.0, 0.0, 0.0], probs, "unknown"), 0.3)

    def test_risk_coverage_beats_random(self):
        es = _entries(n=200)
        for s in ("maxprob", "margin", "energy"):
            r = C.risk_coverage(es, s)
            self.assertTrue(r["beats_random"], s)
            self.assertAlmostEqual(r["random_risk"], r["overall_error"])

    def test_risk_coverage_curve_is_full_coverage_at_the_end(self):
        r = C.risk_coverage(_entries(n=50), "maxprob")
        self.assertAlmostEqual(r["curve"][-1]["coverage"], 1.0)
        self.assertAlmostEqual(r["curve"][-1]["risk"], r["overall_error"],
                               places=6)

    def test_bootstrap_ci_shape(self):
        ci = C.bootstrap_risk_ci(_entries(n=100), "maxprob", b=100)
        for cov, v in ci.items():
            lo, hi = v["ci95"]
            self.assertLessEqual(lo, hi)
            self.assertEqual(v["coverage"], float(cov))

    def test_unknown_strategy_name_rejected(self):
        with self.assertRaises(ValueError):
            C.confidence_scores([1.0, 0.0], [0.9, 0.1], "nope")

    def test_thresholds_cover_every_strategy_and_refuse_test_split(self):
        es = _entries()
        glob = C.fit_temperature(es)
        th = C.fit_thresholds(es, glob)
        self.assertEqual(sorted(th), sorted(C.STRATEGIES))
        with self.assertRaises(ValueError):
            C.fit_thresholds(_entries(split="test"), glob)


class TestCalibratedEntries(unittest.TestCase):
    def test_calibrated_recomputes_probs_and_records_the_source(self):
        es = _entries()
        cal = C.fit_calibrator(es, PREDICTOR)
        out = C.calibrated(cal, es)
        self.assertEqual(len(out), len(es))
        for e in out:
            self.assertAlmostEqual(sum(e["probs"]), 1.0, places=9)
            self.assertEqual(e["temp_source"], "dataset")
        self.assertIsNot(out[0], es[0])

    def test_legacy_artifact_is_left_untouched_on_disk(self):
        """The cosine calibration.json is frozen #T-release evidence."""
        if not os.path.exists(C.LEGACY_COSINE_CALIB):
            self.skipTest("legacy artifact not in this checkout")
        legacy = json.load(open(C.LEGACY_COSINE_CALIB))
        self.assertEqual(legacy["predictor"]["name"], "cosine-char3-softmax")


if __name__ == "__main__":
    unittest.main()
