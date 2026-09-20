"""Tests for #T-curriculum (stdlib unittest)."""
from __future__ import annotations

import os
import unittest

from .curriculum import (
    STAGES,
    Head,
    bench_nll,
    brier_loss,
    build_stage_data,
    ce_loss,
    ece,
    fixed_regression_bench,
    kl_loss,
    margin,
    perm_loss,
    run_curriculum,
    sample_batches,
    total_loss,
    train_stage,
)


class TestLosses(unittest.TestCase):
    def test_terms(self):
        self.assertAlmostEqual(ce_loss(1.0, 1), 0.0, places=9)
        self.assertLess(ce_loss(0.9, 1), ce_loss(0.6, 1))
        self.assertEqual(brier_loss(1.0, 1), 0.0)
        self.assertAlmostEqual(kl_loss(0.7, 0.7), 0.0, places=9)
        self.assertGreater(kl_loss(0.5, 0.9), 0.0)
        self.assertEqual(perm_loss(0.7, 0.3), 0.0)

    def test_v2_terms_refused(self):
        with self.assertRaises(ValueError):
            total_loss(0.7, 1, {"ce": 1.0, "ordinal": 0.5})

    def test_missing_inputs_refused(self):
        with self.assertRaises(ValueError):
            total_loss(0.7, 1, {"ce": 1.0, "kl": 0.5})
        with self.assertRaises(ValueError):
            total_loss(0.7, 1, {"ce": 1.0, "perm": 0.5})

    def test_total_adds(self):
        tot, terms = total_loss(0.8, 1, {"ce": 1.0, "brier": 0.5},
                                teacher=0.9, p_swapped=0.2)
        self.assertAlmostEqual(tot, terms["ce"] + 0.5 * terms["brier"],
                               places=9)
        self.assertNotIn("kl", terms)


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_batches(self):
        data = build_stage_data()["s1-multi"]
        b1 = list(sample_batches(data, 4, 99))
        b2 = list(sample_batches(data, 4, 99))
        self.assertEqual([[i["id"] for i in b] for b in b1],
                         [[i["id"] for i in b] for b in b2])

    def test_same_seed_same_metrics(self):
        data = build_stage_data()["s1-multi"]
        r1 = train_stage(data, Head(), {"ce": 1.0}, 5050)
        r2 = train_stage(data, Head(), {"ce": 1.0}, 5050)
        self.assertEqual(r1["loss_end"], r2["loss_end"])
        self.assertEqual(r1["ece"], r2["ece"])

    def test_resume_matches_uninterrupted(self):
        data = build_stage_data()["s1-multi"]
        full = train_stage(data, Head(), {"ce": 1.0}, 5050, steps=60)
        part = train_stage(data, Head(), {"ce": 1.0}, 5050, steps=30)
        resumed = train_stage(data, Head(), {"ce": 1.0}, 5050,
                              steps=60,
                              from_checkpoint=part["checkpoint"])
        self.assertEqual(resumed["loss_end"], full["loss_end"])
        self.assertEqual(resumed["checkpoint"]["step"], 60)


class TestCurriculum(unittest.TestCase):
    def test_stages_have_contracts(self):
        data = build_stage_data()
        self.assertEqual(set(data), set(STAGES))
        for s in STAGES:
            self.assertGreater(len(data[s]), 0)

    def test_loss_decreases(self):
        data = build_stage_data()["s0-warmup"]
        rep = train_stage(data, Head(), {"ce": 1.0}, 5050)
        self.assertLess(rep["loss_end"], rep["loss_start"])

    def test_ece_recorded(self):
        rep = run_curriculum()
        for s in STAGES:
            self.assertIn("ece", rep["stages"][s])
            self.assertIn("promoted", rep["stages"][s])
        self.assertIn("ce_vs_ce_brier", rep["ablations"])
        self.assertIn("permutation", rep["ablations"])

    def test_no_promotion_on_regression(self):
        bench = fixed_regression_bench()
        h = Head()
        nll = bench_nll(h, bench)
        self.assertGreater(nll, 0.0)

    def test_run_artifacts(self):
        run_curriculum()
        root = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        self.assertTrue(os.path.isfile(
            os.path.join(root, "experiments", "curriculum.yaml")))
        self.assertTrue(os.path.isfile(
            os.path.join(root, "results", "curriculum-seed5050",
                         "manifest.json")))


if __name__ == "__main__":
    unittest.main()
