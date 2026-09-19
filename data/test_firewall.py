"""Tests for #T-firewall (stdlib unittest — no deps yet)."""
from __future__ import annotations

import os
import unittest

from .firewall import (
    BENCHMARKS,
    PERTURBATIONS,
    STRESS_SEED,
    BenchmarkRegistry,
    ContaminationScanner,
    canary_corpus_hash,
    pinned_benchmarks,
    run_stress_suite,
    write_raw_results,
)

GATE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "artifacts",
    "gates",
    "T-firewall",
)


class TestBenchmarkRegistry(unittest.TestCase):
    def test_pinned_hashes_match_corpus(self):
        digest = canary_corpus_hash()
        self.assertEqual(len(digest), 64)
        for card in pinned_benchmarks():
            self.assertEqual(card.sha256, digest)
            self.assertEqual(card.usage, "eval-only")

    def test_refuses_train_usage(self):
        reg = BenchmarkRegistry()
        with self.assertRaises(ValueError):
            from .firewall import BenchmarkCard

            reg.register(BenchmarkCard("evil", "rev-1", "0" * 64, usage="train"))

    def test_train_barrier_blocks_all_roles(self):
        reg = BenchmarkRegistry()
        for role in ("train", "teachers", "hard-negatives"):
            for bid in reg.ids():
                self.assertTrue(reg.check_train_barrier([bid], role), (bid, role))
        self.assertEqual(reg.check_train_barrier(["banking77"], "train"), [])

    def test_clinc_holdout_is_eval_only(self):
        reg = BenchmarkRegistry()
        self.assertIn("clinc150-oos", reg.ids())
        self.assertTrue(reg.check_train_barrier(["clinc150-oos"], "train"))

    def test_manifest_hash(self):
        m = BenchmarkRegistry().manifest()
        self.assertEqual(len(m["manifest_sha256"]), 64)
        self.assertEqual(len(m["benchmarks"]), len(BENCHMARKS))


class TestScanner(unittest.TestCase):
    def test_all_canaries_detected(self):
        scanner = ContaminationScanner()
        from .firewall import canary_texts

        missed = [t for t in canary_texts() if not scanner.scan(t)[0]]
        self.assertEqual(missed, [])

    def test_clean_below_threshold(self):
        from .firewall import clean_texts

        scanner = ContaminationScanner()
        cleans = clean_texts()
        fp = sum(1 for t in cleans if scanner.scan(t)[0])
        self.assertLessEqual(fp / len(cleans), 0.05)


class TestStressSuite(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(run_stress_suite(), run_stress_suite())
        self.assertNotEqual(
            run_stress_suite(STRESS_SEED), run_stress_suite(STRESS_SEED + 1)
        )

    def test_covers_all_perturbations(self):
        results = run_stress_suite()
        self.assertEqual(sorted(results["perturbed"]), sorted(PERTURBATIONS))

    def test_go_verdict_and_raw_results(self):
        results = run_stress_suite()
        self.assertEqual(results["verdict"], "GO")
        path = write_raw_results(results, GATE_DIR)
        self.assertTrue(os.path.isfile(path))


if __name__ == "__main__":
    unittest.main()
