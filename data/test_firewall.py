"""Tests for #T-firewall + #T-halt-contam (stdlib unittest — no deps yet)."""
from __future__ import annotations

import ast
import json
import os
import unittest

from .firewall import (
    BENCHMARKS,
    PERTURBATIONS,
    STRESS_SEED,
    BenchmarkRegistry,
    ContaminationScanner,
    canary_corpus_hash,
    check_job_allowed,
    pinned_benchmarks,
    reject_train_rows,
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


PREFETCH_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "artifacts",
    "data-prefetch",
)


def _row_states(filename, split, limit):
    """Real `state` texts from a converted corpus (split-filtered)."""
    states = []
    path = os.path.join(PREFETCH_DIR, filename)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") == split and r.get("state"):
                states.append(r["state"])
                if len(states) >= limit:
                    break
    return states


class TestEvalOnlyBenchmarks(unittest.TestCase):
    """#T-halt-contam: logiqa/reclor are eval-only; train usage is an error."""

    def test_logiqa_reclor_registered_eval_only(self):
        reg = BenchmarkRegistry()
        self.assertIn("logiqa", reg.ids())
        self.assertIn("reclor", reg.ids())
        for card in pinned_benchmarks():
            if card.id in ("logiqa", "reclor"):
                self.assertEqual(card.usage, "eval-only")

    def test_barrier_blocks_logiqa_reclor_all_roles(self):
        reg = BenchmarkRegistry()
        for role in ("train", "teachers", "hard-negatives"):
            self.assertTrue(reg.check_train_barrier(["logiqa"], role))
            self.assertTrue(reg.check_train_barrier(["reclor"], role))

    def test_check_job_allowed_raises_not_warns(self):
        with self.assertRaises(ValueError):
            check_job_allowed("logiqa")
        with self.assertRaises(ValueError):
            check_job_allowed("reclor")
        with self.assertRaises(ValueError):
            check_job_allowed("synth-loop")
        self.assertTrue(check_job_allowed("banking77"))

    def test_logiqa_heldout_row_in_train_rejected_as_error(self):
        heldout = _row_states("logiqa.jsonl", "test", 5)
        self.assertEqual(len(heldout), 5)
        # Held-out rows smuggled into a train batch: hard error, not warning.
        with self.assertRaises(ValueError):
            reject_train_rows(heldout, blocked_texts=heldout)
        # Normalized layer: same content, different case/padding still caught.
        variant = ["  " + t.upper() + "  " for t in heldout[:2]]
        with self.assertRaises(ValueError):
            reject_train_rows(variant, blocked_texts=heldout)
        # Clean rows from another corpus pass through untouched.
        clean = _row_states("banking77.jsonl", "train", 5)
        self.assertEqual(len(clean), 5)
        self.assertTrue(reject_train_rows(clean, blocked_texts=heldout))

    def test_reclor_heldout_row_in_train_rejected_as_error(self):
        heldout = _row_states("reclor.jsonl", "test", 5)
        self.assertEqual(len(heldout), 5)
        with self.assertRaises(ValueError):
            reject_train_rows(heldout, blocked_texts=heldout)

    def test_train_baseline_jobs_exclude_eval_only(self):
        # AST parse (no sklearn import): JOBS names vs registry ids.
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "data", "train_baseline.py")) as f:
            tree = ast.parse(f.read())
        names = set()
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "JOBS" for t in node.targets
            ):
                for elt in node.value.elts:
                    for k, v in zip(elt.keys, elt.values):
                        if isinstance(k, ast.Constant) and k.value == "name":
                            names.add(v.value)
        self.assertTrue(names, "JOBS not found in train_baseline.py")
        self.assertEqual(names & set(BenchmarkRegistry().ids()), set())
        for banned in ("synth-loop", "logiqa", "reclor"):
            self.assertNotIn(banned, names)


if __name__ == "__main__":
    unittest.main()
