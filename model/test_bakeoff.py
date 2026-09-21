"""Tests for #T-bakeoff (stdlib unittest)."""
from __future__ import annotations

import unittest

from .bakeoff import (
    REAL_CANDIDATES,
    SMOKE_PROXIES,
    assign_markers,
    build_bench,
    fixed_benchmark_hash,
    license_gate,
    pad_masks,
    run_bakeoff,
    run_candidate,
    score_options,
    _ngram_vec,
)


class TestMarkersAndMasks(unittest.TestCase):
    def test_markers_restore_order(self):
        bench = build_bench(n=8)
        for it in bench:
            markers, restore = assign_markers(it["options"], 173)
            self.assertEqual(len(markers), len(it["options"]))
            self.assertEqual(len(set(markers)), len(markers))
            self.assertEqual(sorted(restore.values()),
                             sorted(o["id"] for o in it["options"]))

    def test_scorer_order_invariant(self):
        bench = build_bench(n=4)
        it = bench[0]
        sv = _ngram_vec(it["state"], 256)
        base = score_options(sv, it["question"], it["options"], 256)
        rev = list(reversed(it["options"]))
        got = score_options(sv, it["question"], rev, 256)
        self.assertEqual([round(x, 9) for x in got],
                         [round(x, 9) for x in reversed(base)])

    def test_masks_variable_k(self):
        bench = build_bench()
        ks = {len(it["options"]) for it in bench}
        self.assertGreaterEqual(min(ks), 2)
        self.assertGreater(len(ks), 5)
        max_k = max(ks)
        for it in bench:
            mask = pad_masks(len(it["options"]), max_k)
            self.assertEqual(len(mask), max_k)
            self.assertEqual(sum(mask), len(it["options"]))


class TestComparability(unittest.TestCase):
    def test_same_seed_same_metrics(self):
        bench = build_bench()
        m1 = run_candidate(bench, 256, 173)
        m2 = run_candidate(build_bench(), 256, 173)
        for k in ("accuracy", "nll"):
            self.assertEqual(m1[k], m2[k])
        for k in ("p50_ms", "p95_ms"):
            self.assertGreater(m1[k], 0.0)
            self.assertGreater(m2[k], 0.0)

    def test_bench_hash_stable(self):
        self.assertEqual(fixed_benchmark_hash(build_bench()),
                         fixed_benchmark_hash(build_bench()))


class TestLicenseGate(unittest.TestCase):
    def test_lfm_conditional(self):
        lfm = next(c for c in REAL_CANDIDATES if c["id"].startswith("lfm"))
        g = license_gate(lfm)
        self.assertEqual(g["verdict"], "conditional")

    def test_neobert_blocked_remote_code(self):
        neo = next(c for c in REAL_CANDIDATES if c["id"].startswith("neo"))
        g = license_gate(neo)
        self.assertEqual(g["verdict"], "blocked")

    def test_permissive_clear(self):
        ett = next(c for c in REAL_CANDIDATES if c["id"].startswith("ettin"))
        self.assertEqual(license_gate(ett)["verdict"], "clear")


class TestReport(unittest.TestCase):
    def test_rank_explicit_and_real_candidates_excluded(self):
        rep = run_bakeoff()
        self.assertEqual(len(rep["harness_rank"]), len(SMOKE_PROXIES))
        self.assertTrue(all(m["status"] == "harness_only"
                            for m in rep["measured"]))
        self.assertEqual({m["id"] for m in rep["measured"]},
                         {p["id"] for p in SMOKE_PROXIES})
        for p in rep["pending_real"]:
            self.assertIn(p["status"], ("pending_weights", "weights_local"))
            self.assertIn(p["id"], rep["why_others_lose"])
        for m in rep["measured"]:
            for k in ("accuracy", "nll", "p50_ms", "p95_ms"):
                self.assertIn(k, m)

    def test_proxies_can_never_enter_a_pareto(self):
        """#T-torch-stack: a random-weight proxy is not a backbone."""
        rep = run_bakeoff()
        self.assertFalse(rep["comparable"])
        self.assertEqual(rep["pareto"]["eligible"], [])
        self.assertNotIn("top2", rep)
        for m in rep["measured"]:
            self.assertFalse(m["comparable"])
            self.assertFalse(m["pareto_eligible"])
            self.assertIn("not_comparable_because", m)
            self.assertIn(m["id"], rep["why_others_lose"])

    def test_local_weights_reported_with_hash(self):
        """A materialised backbone shows its sha256, never a metric."""
        rep = run_bakeoff()
        local = [p for p in rep["pending_real"]
                 if p["status"] == "weights_local"]
        for p in local:
            self.assertTrue(p["weights_sha256"])
            self.assertTrue(all(len(h) == 64
                                for h in p["weights_sha256"].values()))
            for metric in ("accuracy", "nll", "p50_ms"):
                self.assertNotIn(metric, p)


if __name__ == "__main__":
    unittest.main()
