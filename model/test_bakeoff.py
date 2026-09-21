"""Tests for #T-bakeoff / #T-bakeoff-real (stdlib unittest).

Two halves, deliberately:

* the harness sanity checks (proxies, markers, masks, bench stability) —
  pure stdlib, no torch, no weights;
* the **publication gate** over the artifact that #T-bakeoff-real
  publishes. `verify_report` is the only authority, and these tests run
  it over the real `artifacts/gates/T-bakeoff/report.json` plus over
  poisoned copies of it: a Pareto row that is `pending_weights`, one that
  is `random_init`, one without lineage. Any of those must refuse
  publication.
"""
from __future__ import annotations

import copy
import os
import unittest

from .bakeoff import (
    MINIMUM_SCOPE,
    PROXY_IDS,
    REAL_CANDIDATES,
    REPORT_PATH,
    SMOKE_PROXIES,
    assign_markers,
    availability,
    build_bench,
    fixed_benchmark_hash,
    license_gate,
    load_report,
    pad_masks,
    run_candidate,
    run_sanity_checks,
    score_options,
    verify_report,
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

    def test_unavailable_candidates_carry_a_reason(self):
        """STUB rule: an external dependency may be missing — with a why."""
        for cid in ("lfm2.5-230m", "neobert-250m"):
            cand = next(c for c in REAL_CANDIDATES if c["id"] == cid)
            avail = availability(cand)
            self.assertEqual(avail["status"], "not_available")
            self.assertTrue(avail["reason"])


class TestSanityChecks(unittest.TestCase):
    def test_proxies_live_outside_the_pareto(self):
        sanity = run_sanity_checks()
        self.assertEqual({m["id"] for m in sanity["harness_proxies"]},
                         {p["id"] for p in SMOKE_PROXIES})
        for m in sanity["harness_proxies"]:
            self.assertEqual(m["status"], "harness_only")
            self.assertFalse(m["comparable"])
            self.assertFalse(m["pareto_eligible"])
            self.assertIn("not_comparable_because", m)
            for k in ("accuracy", "nll", "p50_ms", "p95_ms"):
                self.assertIn(k, m)
        self.assertEqual(len(sanity["harness_rank"]), len(SMOKE_PROXIES))


@unittest.skipUnless(os.path.exists(REPORT_PATH),
                     f"{REPORT_PATH} not published yet")
class TestPublicationGate(unittest.TestCase):
    """`verify_report` is what may refuse to publish. Run it for real."""

    def setUp(self):
        self.report = load_report()

    def test_published_report_passes_its_own_gate(self):
        checks = verify_report(self.report)
        failed = {n: c["offenders"] for n, c in checks.items()
                  if not c["pass"]}
        self.assertEqual(failed, {})
        self.assertTrue(self.report["pass"])

    def test_pending_weights_row_refuses_publication(self):
        """The check the task asks for, stated as a test."""
        poisoned = copy.deepcopy(self.report)
        poisoned["pareto"]["rows"].append(
            {"id": "lfm2.5-230m", "status": "pending_weights"})
        checks = verify_report(poisoned)
        self.assertFalse(checks["no_proxy_in_pareto"]["pass"])
        self.assertIn("lfm2.5-230m", checks["no_proxy_in_pareto"]["offenders"])

    def test_random_init_row_refuses_publication(self):
        poisoned = copy.deepcopy(self.report)
        poisoned["pareto"]["rows"].append(
            {"id": "proxy-L", "status": "random_init"})
        checks = verify_report(poisoned)
        self.assertFalse(checks["no_proxy_in_pareto"]["pass"])
        self.assertFalse(checks["pareto_lineage"]["pass"])

    def test_row_without_lineage_refuses_publication(self):
        poisoned = copy.deepcopy(self.report)
        poisoned["pareto"]["rows"][0]["lineage"]["weights_sha256"] = {}
        checks = verify_report(poisoned)
        self.assertFalse(checks["pareto_lineage"]["pass"])

    def test_row_without_run_id_refuses_publication(self):
        poisoned = copy.deepcopy(self.report)
        poisoned["pareto"]["rows"][0]["lineage"]["run_id"] = None
        checks = verify_report(poisoned)
        self.assertFalse(checks["pareto_lineage"]["pass"])

    def test_imputed_metric_refuses_publication(self):
        poisoned = copy.deepcopy(self.report)
        poisoned["pareto"]["rows"][0]["quality"]["unseen_ece"] = None
        checks = verify_report(poisoned)
        self.assertFalse(checks["measured_not_imputed"]["pass"])

    def test_no_proxy_id_anywhere_on_the_pareto(self):
        for row in self.report["pareto"]["rows"]:
            self.assertNotIn(row["id"], PROXY_IDS)
            self.assertEqual(row["status"], "trained")
        self.assertTrue(self.report["sanity_checks"]["harness_proxies"])

    def test_every_pareto_row_carries_lineage_and_numbers(self):
        for row in self.report["pareto"]["rows"]:
            lin = row["lineage"]
            self.assertTrue(lin["run_id"])
            self.assertTrue(lin["checkpoint"])
            self.assertTrue(all(len(h) == 64 for h in
                                lin["weights_sha256"].values()))
            self.assertEqual(len(lin["head_weights_sha256"]), 64)
            self.assertIsInstance(row["quality"]["unseen_accuracy"], float)
            self.assertIsInstance(row["quality"]["unseen_ece"], float)
            for device in ("mps", "cpu"):
                lat = row["latency"][device]
                self.assertEqual(lat["k"], 4)
                self.assertGreater(lat["p95_ms"], 0.0)
            self.assertGreater(row["cost"]["backbone_params"], 0)
            self.assertGreater(row["cost"]["train_seconds_to_cutoff"], 0.0)

    def test_minimum_real_scope_and_top2_prose(self):
        ids = {row["id"] for row in self.report["pareto"]["rows"]}
        self.assertTrue(MINIMUM_SCOPE.issubset(ids))
        self.assertTrue(self.report["pareto"]["top2"])
        self.assertIn("Top-2", self.report["pareto"]["verdict"])

    def test_unavailable_candidates_are_declared_not_imputed(self):
        for entry in self.report["not_available"]:
            self.assertEqual(entry["status"], "not_available")
            self.assertTrue(entry["reason"])
            self.assertIn(entry["id"], self.report["why_others_lose"])


if __name__ == "__main__":
    unittest.main()
