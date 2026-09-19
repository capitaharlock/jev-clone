"""Tests for #T-recon (stdlib unittest)."""
from __future__ import annotations

import os
import unittest

from .recon import (
    BASELINES,
    RECON_SEED,
    inventory,
    measure,
    percentile,
    run_local_baselines,
    run_smoke,
    _load_snapshot,
)


class TestLatencyProtocol(unittest.TestCase):
    def test_percentiles_on_raw(self):
        s = [float(i) for i in range(1, 101)]
        self.assertEqual(percentile(s, 50), 50.0)
        self.assertEqual(percentile(s, 95), 95.0)
        self.assertEqual(percentile(s, 99), 99.0)
        with self.assertRaises(ValueError):
            percentile([], 50)

    def test_measure_warmup_and_raw(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
        r = measure(fn, warmup=3, iters=5)
        self.assertEqual(calls["n"], 3 + 1 + 5)
        self.assertEqual(len(r["raw_s"]), 5)
        for k in ("p50", "p95", "p99"):
            self.assertIn(k, r["warm"])

    def test_tokenize_isolated_from_encode(self):
        # El protocolo mide por separado: cambiar el input cambia encode
        # pero no el coste fijo de tokenizar un literal.
        from .recon import _ngrams
        a = _ngrams("hola mundo")
        b = _ngrams("hola mundo " * 50)
        self.assertGreater(sum(b.values()), sum(a.values()))


class TestReconSmoke(unittest.TestCase):
    def test_same_snapshot_protocol(self):
        _, h1 = _load_snapshot()
        _, h2 = _load_snapshot()
        self.assertEqual(h1, h2)
        self.assertTrue(h1.startswith("sha256:"))

    def test_deterministic_table(self):
        r1 = run_smoke(seed=RECON_SEED, iters=4)
        r2 = run_smoke(seed=RECON_SEED, iters=4)
        self.assertEqual(r1["report_hash"], r2["report_hash"])
        self.assertEqual(r1["snapshot_hash"], r2["snapshot_hash"])

    def test_baselines_fixed_and_absent_marked(self):
        r = run_smoke(seed=RECON_SEED, iters=4)
        self.assertEqual(len(r["baselines"]), 8)
        self.assertEqual([b["name"] for b in BASELINES],
                         list(r["baselines"].keys()))
        for name in ("random", "majority", "char-ngram-centroid",
                     "tiny-logreg-tfidf"):
            row = r["baselines"][name]
            self.assertEqual(row["status"], "measured", name)
            self.assertGreaterEqual(row["acc"], 0.0)
            self.assertLessEqual(row["acc"], 1.0)
        for name in ("gliclass-hf", "laya", "verdict-kev-jevlike",
                     "small-llm"):
            row = r["baselines"][name]
            self.assertEqual(row["status"], "not_available", name)
            self.assertTrue(row["reason"], name)

    def test_local_beats_chance_sanity(self):
        # El smoke debe ser informativo: majority >= random en este snapshot.
        r = run_smoke(seed=RECON_SEED, iters=4)
        self.assertGreaterEqual(r["baselines"]["majority"]["acc"],
                                r["baselines"]["random"]["acc"])

    def test_report_regenerates_reference(self):
        import json
        import tempfile
        from .recon import write_report
        r = run_smoke(seed=RECON_SEED, iters=4)
        with tempfile.TemporaryDirectory() as d:
            p = write_report(r, os.path.join(d, "rep.json"))
            with open(p) as f:
                back = json.load(f)
        self.assertEqual(back["report_hash"], r["report_hash"])

    def test_inventory_present(self):
        inv = inventory()
        for k in ("machine", "os", "python", "cpu"):
            self.assertTrue(inv[k], k)


if __name__ == "__main__":
    unittest.main()
