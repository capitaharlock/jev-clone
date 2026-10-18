"""Encode-once + 100 packs benchmark for the state cache (#T-state-cache).

Runs the Rust bench binary (single source of truth for the numbers),
validates the §177 shape — one miss, N hits, one scheduler batch, GO
verdict — and records the raw-sample report under
``artifacts/gates/T-state-cache/bench.json``. No number is invented:
every field is measured on this machine by the binary.
"""
from __future__ import annotations

import json
import os
import subprocess
import unittest

ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
OUT = os.path.join(ROOT, "artifacts", "gates", "T-state-cache", "bench.json")


def run_bench() -> dict:
    out = subprocess.run(
        ["cargo", "run", "--quiet", "-p", "jev-bench", "--", "--state-cache"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout)


class StateCacheBenchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_bench()
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w") as f:
            json.dump(cls.report, f, indent=2)
            f.write("\n")

    def test_encode_once_shape(self):
        r = self.report
        self.assertEqual(r["mode"], "encode-once-100-packs")
        self.assertEqual(r["packs"], 100)
        # Exactly one miss (the encode), the rest hits.
        self.assertEqual(r["misses"], 1)
        self.assertEqual(r["hits"], 100)
        self.assertAlmostEqual(r["hit_rate"], 100.0 / 101.0, places=12)

    def test_raw_samples_and_batching(self):
        r = self.report
        self.assertEqual(len(r["warm_samples_ms"]), 100)
        self.assertTrue(all(s >= 0.0 for s in r["warm_samples_ms"]))
        self.assertLessEqual(r["warm_p50_ms"], r["warm_p95_ms"])
        self.assertLessEqual(r["warm_p95_ms"], r["warm_p99_ms"])
        self.assertGreaterEqual(r["queue_p50_ms"], 0.0)
        self.assertEqual(r["batches"], 1)

    def test_verdict_is_measured_go(self):
        r = self.report
        self.assertLess(r["warm_p50_ms"], r["smoke_threshold_ms"])
        self.assertEqual(r["verdict"], "GO")


if __name__ == "__main__":
    unittest.main()
