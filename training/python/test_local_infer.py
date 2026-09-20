"""Golden vectors for local inference (#T-local-infer).

Cross-checks the Python reference against itself across the stack §30
matrix (determinism, boolean binary, simplex), and records the device
reality of this machine: CPU executes, Metal/CUDA stay `not_available`
until real hardware validates them. No number is invented for a backend
that did not run.
"""
from __future__ import annotations

import json
import subprocess
import sys
import unittest

from training.python.parity.reference import MATRIX, case

DEVICE_INVENTORY = [
    {"device": "cpu", "available": True, "reason": None},
    {"device": "metal", "available": False,
     "reason": "metal backend not compiled in this workspace build"},
    {"device": "cuda", "available": False,
     "reason": "cuda backend not compiled in this workspace build"},
]


class LocalInferGoldenTest(unittest.TestCase):
    def test_matrix_is_deterministic(self):
        first = [case(s, d, o, False) for (_, s, d, o, _) in MATRIX]
        second = [case(s, d, o, False) for (_, s, d, o, _) in MATRIX]
        self.assertEqual(first, second)

    def test_distributions_are_simplex_and_boolean_binary(self):
        for (name, seed, dim, n, kind) in MATRIX:
            probs = case(seed, dim, n, False)
            self.assertEqual(len(probs), n, name)
            self.assertAlmostEqual(sum(probs), 1.0, places=12, msg=name)
            self.assertTrue(all(p >= 0.0 for p in probs), name)
            if kind == "boolean":
                self.assertEqual(n, 2, name)

    def test_f16_stays_close_to_f32(self):
        for (_, seed, dim, n, _) in MATRIX:
            a, b = case(seed, dim, n, False), case(seed, dim, n, True)
            self.assertLess(max(abs(x - y) for x, y in zip(a, b)), 1e-2)

    def test_rust_devices_reports_cpu_only(self):
        out = subprocess.run(
            ["cargo", "run", "--quiet", "-p", "jev-cli", "--", "devices"],
            capture_output=True, text=True, timeout=600,
        )
        self.assertEqual(out.returncode, 0, out.stderr[-1000:])
        got = json.loads(out.stdout)["devices"]
        self.assertEqual(
            [(d["device"], d["available"]) for d in got],
            [(d["device"], d["available"]) for d in DEVICE_INVENTORY],
        )


if __name__ == "__main__":
    sys.exit(unittest.main())
