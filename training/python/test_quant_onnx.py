"""Parity + acceptance for #T-quant-onnx: reference dtypes, INT8/INT4,
ONNX manifest buckets/invalid shapes/round-trip, keep/reject policy."""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import quant_onnx as q

GATE_ABS = 0.05
GATE_MEAN = 0.01


def w():
    return [((i * 0.37) % 1.7) - 0.85 for i in range(256)]


class TestQuantOnnx(unittest.TestCase):
    def test_fp16_bf16_within_gate(self):
        weights = w()
        for rt in (q.fp16_roundtrip, q.bf16_roundtrip):
            mx, mean = q.error_stats(weights, rt(weights))
            self.assertLessEqual(mx, GATE_ABS)
            self.assertLessEqual(mean, GATE_MEAN)

    def test_int8_dynamic_roundtrip(self):
        weights = w()
        dq = q.int8_dequantize(q.int8_quantize_dynamic(weights))
        mx, mean = q.error_stats(weights, dq)
        self.assertLessEqual(mx, GATE_ABS)
        self.assertLessEqual(mean, GATE_MEAN)

    def test_int8_per_channel_beats_per_tensor(self):
        weights = [0.01] * 64
        weights[48:] = [8.0] * 16
        mx_t, _ = q.error_stats(weights, q.int8_dequantize(q.int8_quantize_dynamic(weights)))
        per_c = q.int8_quantize_per_channel(weights, 4)
        mx_c, _ = q.error_stats(weights, q.int8_dequantize(per_c, channels=4))
        self.assertLessEqual(mx_c, mx_t)

    def test_int8_zeros_safe(self):
        self.assertEqual(q.int8_dequantize(q.int8_quantize_dynamic([0.0] * 16)), [0.0] * 16)

    def test_int4_experimental_flagged(self):
        qq = q.int4_quantize_experimental(w())
        self.assertTrue(qq["experimental"])
        self.assertEqual(len(q.int4_dequantize(qq)), 256)

    def test_onnx_buckets_roundtrip_and_invalid(self):
        weights = w()
        m = q.onnx_export_manifest(weights, [128, 512, 2048], ["cpu"])
        self.assertTrue(m["roundtrip_ok"])
        self.assertEqual(m["canonical_hash"], q.canonical_hash(weights))
        for bad in ([], [0], [1 << 20]):
            with self.assertRaises(ValueError, msg=f"{bad}"):
                q.onnx_export_manifest(weights, bad, [])

    def test_canonical_hash_stable(self):
        self.assertEqual(q.canonical_hash(w()), q.canonical_hash(w()))

    def test_parity_golden_vectors(self):
        golden = [0.0, 0.5, -0.5, 1.0]
        self.assertEqual(q.fp16_roundtrip(golden), golden)
        mx, _ = q.error_stats([1.0], q.fp16_roundtrip([1.0001]))
        self.assertLess(mx, 1e-3)
        self.assertEqual(q.bf16_roundtrip([float("inf")])[0], float("inf"))


if __name__ == "__main__":
    unittest.main()
