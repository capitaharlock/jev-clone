"""Tests for #T-shared-state (stdlib unittest)."""
from __future__ import annotations

import unittest

from .shared_state import (
    LENGTHS,
    POSITIONS,
    QUALITY_MARGIN,
    StateCache,
    _encode_calls,
    answer_full_cross,
    answer_shared,
    accuracy,
    build_long_state,
    build_multiq,
    reset_counters,
    run_shared_state,
)


class TestEncodeOnce(unittest.TestCase):
    def test_state_encoded_exactly_once(self):
        for nq in (1, 3, 8):
            state, questions = build_multiq(nq)
            reset_counters()
            answer_shared(state, questions)
            self.assertEqual(_encode_calls["state"], 1)

    def test_full_cross_encodes_per_question(self):
        state, questions = build_multiq(4)
        reset_counters()
        answer_full_cross(state, questions)
        self.assertEqual(_encode_calls["state"], 4)


class TestShapes(unittest.TestCase):
    def test_option_range_and_multiq(self):
        state, questions = build_multiq(5)
        probs = answer_shared(state, questions)
        self.assertEqual(len(probs), 5)
        for p, q in zip(probs, questions):
            self.assertEqual(len(p), len(q["options"]))
            self.assertAlmostEqual(sum(p), 1.0, places=9)

    def test_long_state_lengths(self):
        for n in LENGTHS:
            for pos in POSITIONS:
                text, got = build_long_state(n, pos, "harbor open")
                self.assertGreaterEqual(got, n)
                self.assertIn("harbor open", text)


class TestParity(unittest.TestCase):
    def test_cache_matches_uncached(self):
        state, questions = build_multiq(3)
        cache = StateCache()
        p_cached = answer_shared(state, questions, cache)
        reset_counters()
        p_fresh = answer_shared(state, questions)
        for a, b in zip(p_cached, p_fresh):
            for x, y in zip(a, b):
                self.assertAlmostEqual(x, y, places=9)

    def test_invalidation_by_version_and_hash(self):
        state, questions = build_multiq(2)
        cache = StateCache()
        answer_shared(state, questions, cache)
        misses = cache.misses
        answer_shared(state, questions, cache, version=2)
        self.assertEqual(cache.misses, misses + 1)
        answer_shared(state + " extra", questions, cache)
        self.assertEqual(cache.misses, misses + 2)
        k1 = StateCache.key(state, 1)
        self.assertNotEqual(k1, StateCache.key(state, 2))
        self.assertNotEqual(k1, StateCache.key(state + " extra", 1))

    def test_single_vs_batch(self):
        state, questions = build_multiq(4)
        reset_counters()
        pall = answer_shared(state, questions)
        reset_counters()
        parts = [answer_shared(state, questions[:2]),
                 answer_shared(state, questions[2:])]
        flat = [r for part in parts for r in part]
        for a, b in zip(pall, flat):
            for x, y in zip(a, b):
                self.assertAlmostEqual(x, y, places=9)


class TestQualityAndScaling(unittest.TestCase):
    def test_shared_within_margin_of_full(self):
        state, questions = build_multiq(8)
        a_shared = accuracy(answer_shared(state, questions), questions)
        a_full = accuracy(answer_full_cross(state, questions), questions)
        self.assertGreaterEqual(a_shared, a_full - QUALITY_MARGIN)

    def test_sublinear_encodes(self):
        rep = run_shared_state()
        for row in rep["q_curves"]:
            self.assertEqual(row["shared_encodes"], 1)
            self.assertEqual(row["full_encodes"], row["Q"])

    def test_v1_limit_published(self):
        rep = run_shared_state()
        self.assertIn(rep["v1_state_limit_tokens"], LENGTHS)
        self.assertEqual(len(rep["len_curves"]), len(LENGTHS))


if __name__ == "__main__":
    unittest.main()
