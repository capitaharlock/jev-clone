"""Tests for #T-option-mixer (stdlib unittest)."""
from __future__ import annotations

import unittest

from .option_mixer import (
    INSERT_STABILITY_MIN,
    PERM_STABILITY_MIN,
    SCORERS,
    build_siblings,
    evaluate,
    insertion_stability,
    permutation_loss,
    run_option_mixer,
    score_independent,
    _vec,
)


class TestSameMatrix(unittest.TestCase):
    def test_all_scorers_same_bench(self):
        bench = build_siblings()
        for name, scorer in SCORERS.items():
            r = evaluate(scorer, bench)
            self.assertEqual(r["n"], len(bench))
            for k in ("accuracy", "perm_stability", "insert_stability",
                      "p50_ms"):
                self.assertIn(k, r)

    def test_deterministic(self):
        bench = build_siblings()
        r1 = evaluate(score_independent, bench)
        r2 = evaluate(score_independent, build_siblings())
        self.assertEqual(r1["accuracy"], r2["accuracy"])
        self.assertEqual(r1["perm_stability"], r2["perm_stability"])


class TestInvariance(unittest.TestCase):
    def test_permutation_loss_bounded(self):
        bench = build_siblings(n=6)
        for _, scorer in SCORERS.items():
            for it in bench:
                ctx = _vec(it["state"] + " " + it["question"])
                opts = [_vec(o["text"]) for o in it["options"]]
                loss = permutation_loss(scorer, ctx, opts)
                self.assertGreaterEqual(loss, 0.0)
                self.assertLessEqual(loss, 1.0)

    def test_inverse_permutation_restores(self):
        # Direct property: shuffling twice with inverse maps restores.
        bench = build_siblings(n=2)
        it = bench[0]
        ctx = _vec(it["state"])
        opts = [_vec(o["text"]) for o in it["options"]]
        base = score_independent(ctx, opts)
        perm = list(reversed(range(len(opts))))
        shuf = score_independent(ctx, [opts[i] for i in perm])
        inv = [0.0] * len(opts)
        for new_pos, old_pos in enumerate(perm):
            inv[old_pos] = shuf[new_pos]
        for a, b in zip(base, inv):
            self.assertAlmostEqual(a, b, places=9)

    def test_insertion_stability_bounded(self):
        bench = build_siblings(n=6)
        for _, scorer in SCORERS.items():
            for it in bench:
                s = insertion_stability(scorer, it)
                self.assertGreaterEqual(s, 0.0)
                self.assertLessEqual(s, 1.0)


class TestSelection(unittest.TestCase):
    def test_selected_explicit_with_cost(self):
        rep = run_option_mixer()
        self.assertIn(rep["selected"], SCORERS)
        self.assertEqual(set(rep["ranked"]), set(SCORERS))
        for name in SCORERS:
            self.assertGreater(rep["results"][name]["p50_ms"], 0.0)
        # Thresholds are published, not fitted post-hoc.
        self.assertEqual(rep["perm_stability_min"], PERM_STABILITY_MIN)
        self.assertEqual(rep["insert_stability_min"],
                         INSERT_STABILITY_MIN)


if __name__ == "__main__":
    unittest.main()
