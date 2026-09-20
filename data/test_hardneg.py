"""Tests for #T-hardneg (stdlib unittest)."""
from __future__ import annotations

import unittest

from .firewall import ContaminationScanner, canary_texts
from .hardneg import (
    OOD_GENERATORS,
    firewall_screen,
    gen_contradictory,
    gen_corrupt,
    gen_missing_answer,
    gen_unrelated,
    nearest_labels,
    pools_from_fixtures,
    run_hardneg,
    sample_hardneg,
    unknown_metrics,
    _demo_items,
)


def _item():
    return {"state": "notice: harbor status is open",
            "question": "what is harbor status?",
            "options": [{"id": "o0", "text": "harbor status is open"},
                        {"id": "o1", "text": "harbor status is closed"},
                        {"id": "o2", "text": "harbor status is delayed"}],
            "answer": "o0", "id": "d00"}


class TestSampler(unittest.TestCase):
    def test_nearest_label_monotonic(self):
        pool = ["card_declined", "card_stolen", "card_balance",
                "mortgage_rate", "weather_today", "pizza_topping"]
        near = nearest_labels("card_lost", pool, 5)
        self.assertEqual(len(near), 5)
        self.assertNotIn("card_lost", [c["label"] for c in near])
        diffs = [c["difficulty"] for c in near]
        self.assertEqual(diffs, sorted(diffs, reverse=True))

    def test_seeded_reproducible(self):
        pools = {"a": ["alpha", "alpine", "beta", "gamma", "delta"]}
        s1 = sample_hardneg("alpha", pools, "a", seed=7)
        s2 = sample_hardneg("alpha", pools, "a", seed=7)
        self.assertEqual(s1, s2)
        s3 = sample_hardneg("alpha", pools, "a", seed=8)
        self.assertIsNot(s1, s3)

    def test_cross_dataset(self):
        pools = pools_from_fixtures()
        own = next(d for d in sorted(pools) if len(pools[d]) >= 4)
        gold = sorted(pools[own])[0]
        s = sample_hardneg(gold, pools, own)
        self.assertTrue(s["in_domain"])
        self.assertTrue(s["cross_dataset"])
        self.assertTrue(all(c["from"] != own for c in s["cross_dataset"]))


class TestOODGenerators(unittest.TestCase):
    def test_all_four_kinds(self):
        it = _item()
        memos = ["unrelated memo clouds"]
        got = {
            "missing-answer": gen_missing_answer(it, 1),
            "unrelated": gen_unrelated(it, memos, 1),
            "corrupt": gen_corrupt(it, 1),
            "contradictory": gen_contradictory(it, 1),
        }
        self.assertEqual(set(got), set(OOD_GENERATORS))
        for kind, g in got.items():
            self.assertEqual(g["kind"], kind)
            self.assertEqual(g["answer"], "unknown")

    def test_missing_answer_drops_gold(self):
        g = gen_missing_answer(_item(), 1)
        self.assertNotIn("o0", [o["id"] for o in g["options"]])

    def test_corrupt_shortens(self):
        g = gen_corrupt(_item(), 1)
        self.assertLess(len(g["state"]), len(_item()["state"]))


class TestFirewallRescan(unittest.TestCase):
    def test_real_canary_rejected(self):
        scanner = ContaminationScanner()
        hit, _ = scanner.scan(canary_texts()[0])
        self.assertTrue(hit)
        _, rej = firewall_screen([{"state": canary_texts()[0],
                                   "question": "q?", "options": [],
                                   "answer": "unknown", "seed": 0}])
        self.assertEqual(len(rej), 1)

    def test_clean_demo_passes(self):
        demo, memos = _demo_items(442)
        items = [gen_missing_answer(d, 442 + i) for i, d in enumerate(demo)]
        clean, rej = firewall_screen(items)
        self.assertEqual(rej, [])
        self.assertEqual(len(clean), len(items))


class TestMetrics(unittest.TestCase):
    def test_unknown_recall_reasonable(self):
        demo, memos = _demo_items(442)
        ood = [gen_missing_answer(d, 442 + i) for i, d in enumerate(demo)]
        m = unknown_metrics(demo, ood)
        self.assertGreaterEqual(m["unknown_recall"], 0.0)
        self.assertLessEqual(m["unknown_recall"], 1.0)
        self.assertIn("high_confidence_errors", m)

    def test_registry_end_to_end(self):
        reg = run_hardneg()
        self.assertTrue(reg["canary_rejected"])
        self.assertEqual(reg["n_rejected"], 0)
        self.assertGreater(reg["n_ood"], 0)
        self.assertEqual(len(reg["ood_kinds"]), 4)


if __name__ == "__main__":
    unittest.main()
