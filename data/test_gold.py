"""Tests for #T-gold (stdlib unittest)."""
from __future__ import annotations

import unittest

from .firewall import canary_texts
from .gold import (
    MASSIVE_LOCALES,
    PILOT_N,
    PROVENANCE,
    annotation_schema,
    benchmark_gap,
    build_pilot,
    cohen_kappa,
    collapse_to_onehot,
    exact_agree_rate,
    leakage_proof,
    massive_adapter,
    massive_card,
    p0_train_texts,
    reserve_splits,
    run_gold,
    validate_annotation,
)


class TestSchema(unittest.TestCase):
    def test_schema_contract(self):
        s = annotation_schema()
        self.assertIn("distribution", s["fields"])
        self.assertTrue(any("one-hot" in r for r in s["rules"]))

    def test_validation(self):
        ids = {"g0001"}
        ok = {"item_id": "g0001", "annotator": "a",
              "distribution": {"o0": 0.7, "o1": 0.3},
              "ambiguous": False, "notes": ""}
        self.assertEqual(validate_annotation(ok, ids), [])
        bad = dict(ok, distribution={"o0": 0.5, "o1": 0.3})
        self.assertTrue(validate_annotation(bad, ids))
        missing = dict(ok, item_id="g9999")
        self.assertTrue(validate_annotation(missing, ids))

    def test_no_onehot_collapse(self):
        with self.assertRaises(ValueError):
            collapse_to_onehot({"o0": 0.6, "o1": 0.4})


class TestPilot(unittest.TestCase):
    def test_pilot_size_and_locales(self):
        items, anns = build_pilot()
        self.assertEqual(len(items), PILOT_N)
        self.assertEqual(len(anns), 2 * PILOT_N)
        locs = {it["locale"] for it in items}
        self.assertEqual(locs, set(MASSIVE_LOCALES))
        self.assertTrue(all(it["provenance"] == PROVENANCE
                            for it in items))

    def test_iaa_recomputed(self):
        _, anns = build_pilot()
        kappa = cohen_kappa(anns)
        agree = exact_agree_rate(anns)
        self.assertGreaterEqual(kappa, -1.0)
        self.assertLessEqual(kappa, 1.0)
        # Seeded ~0.85 agreement must show, not a fabricated number.
        self.assertGreater(agree, 0.7)
        self.assertLess(agree, 1.0)

    def test_ambiguous_keeps_distribution(self):
        _, anns = build_pilot()
        amb = [a for a in anns if a["ambiguous"]]
        self.assertTrue(amb)
        for a in amb:
            self.assertAlmostEqual(sum(a["distribution"].values()), 1.0,
                                   places=9)
        # At least one annotator per ambiguous item stays spread.
        by_item: dict[str, list[dict]] = {}
        for a in amb:
            by_item.setdefault(a["item_id"], []).append(a)
        for dists in by_item.values():
            self.assertTrue(any(min(d["distribution"].values()) > 0.2
                                for d in dists))

    def test_splits_reserved(self):
        items, _ = build_pilot()
        splits = reserve_splits(items)
        self.assertEqual(len(splits["calibration"]), 100)
        self.assertEqual(len(splits["test"]), 400)


class TestMassive(unittest.TestCase):
    def test_card_pinned(self):
        card = massive_card()
        self.assertEqual(card["license"], "CC-BY-4.0")
        self.assertIn("sha256", card)
        self.assertIn("attribution", card)

    def test_adapter_and_gap(self):
        import json
        from .gold import massive_fixture_path
        rows = [json.loads(l) for l in open(massive_fixture_path())
                if l.strip()]
        intents = sorted({r["intent"] for r in rows})
        adapted = [massive_adapter(r, intents) for r in rows]
        self.assertTrue(all(a["answer"].startswith("o") for a in adapted))
        gap = benchmark_gap(adapted)
        self.assertEqual(set(gap["per_locale"]), set(MASSIVE_LOCALES))
        self.assertGreaterEqual(gap["gap"], 0.0)


class TestLeakage(unittest.TestCase):
    def test_gold_clean_vs_train(self):
        items, _ = build_pilot()
        proof = leakage_proof(items, p0_train_texts())
        self.assertTrue(proof["clean"])

    def test_canary_would_be_caught(self):
        items, _ = build_pilot()
        poisoned = list(items)
        poisoned[0] = {**items[0], "state": canary_texts()[0]}
        proof = leakage_proof(poisoned, p0_train_texts())
        self.assertFalse(proof["clean"])

    def test_end_to_end(self):
        rep = run_gold()
        self.assertEqual(rep["n_pilot"], PILOT_N)
        self.assertTrue(rep["leakage"]["clean"])
        self.assertEqual(rep["human_collection"], "pending")


if __name__ == "__main__":
    unittest.main()
