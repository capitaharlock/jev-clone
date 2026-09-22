"""Tests for the synthetic factory V1 (#T-synth-factory gate, stdlib unittest)."""
from __future__ import annotations

import unittest

from tools.data_factory.council import judge, teacher_router
from tools.data_factory.deduper import Deduper, group_split
from tools.data_factory.validator import validate_record


def good_rec() -> dict:
    return {
        "state": "El contrato establece un plazo de treinta días para la entrega.",
        "questions": [{
            "id": "q1", "kind": "choice",
            "options": [{"id": "a", "text": "Treinta días"}, {"id": "b", "text": "Siete días"}],
            "answer": "a",
        }],
        "split": "train",
        "parent_example_id": "p1",
    }


class TestValidator(unittest.TestCase):
    def test_accepts_good(self):
        self.assertEqual(validate_record(good_rec()), [])

    def test_rejects_gold_outside_candidates(self):
        r = good_rec()
        r["questions"][0]["answer"] = "zzz"
        self.assertTrue(any("gold" in e for e in validate_record(r)))

    def test_rejects_k1(self):
        r = good_rec()
        r["questions"][0]["options"] = [{"id": "a", "text": "Solo una"}]
        self.assertTrue(any("K=" in e for e in validate_record(r)))

    def test_rejects_k_over_255(self):
        r = good_rec()
        r["questions"][0]["options"] = [
            {"id": f"o{i}", "text": f"opt {i}"} for i in range(256)]
        self.assertTrue(any("K=" in e for e in validate_record(r)))

    def test_rejects_duplicate_ids(self):
        r = good_rec()
        r["questions"].append(dict(r["questions"][0]))
        self.assertTrue(any("duplicate question id" in e for e in validate_record(r)))

    def test_rejects_cot(self):
        r = good_rec()
        r["chain_of_thought"] = "should never be stored"
        self.assertTrue(any("CoT" in e for e in validate_record(r)))

    def test_rejects_missing_parent(self):
        r = good_rec()
        del r["parent_example_id"]
        self.assertTrue(any("parent_example_id" in e for e in validate_record(r)))


class TestCouncil(unittest.TestCase):
    def test_agreement_measured_and_disputes_adjudicated(self):
        agreed = adjudicated = 0
        for seq in range(200):
            v = teacher_router("El texto habla de plazos de entrega.", f"k{seq}", seq)
            self.assertIn(v.answer, [o["id"] for o in v.question["options"]])
            agreed += v.agreed
            adjudicated += v.adjudicated
        self.assertGreater(agreed, 0)
        self.assertEqual(agreed + adjudicated, 200)

    def test_judge_passthrough_on_agreement(self):
        q = {"gold": "a", "options": [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}]}
        self.assertEqual(judge("s", q, "a", "a", "k"), "a")
        self.assertIn(judge("s", q, "a", "b", "k"), ("a", "b"))

    def test_roles_inverted_half_the_time(self):
        inv = sum(1 for seq in range(100)
                  if teacher_router("estado de prueba", f"inv{seq}", seq).roles_inverted)
        self.assertEqual(inv, 50)


class TestDedupeSplit(unittest.TestCase):
    def test_dedupe_kills_exact_and_semantic(self):
        d = Deduper()
        r1 = good_rec()
        self.assertIsNone(d.check(r1, "p1"))
        self.assertEqual(d.check(dict(r1), "p1"), "exact")
        r2 = good_rec()
        r2["state"] = "El contrato establece un plazo de treinta días para la entrega total"
        r2["parent_example_id"] = "p2"
        self.assertEqual(d.check(r2, "p2"), "semantic")

    def test_group_split_deterministic_and_spread(self):
        self.assertEqual(group_split("same-parent"), group_split("same-parent"))
        splits = {group_split(f"parent-{i}") for i in range(200)}
        self.assertGreater(len(splits), 1)


if __name__ == "__main__":
    unittest.main()
