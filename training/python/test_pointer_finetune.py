"""Views and loss for the pointer LoRA trainer. No backbone is loaded."""

import random
import unittest

import mlx.core as mx

from training.python import pointer_finetune as PF


def episode():
    return {
        "id": "e1",
        "lang": "en",
        "family": "attribute_comparison",
        "variant_group": "rule-1",
        "answer": "c2",
        "candidates": [
            {"id": "c1", "text": "alpha"},
            {"id": "c2", "text": "beta"},
            {"id": "c3", "text": "gamma"},
        ],
    }


class Scripted(random.Random):
    def __init__(self, draws):
        super().__init__(0)
        self.draws = list(draws)

    def random(self):
        return self.draws.pop(0)


class ViewTests(unittest.TestCase):
    def test_none_drop_removes_the_gold_option(self):
        cands, gold = PF.decision_view(
            episode(), Scripted([0.0, 0.0]), p_none=0.25, shuffle=False)
        self.assertEqual(gold, ["n0"])
        self.assertNotIn("c2", [c["id"] for c in cands])
        self.assertEqual(cands[-1]["text"], PF.NONE_TEXT["en"])
        self.assertGreaterEqual(len(cands), 2)

    def test_none_keep_adds_a_distractor(self):
        cands, gold = PF.decision_view(
            episode(), Scripted([0.0, 0.9]), p_none=0.25, shuffle=False)
        self.assertEqual(gold, ["c2"])
        self.assertEqual([c["id"] for c in cands], ["c1", "c2", "c3", "n0"])

    def test_shuffle_keeps_each_text_with_its_id(self):
        rng = random.Random(7)
        cands, gold = PF.decision_view(
            episode(), rng, p_none=0.0, shuffle=True)
        self.assertEqual(gold, ["c2"])
        self.assertEqual(
            {c["id"]: c["text"] for c in cands},
            {"c1": "alpha", "c2": "beta", "c3": "gamma"})

    def test_dev_group_is_refused(self):
        bad = dict(episode(), variant_group="dev-12")
        with self.assertRaises(RuntimeError):
            PF.guard_training_cut([bad])


class LossTests(unittest.TestCase):
    def test_one_gold_matches_cross_entropy(self):
        logits = mx.array([[0.2, -0.4, 1.1]])
        mask = mx.array([[0.0, 0.0, 1.0]])
        got = PF.pointer_nll(logits, mask)
        probs = mx.softmax(logits, axis=-1)
        expect = -mx.log(probs[0, 2])
        mx.eval(got, expect)
        self.assertAlmostEqual(float(got.item()), float(expect.item()), places=5)

    def test_two_golds_sum_their_mass(self):
        logits = mx.array([[0.0, 1.0, 0.5]])
        mask = mx.array([[1.0, 0.0, 1.0]])
        got = PF.pointer_nll(logits, mask)
        probs = mx.softmax(logits, axis=-1)
        expect = -mx.log(probs[0, 0] + probs[0, 2])
        mx.eval(got, expect)
        self.assertAlmostEqual(float(got.item()), float(expect.item()), places=5)


class ReportTests(unittest.TestCase):
    def test_forced_is_hits_over_rows(self):
        episodes = [
            {"family": "priority_decision", "candidates": [{}, {}]},
            {"family": "priority_decision", "candidates": [{}, {}, {}]},
            {"family": "textual_inference_negation", "candidates": [{}, {}]},
        ]
        report = PF.battery_report(episodes, [True, False, True])
        self.assertEqual(report["hits"], 2)
        self.assertAlmostEqual(report["forced"], 2 / 3)
        self.assertEqual(report["by_family"]["priority_decision"]["hits"], 1)
        self.assertEqual(report["by_family"]["priority_decision"]["n"], 2)


if __name__ == "__main__":
    unittest.main()
