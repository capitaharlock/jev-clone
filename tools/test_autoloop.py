import random
import unittest

from tools import autoloop as AL


def summary(mean, low, **families):
    return {"mean": mean, "min": low, "by_family": families}


class PromotionTest(unittest.TestCase):

    def test_better_mean_and_floor_promotes(self):
        ok, reasons = AL.promotion(
            summary(0.945, 0.94, priority_decision=0.775),
            summary(0.955, 0.945, priority_decision=0.85))
        self.assertTrue(ok)
        self.assertEqual(reasons, [])

    def test_equal_mean_is_not_enough(self):
        ok, reasons = AL.promotion(summary(0.95, 0.94), summary(0.95, 0.95))
        self.assertFalse(ok)
        self.assertIn("mean", reasons[0])

    def test_family_regression_blocks_a_better_mean(self):
        ok, reasons = AL.promotion(
            summary(0.945, 0.94, extraction_paraphrase=1.0),
            summary(0.96, 0.95, extraction_paraphrase=0.975))
        self.assertFalse(ok)
        self.assertTrue(any("extraction_paraphrase" in r for r in reasons))

    def test_lower_worst_order_blocks(self):
        ok, _ = AL.promotion(summary(0.945, 0.94), summary(0.95, 0.935))
        self.assertFalse(ok)

    def test_missing_candidate_score_never_promotes(self):
        ok, _ = AL.promotion(summary(0.9, 0.9), {"mean": None})
        self.assertFalse(ok)


class SummarizeTest(unittest.TestCase):

    def test_mean_and_floor_cover_every_order(self):
        report = {
            "adapter_step": 7, "forced": 0.95,
            "orders": [{"forced": 0.95}, {"forced": 0.93}, {"forced": 0.97}],
            "by_family": {"priority_decision": {"forced": 0.8}},
        }
        got = AL.summarize(report)
        self.assertAlmostEqual(got["mean"], 0.95)
        self.assertEqual(got["min"], 0.93)
        self.assertEqual(got["by_family"], {"priority_decision": 0.8})


class WeakFamiliesTest(unittest.TestCase):

    def test_worst_first_then_one_strong_family_in_turn(self):
        scores = {f: 1.0 for f in AL.FAMILIES}
        scores["priority_decision"] = 0.775
        scores["textual_inference_negation"] = 0.9625
        got = AL.weak_families({"by_family": scores}, cycle=1)
        self.assertEqual(got[:2],
                         ["priority_decision", "textual_inference_negation"])
        self.assertEqual(len(got), 3)
        self.assertNotIn(got[2], got[:2])

    def test_no_scores_means_every_family(self):
        self.assertEqual(AL.weak_families({}, cycle=4), list(AL.FAMILIES))


class SubsampleTest(unittest.TestCase):

    def episode(self, k, answer="c7"):
        return {
            "id": "e1", "answer": answer, "acceptable_answers": None,
            "preference": None,
            "candidates": [{"id": f"c{i}", "text": f"t{i}"}
                           for i in range(1, k + 1)],
        }

    def test_gold_survives_and_order_is_kept(self):
        rng = random.Random(3)
        for _ in range(50):
            out = AL.subsample_options(self.episode(60), 3, rng)
            ids = [c["id"] for c in out["candidates"]]
            self.assertEqual(len(ids), 3)
            self.assertIn("c7", ids)
            self.assertEqual(ids, sorted(ids, key=lambda s: int(s[1:])))
            self.assertEqual(out["options_subsampled_from"], 60)

    def test_narrow_episode_is_untouched(self):
        ep = self.episode(5, answer="c2")
        self.assertIs(AL.subsample_options(ep, 8, random.Random(1)), ep)

    def test_preference_order_is_filtered(self):
        ep = self.episode(28)
        ep["answer"] = None
        ep["preference"] = {"order": [f"c{i}" for i in range(28, 0, -1)]}
        out = AL.subsample_options(ep, 2, random.Random(5))
        kept = {c["id"] for c in out["candidates"]}
        self.assertEqual(out["preference"]["order"][0], "c28")
        self.assertEqual(set(out["preference"]["order"]), kept)


class TornTailTest(unittest.TestCase):

    def test_half_written_last_line_is_dropped(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodes.jsonl"
            path.write_text('{"id": 1}\n{"id": 2}\n{"id": ', encoding="utf-8")
            AL.drop_torn_tail(path)
            self.assertEqual([r["id"] for r in AL.read_jsonl(path)], [1, 2])


class ModelsTest(unittest.TestCase):

    def test_override_is_per_role_and_ignores_unknown_keys(self):
        import tempfile
        from pathlib import Path
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            path.write_text('{"generate": "moe", "teacher": "x"}',
                            encoding="utf-8")
            with mock.patch.object(AL, "MODELS_FILE", path):
                got = AL.qwen_models()
        self.assertEqual(got, {"generate": "moe", "verify": AL.QWEN_MODEL})


class StrategyTest(unittest.TestCase):

    def test_empty_file_keeps_the_base_recipe(self):
        got = AL.load_strategy({})
        self.assertEqual(got["arm"], "base")
        self.assertEqual(got["steps"], AL.TRAIN_STEPS)
        self.assertEqual(got["shares"], AL.SHARES)

    def test_out_of_range_values_are_clamped(self):
        got = AL.load_strategy({"arm": "x", "steps": 5000, "lr": 1.0,
                                "warmup": 900, "rule_weak_weight": 50})
        self.assertEqual(got["steps"], 600)
        self.assertEqual(got["lr"], 1e-4)
        self.assertEqual(got["warmup"], 300)
        self.assertEqual(got["rule_weak_weight"], 8)

    def test_shares_are_normalised_and_unknown_families_dropped(self):
        got = AL.load_strategy({
            "shares": {"qwen": 2, "public": 1, "curriculum": 1, "rule": 1},
            "families": ["priority_decision", "nope"]})
        self.assertAlmostEqual(got["shares"]["qwen"], 0.4)
        self.assertEqual(got["families"], ["priority_decision"])

    def test_garbage_falls_back_to_base(self):
        self.assertEqual(AL.load_strategy({"steps": "many"})["arm"], "base")


class NoReplyTest(unittest.TestCase):

    def test_share_of_unanswered_rows(self):
        manifest = {"n_input": 800, "quarantine_reasons": {
            "verifier:no-reply": 620, "verifier:abstain": 19}}
        self.assertGreater(AL.no_reply_share(manifest), AL.NO_REPLY_MAX)
        self.assertEqual(AL.no_reply_share({}), 0.0)


if __name__ == "__main__":
    unittest.main()
