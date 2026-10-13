import unittest

from tools import autoloop as AL
from tools import devloop as DL


def decision(arm, cand, before, promoted=False):
    row = {"candidate": {"mean": cand}, "champion_before": {"mean": before},
           "promoted": promoted}
    if arm:
        row["strategy"] = {"arm": arm}
    return row


class ArmResultsTest(unittest.TestCase):

    def test_rows_without_a_strategy_count_as_base(self):
        got = DL.arm_results([decision(None, 0.94, 0.95),
                              decision("gentle", 0.96, 0.95)])
        self.assertAlmostEqual(got["base"][0], -0.01)
        self.assertAlmostEqual(got["gentle"][0], 0.01)

    def test_failed_cycles_are_not_scored(self):
        got = DL.arm_results([{"cycle": 2, "failed_phase": "train"}])
        self.assertTrue(all(not v for v in got.values()))


class ChooseArmTest(unittest.TestCase):

    def test_untried_arms_go_first_in_order(self):
        results = {name: [] for name in DL.ARMS}
        results["base"] = [-0.01]
        arm, why = DL.choose_arm(results)
        self.assertEqual(arm, list(DL.ARMS)[1])
        self.assertIn("not measured", why)

    def test_best_measured_gain_wins_once_all_ran(self):
        results = {name: [-0.01, -0.012] for name in DL.ARMS}
        results["gentle"] = [0.004, 0.002]
        self.assertEqual(DL.choose_arm(results)[0], "gentle")

    def test_a_rarely_tried_arm_gets_another_turn(self):
        results = {name: [-0.001] * 20 for name in DL.ARMS}
        results["longer"] = [-0.002]
        self.assertEqual(DL.choose_arm(results)[0], "longer")


class PlateauTest(unittest.TestCase):

    def test_counts_scored_cycles_since_the_last_promotion(self):
        history = [decision(None, 0.96, 0.95, promoted=True),
                   decision(None, 0.94, 0.96), {"failed_phase": "train"},
                   decision("gentle", 0.95, 0.96)]
        self.assertEqual(DL.cycles_since_promotion(history), 2)


class ArmSpecTest(unittest.TestCase):

    def test_every_arm_survives_the_autoloop_clamp(self):
        for name in DL.ARMS:
            spec = DL.arm_strategy(name)
            loaded = AL.load_strategy(spec)
            self.assertEqual(loaded["arm"], name)
            for key in ("steps", "lr", "warmup", "rule_weak_weight"):
                self.assertEqual(loaded[key], spec[key], (name, key))


if __name__ == "__main__":
    unittest.main()
