"""Tests de `training.python.loop`: escalera y promocion, sin torch."""
from __future__ import annotations

import unittest

from training.python import loop as L


def _eval(forced: float, joint: float, fam: dict, lang: dict) -> dict:
    return {
        "figures": {
            "forced": {"accuracy": forced},
            "counterfactual_joint": {"accuracy": joint},
            "macro_by_family": {
                "by_family": {k: {"accuracy": v} for k, v in fam.items()},
            },
            "by_lang": {k: {"accuracy": v} for k, v in lang.items()},
        },
        "rows": [],
    }


class Ladder(unittest.TestCase):
    def test_first_three_steps(self):
        self.assertEqual(L.budget_of(0), 2000)
        self.assertEqual(L.budget_of(1), 5000)
        self.assertEqual(L.budget_of(2), 10000)

    def test_beyond_table_doubles(self):
        self.assertEqual(L.budget_of(6), 200000)

    def test_auto_stays_on_minilm_before_three_failures(self):
        state = {"streak_no_promote": 2}
        got = L.select_weights(state, "auto", present=lambda _: True)
        self.assertEqual(got, L.MINILM_WEIGHTS)

    def test_auto_moves_to_verified_mdeberta_after_three_failures(self):
        state = {"streak_no_promote": 3}
        got = L.select_weights(state, "auto", present=lambda _: True)
        self.assertEqual(got, L.MDEBERTA_WEIGHTS)

    def test_auto_does_not_select_missing_weights(self):
        state = {"streak_no_promote": 300}
        got = L.select_weights(state, "auto", present=lambda _: False)
        self.assertEqual(got, L.MINILM_WEIGHTS)

    def test_larger_backbone_is_sticky(self):
        state = {
            "active_weights": L.MDEBERTA_WEIGHTS,
            "streak_no_promote": 0,
        }
        got = L.select_weights(state, "auto", present=lambda _: False)
        self.assertEqual(got, L.MDEBERTA_WEIGHTS)

    def test_switch_resets_checkpoint_and_starts_with_5k(self):
        state = {
            "active_weights": L.MINILM_WEIGHTS,
            "current": "/tmp/minilm",
            "step": 0,
            "streak_no_promote": 388,
        }
        self.assertTrue(L.activate_weights(state, L.MDEBERTA_WEIGHTS))
        self.assertIsNone(state["current"])
        self.assertIsNone(state["resume"])
        self.assertEqual(state["step"], 1)
        self.assertEqual(state["streak_no_promote"], 0)

    def test_resume_keeps_the_stronger_run(self):
        state = {}
        L.keep_resume(state, "/tmp/a", 0.64)
        L.keep_resume(state, "/tmp/b", 0.61)
        self.assertEqual(state["resume"], "/tmp/a")
        L.keep_resume(state, "/tmp/c", 0.70)
        self.assertEqual(state["resume"], "/tmp/c")
        self.assertEqual(state["resume_forced"], 0.70)

    def test_battery_layout_drops_a_checkpoint_from_the_other_shape(self):
        state = {"resume": "/tmp/old", "resume_forced": 0.63,
                 "current": "/tmp/promoted", "data_layout": "in_option"}
        self.assertTrue(L.adopt_battery_layout(state))
        self.assertIsNone(state["resume"])
        self.assertIsNone(state["current"])
        self.assertFalse(L.adopt_battery_layout(state))

    def test_mdeberta_stays_on_the_requested_device(self):
        self.assertEqual(
            L.select_device("mps", L.MDEBERTA_WEIGHTS),
            "mps",
        )
        self.assertEqual(
            L.select_device("mps", L.MINILM_WEIGHTS),
            "mps",
        )


class Promote(unittest.TestCase):
    def test_both_up_and_no_drop_promotes(self):
        c = _eval(0.62, 0.34, {"a": 0.4}, {"es": 0.6})
        t = _eval(0.70, 0.45, {"a": 0.5}, {"es": 0.61})
        v = L.promote_rule(c, t)
        self.assertTrue(v["promote"])

    def test_family_drop_blocks(self):
        c = _eval(0.62, 0.34, {"a": 0.5}, {"es": 0.6})
        t = _eval(0.70, 0.45, {"a": 0.4}, {"es": 0.61})
        v = L.promote_rule(c, t)
        self.assertFalse(v["promote"])
        self.assertIn("family:a", v["drops"])

    def test_forced_flat_blocks(self):
        c = _eval(0.62, 0.34, {"a": 0.4}, {"es": 0.6})
        t = _eval(0.62, 0.45, {"a": 0.5}, {"es": 0.61})
        self.assertFalse(L.promote_rule(c, t)["promote"])

    def test_signed_gate_is_the_authority(self):
        gate = {
            "comparison": {
                "forced": {"point": 0.08},
                "counterfactual_joint": {"point": 0.11},
                "drops": ["family:a"],
            },
            "pass": False,
            "verdict": "NO-GO",
            "reason": "family a fell",
        }
        verdict = L.verdict_from_gate(gate)
        self.assertFalse(verdict["promote"])
        self.assertEqual(verdict["gate_verdict"], "NO-GO")


if __name__ == "__main__":
    unittest.main()
