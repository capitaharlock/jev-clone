"""Tests for the #T-mix-5m scaling NO-GO (stdlib unittest).

One property carries all the weight: **the published gate cannot say GO
while the reasoning evals are at chance.** Everything else here exists to
keep that assertion from becoming vacuous — a rule that only ever sees a
NO-GO gate never proves it would have caught a GO one, so the tampering
tests build the green gate the evidence does not support and demand a
finding.

`artifacts/runs/` and `artifacts/mix-1m/curve.jsonl` are gitignored, so the
gate carries its own evidence and these tests run in CI without them. The
cross-check against the raw job log is skipped, never faked, when the log
is not on this machine.
"""
from __future__ import annotations

import copy
import json
import os
import unittest

from tools.mix_5m import nogo


def gate() -> dict:
    return nogo.load(nogo.GATE)


def green(doc: dict) -> dict:
    """The same gate, edited to claim the 5 M spend it did not earn."""
    doc = copy.deepcopy(doc)
    doc["pass"] = True
    doc["verdict"] = "GO"
    doc["decision"]["authorises"] = {
        k: True for k in doc["decision"]["authorises"]}
    return doc


def above_chance(margin: float = 0.05) -> dict:
    """A metric that clears its chance rate by CI95 lower bound."""
    return {"n": 1500, "accuracy": 0.25 + margin,
            "accuracy_ci95": [0.25 + margin / 2, 0.25 + margin * 2],
            "chance": 0.2,
            "accuracy_options_only": 0.25 + margin,
            "accuracy_options_only_ci95": [0.25 + margin / 2, 0.25 + margin * 2],
            "chance_options_only": 0.25, "ece": 0.03,
            "abstain_rate": 0.0, "mean_k": 4.0}


class TestChanceTest(unittest.TestCase):
    """`beats_chance` is eval/unseen.py's test, not a looser one."""

    def test_point_estimate_over_chance_is_not_enough(self):
        m = above_chance()
        m["accuracy_options_only_ci95"] = [0.24, 0.31]
        self.assertFalse(nogo.beats_chance(m))

    def test_ci_low_above_chance_clears(self):
        self.assertTrue(nogo.beats_chance(above_chance()))

    def test_ci_low_equal_to_chance_does_not_clear(self):
        m = above_chance()
        m["accuracy_options_only_ci95"] = [0.25, 0.31]
        self.assertFalse(nogo.beats_chance(m))


class TestGreenClaim(unittest.TestCase):
    def test_no_go_is_not_green_despite_the_substring(self):
        self.assertFalse(nogo.is_green({"verdict": "NO-GO"}))

    def test_top_level_pass_outranks_the_verdict(self):
        self.assertFalse(nogo.is_green({"pass": False, "verdict": "GO"}))

    def test_go_without_a_pass_key_is_green(self):
        self.assertTrue(nogo.is_green({"verdict": "GO"}))


class TestPublishedGate(unittest.TestCase):
    """What `artifacts/gates/T-mix-5m/gate.json` actually says."""

    def setUp(self):
        self.gate = gate()

    def test_is_a_no_go(self):
        self.assertIs(self.gate["pass"], False)
        self.assertEqual(self.gate["verdict"], "NO-GO")
        self.assertFalse(nogo.is_green(self.gate))

    def test_authorises_nothing(self):
        self.assertEqual(
            {k: False for k in self.gate["decision"]["authorises"]},
            self.gate["decision"]["authorises"])

    def test_is_internally_consistent(self):
        self.assertEqual([], nogo.violations(self.gate))

    def test_every_reasoning_cut_is_at_chance(self):
        """The finding itself: nine cuts, none of them clears chance."""
        rows = nogo._reasoning_rows(self.gate)
        self.assertEqual(9, len(rows))
        self.assertEqual(len(rows), len(nogo.at_chance(self.gate)))

    def test_both_backbones_got_worse_from_250k_to_1m(self):
        deltas = self.gate["evidence"]["scaling_250k_to_1m"]
        self.assertEqual(2, len(deltas))
        for run, d in deltas.items():
            self.assertLess(d["delta"], 0.0, run)
            self.assertLess(d["at_1m"], d["chance"], run)

    def test_the_frozen_backbone_is_quoted_from_the_runs(self):
        surface = self.gate["evidence"]["trainable_surface"]
        self.assertTrue(surface)
        for run, c in surface.items():
            self.assertIs(c["backbone_frozen"], True, run)
            self.assertLess(c["trainable_fraction"], 0.05, run)


class TestTheRuleBites(unittest.TestCase):
    """A green gate on this evidence must be rejected, loudly."""

    def setUp(self):
        self.gate = gate()

    def test_go_while_every_reasoning_eval_is_at_chance(self):
        rules = {v["rule"] for v in nogo.violations(green(self.gate))}
        self.assertIn("S1", rules)

    def test_go_while_the_scaling_deltas_are_negative(self):
        rules = {v["rule"] for v in nogo.violations(green(self.gate))}
        self.assertIn("S2", rules)

    def test_one_eval_above_chance_is_still_not_a_positive_curve(self):
        """S1 clears, S2 does not: the curve is the other half of §136."""
        doc = green(self.gate)
        ckpt = sorted(doc["evidence"]["reasoning_evals"])[0]
        doc["evidence"]["reasoning_evals"][ckpt]["reclor"] = above_chance()
        rules = {v["rule"] for v in nogo.violations(doc)}
        self.assertNotIn("S1", rules)
        self.assertIn("S2", rules)

    def test_a_go_needs_both_halves(self):
        doc = green(self.gate)
        ckpt = sorted(doc["evidence"]["reasoning_evals"])[0]
        doc["evidence"]["reasoning_evals"][ckpt]["reclor"] = above_chance()
        run = sorted(doc["evidence"]["scaling_250k_to_1m"])[0]
        d = doc["evidence"]["scaling_250k_to_1m"][run]
        d["at_1m"] = round(d["at_250k"] + 0.05, 6)
        d["delta"] = round(d["at_1m"] - d["at_250k"], 6)
        self.assertEqual([], nogo.violations(doc))

    def test_a_delta_that_is_not_the_difference_is_caught(self):
        doc = copy.deepcopy(self.gate)
        run = sorted(doc["evidence"]["scaling_250k_to_1m"])[0]
        doc["evidence"]["scaling_250k_to_1m"][run]["delta"] = 0.4
        rules = {v["rule"] for v in nogo.violations(doc)}
        self.assertIn("S3", rules)

    def test_a_no_go_may_not_authorise_work(self):
        doc = copy.deepcopy(self.gate)
        doc["decision"]["authorises"]["mix_5m"] = True
        rules = {v["rule"] for v in nogo.violations(doc)}
        self.assertIn("S4", rules)

    def test_a_go_that_authorises_nothing_is_not_a_go(self):
        doc = green(self.gate)
        doc["decision"]["authorises"]["mix_5m"] = False
        rules = {v["rule"] for v in nogo.violations(doc)}
        self.assertIn("S5", rules)


@unittest.skipUnless(os.path.exists(nogo.CURVE),
                     f"{nogo.CURVE} is gitignored — job log not on this machine")
class TestAgainstTheJobLog(unittest.TestCase):
    """The gate against `artifacts/mix-1m/curve.jsonl`, the raw record."""

    def setUp(self):
        self.gate = gate()

    def test_gate_agrees_with_the_curve(self):
        self.assertEqual([], nogo.curve_disagreements(self.gate))

    def test_every_cited_run_exited_zero_in_the_log(self):
        with open(nogo.CURVE, encoding="utf-8") as fh:
            ends = {json.loads(ln)["run_id"]: json.loads(ln)
                    for ln in fh if ln.strip()
                    and json.loads(ln).get("t") == "run-end"}
        for run in self.gate["evidence"]["in_corpus_rescore"]:
            self.assertIn(run, ends)
            self.assertEqual(0, ends[run]["rc"], run)

    def test_a_rewritten_accuracy_is_caught(self):
        doc = copy.deepcopy(self.gate)
        run = sorted(doc["evidence"]["in_corpus_rescore"])[0]
        doc["evidence"]["in_corpus_rescore"][run]["swag"]["accuracy"] = 0.87
        self.assertTrue(nogo.curve_disagreements(doc))

    def test_swag_is_at_chance_on_rows_the_model_trained_on(self):
        """K=4 by construction, so options-only chance is 0.25."""
        for run, datasets in self.gate["evidence"]["in_corpus_rescore"].items():
            swag = datasets["swag"]
            self.assertLess(swag["accuracy_options_only"], 0.27, run)
            self.assertGreater(swag["n"], 50000, run)


if __name__ == "__main__":
    unittest.main()
