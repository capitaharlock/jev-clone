"""Tests for the #T-gen-objective sweep: comparability, eligibility, honesty.

Stdlib unittest, no torch: everything here is the algebra that decides
which candidate is adopted and the guardrail that refuses a gate whose
numbers are not on disk. The candidates' own mechanics are tested in
`training/python/test_train_decision.py` (they need the real stack).

Run:
    python3 -m unittest tools.gen_objective.test_arms -v
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from . import arms


def stage(samples: int, acc: float, rank: float, n: int = 3000,
          chance: float = 0.165236, rank_chance: float = 0.201924,
          abstain: float = 0.5) -> dict:
    """A stage record shaped exactly like the trainer's, with real Wilson CIs."""
    lo, hi = arms_wilson(acc, n)
    rlo, rhi = arms_wilson(rank, n)
    return {
        "t": "stage", "stage": samples, "samples": samples, "tokens": 10 * samples,
        "checkpoint": f"artifacts/checkpoints/decision/x/stage-{samples:09d}",
        "model_version": "test-version", "elapsed_s": 1.0,
        "seen": {"accuracy": 0.3, "n": n},
        "unseen": {"n": n, "accuracy": acc, "accuracy_ci95": [lo, hi],
                   "chance": chance, "accuracy_options_only": rank,
                   "accuracy_options_only_ci95": [rlo, rhi],
                   "chance_options_only": rank_chance,
                   "abstain_rate": abstain, "ece": 0.4, "nll": 2.0,
                   "per_dataset": {}},
    }


def arms_wilson(p: float, n: int) -> list:
    from training.python.train_decision import wilson_interval
    lo, hi = wilson_interval(int(round(p * n)), n)
    return [round(lo, 6), round(hi, 6)]


class TestComparability(unittest.TestCase):
    """The arms must differ by their one flag and by nothing else."""

    @staticmethod
    def _shared(arm: dict) -> list:
        """The argv with the arm's own flags and its run id removed.

        What is left must be byte-identical across the five arms: that is
        the whole claim "same seed, same mix, same eval, same budget".
        """
        cmd = arms.train_command(arm)
        out, skip = [], 0
        for i, token in enumerate(cmd):
            if skip:
                skip -= 1
                continue
            if token == "--run-id":
                skip = 1
                continue
            if token in arm["flags"]:
                # a value flag consumes its value too
                if i + 1 < len(cmd) and cmd[i + 1] in arm["flags"]:
                    skip = 1
                continue
            out.append(token)
        return out

    def test_every_arm_shares_the_whole_command_but_its_flags(self):
        self.assertEqual(arms.ARMS[0]["flags"], [])
        base = self._shared(arms.ARMS[0])
        for arm in arms.ARMS[1:]:
            self.assertEqual(self._shared(arm), base,
                             f"{arm['id']} differs by more than its flags")
            for flag in arm["flags"]:
                self.assertIn(flag, arms.train_command(arm))

    def test_the_shared_axes_are_literally_shared(self):
        for arm in arms.ARMS:
            cmd = arms.train_command(arm)
            for expect in ("--seed", str(arms.SEED), "--mix-seed",
                           str(arms.CLEAN_1M_SEED), "--backbone",
                           arms.BACKBONE, "--max-samples", str(arms.BUDGET),
                           "--fence-clean", "--device", arms.DEVICE):
                self.assertIn(expect, cmd, f"{arm['id']} lost {expect}")

    def test_the_five_arms_cover_the_four_candidates_plus_a_control(self):
        self.assertEqual(sorted(a["candidate"] for a in arms.ARMS),
                         [0, 1, 2, 3, 4])
        self.assertEqual(len({a["id"] for a in arms.ARMS}), 5)

    def test_run_ids_are_distinct_per_arm(self):
        ids = [arms.run_id(a["id"]) for a in arms.ARMS]
        self.assertEqual(len(set(ids)), len(ids))


class TestEligibility(unittest.TestCase):
    """Both bounds, on the LOWER end of the CI, or the arm is discarded."""

    def test_a_point_estimate_above_chance_is_not_enough(self):
        # 0.18 point estimate over chance 0.1652, but n=3000 leaves the
        # lower bound at ~0.167 — clears; at n=200 it does not.
        wide = arms._margin(arms._point(stage(125000, 0.18, 0.30, n=60)))
        self.assertGreater(wide["accuracy_chance"], 0.0)
        self.assertFalse(wide["accuracy_beats_chance"])
        self.assertLess(wide["accuracy_margin"], 0.0)

    def test_both_criteria_must_clear(self):
        # accuracy clears, ranking does not -> not eligible
        m = arms._margin(arms._point(stage(125000, 0.40, 0.20)))
        self.assertTrue(m["accuracy_beats_chance"])
        self.assertFalse(m["ranking_beats_chance"])
        reason = arms._discard_reason(m)
        self.assertIn("ranking", reason)
        self.assertNotIn("unseen accuracy CI95", reason)

    def test_discard_reason_is_numeric_not_a_verdict_word(self):
        m = arms._margin(arms._point(stage(125000, 0.05, 0.19)))
        reason = arms._discard_reason(m)
        self.assertIn(str(m["accuracy_ci95_lower"]), reason)
        self.assertIn(str(m["accuracy_chance"]), reason)
        self.assertIn(str(m["ranking_ci95_lower"]), reason)
        for word in ("bad", "worse", "fails", "NO-GO"):
            self.assertNotIn(word, reason)

    def test_slope_needs_two_points(self):
        one = arms._slope([arms._point(stage(62500, 0.1, 0.2))])
        self.assertFalse(one["measured"])
        two = arms._slope([arms._point(stage(62500, 0.10, 0.24)),
                           arms._point(stage(125000, 0.06, 0.19))])
        self.assertTrue(two["measured"])
        self.assertAlmostEqual(two["unseen_accuracy"], -0.04, places=6)
        self.assertAlmostEqual(two["unseen_ranking"], -0.05, places=6)


class TestChoose(unittest.TestCase):
    def _report(self, acc, rank, complete=True, measured=True):
        point = arms._point(stage(125000, acc, rank))
        margin = arms._margin(point)
        r = {"measured": measured, "complete": complete, "points": [point],
             "final": point, "final_margin": margin,
             "eligible": margin["accuracy_beats_chance"]
             and margin["ranking_beats_chance"]}
        return r

    def test_none_eligible_adopts_nothing(self):
        reports = {"baseline": self._report(0.06, 0.19),
                   "prior": self._report(0.07, 0.20)}
        adopted, ranking = arms.choose(reports)
        self.assertIsNone(adopted)
        self.assertEqual(ranking, [])

    def test_highest_unseen_accuracy_among_eligible_wins(self):
        reports = {"baseline": self._report(0.30, 0.40),
                   "episodic": self._report(0.45, 0.35),
                   "prior": self._report(0.05, 0.19)}
        adopted, ranking = arms.choose(reports)
        self.assertEqual(adopted, "episodic")
        self.assertEqual([r["arm"] for r in ranking],
                         ["episodic", "baseline"])

    def test_an_incomplete_arm_cannot_be_adopted(self):
        reports = {"episodic": self._report(0.45, 0.35, complete=False)}
        self.assertIsNone(arms.choose(reports)[0])

    def test_an_unmeasured_arm_cannot_be_adopted(self):
        reports = {"episodic": self._report(0.45, 0.35, measured=False)}
        self.assertIsNone(arms.choose(reports)[0])


class TestGateHonesty(unittest.TestCase):
    """`audit_gate` must reject a number that is not on disk."""

    def _measured_arm(self, samples=125000, acc=0.30, rank=0.40):
        point = arms._point(stage(samples, acc, rank))
        margin = arms._margin(point)
        return {"measured": True, "complete": True, "run_id": "ghost-run",
                "points": [point], "final": point, "final_margin": margin,
                "eligible": True, "slope": {"measured": False}}

    def test_a_fabricated_measured_arm_is_rejected(self):
        gate = {"arms": {"episodic": self._measured_arm()}, "adopted": None}
        report = arms.audit_gate(gate, strict_disk=True)
        self.assertFalse(report["pass"])
        self.assertTrue(any("no stage record exists on disk" in v
                            for v in report["violations"]))

    def test_an_unmeasured_arm_carrying_numbers_is_rejected(self):
        arm = self._measured_arm()
        arm["measured"] = False
        arm["not_measured"] = "the run never started"
        gate = {"arms": {"episodic": arm}, "adopted": None}
        report = arms.audit_gate(gate, strict_disk=False)
        self.assertFalse(report["pass"])
        self.assertTrue(any("without a measurement behind it" in v
                            for v in report["violations"]))

    def test_an_unmeasured_arm_must_say_why(self):
        gate = {"arms": {"episodic": {"measured": False}}, "adopted": None}
        report = arms.audit_gate(gate, strict_disk=False)
        self.assertFalse(report["pass"])
        self.assertTrue(any("no `not_measured` reason" in v
                            for v in report["violations"]))

    def test_adopting_an_ineligible_arm_is_rejected(self):
        arm = self._measured_arm()
        arm["eligible"] = False
        gate = {"arms": {"episodic": arm}, "adopted": "episodic"}
        report = arms.audit_gate(gate, strict_disk=False)
        self.assertFalse(report["pass"])
        self.assertTrue(any("not a measured, complete, eligible arm" in v
                            for v in report["violations"]))

    def test_adopting_an_arm_that_does_not_exist_is_rejected(self):
        gate = {"arms": {"episodic": self._measured_arm()},
                "adopted": "contrastive"}
        report = arms.audit_gate(gate, strict_disk=False)
        self.assertFalse(report["pass"])
        self.assertTrue(any("is not one of the arms" in v
                            for v in report["violations"]))

    def test_a_gate_with_no_arms_is_rejected(self):
        self.assertFalse(arms.audit_gate({"arms": {}},
                                         strict_disk=False)["pass"])

    def test_a_point_whose_number_disagrees_with_disk_is_rejected(self):
        """The teeth: same run, same stage, a number quietly edited."""
        with tempfile.TemporaryDirectory() as tmp:
            run = os.path.join(tmp, arms.run_id("episodic"))
            os.makedirs(run)
            with open(os.path.join(run, "metrics.jsonl"), "w") as fh:
                fh.write(json.dumps(stage(125000, 0.30, 0.40)) + "\n")
            arm = self._measured_arm()
            arm["points"][0]["unseen_accuracy"] = 0.99   # the edit
            arm["final"]["unseen_accuracy"] = 0.99
            gate = {"arms": {"episodic": arm}, "adopted": None}
            original = arms.RUNS
            try:
                arms.RUNS = tmp
                report = arms.audit_gate(gate, strict_disk=True)
            finally:
                arms.RUNS = original
        self.assertFalse(report["pass"])
        self.assertTrue(any("gate says unseen_accuracy=0.99" in v
                            for v in report["violations"]))

    def test_a_faithful_gate_passes_the_same_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = os.path.join(tmp, arms.run_id("episodic"))
            os.makedirs(run)
            record = stage(125000, 0.30, 0.40)
            with open(os.path.join(run, "metrics.jsonl"), "w") as fh:
                fh.write(json.dumps(record) + "\n")
            gate = {"arms": {"episodic": self._measured_arm()},
                    "adopted": "episodic"}
            original = arms.RUNS
            try:
                arms.RUNS = tmp
                report = arms.audit_gate(gate, strict_disk=True)
            finally:
                arms.RUNS = original
        self.assertTrue(report["pass"], report["violations"])


class TestPublishedGate(unittest.TestCase):
    """The gate actually on disk must survive its own guardrail."""

    @unittest.skipUnless(os.path.exists(arms.GATE_PATH),
                         "the gate has not been written yet")
    def test_the_published_gate_passes_its_audit(self):
        with open(arms.GATE_PATH) as fh:
            gate = json.load(fh)
        # `artifacts/runs/` is gitignored: the factual half only runs where
        # the runs survive. The structural half always does.
        strict = os.path.isdir(arms.RUNS)
        report = arms.audit_gate(gate, strict_disk=strict)
        self.assertTrue(report["pass"], report["violations"])

    @unittest.skipUnless(os.path.exists(arms.GATE_PATH),
                         "the gate has not been written yet")
    def test_the_published_gate_declares_its_subset_and_rule(self):
        with open(arms.GATE_PATH) as fh:
            gate = json.load(fh)
        self.assertEqual(gate["task"], arms.TASK)
        self.assertTrue(gate["backbone_frozen"])
        declared = gate["declared_subset"]
        self.assertEqual(declared["budget_decisions"], arms.BUDGET)
        self.assertEqual(declared["seed"], arms.SEED)
        self.assertIn("why_not_1m", declared)
        self.assertIn("criterion_1", gate["adoption_rule"])
        self.assertIn("criterion_2", gate["adoption_rule"])
        self.assertEqual(sorted(gate["arms"]),
                         sorted(a["id"] for a in arms.ARMS))
        # a discarded arm keeps its number instead of being deleted
        for rid, report in gate["arms"].items():
            if report.get("measured") and not report.get("eligible"):
                self.assertIn("discard_reason", report)
                self.assertIsNotNone(report["final"]["unseen_accuracy"])


if __name__ == "__main__":
    unittest.main()
