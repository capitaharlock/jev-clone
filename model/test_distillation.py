"""Tests for #T-distillation (stdlib unittest)."""
from __future__ import annotations

import unittest

from data.firewall import ContaminationScanner, canary_texts

from training.python.curriculum import build_stage_data

from .distillation import (
    PHASE_BUDGET_EUR_PER_1K,
    REQUIRED_RECORD_KEYS,
    adjudicate,
    cost_per_1k,
    evaluate_head,
    prompt_hash,
    run_distillation,
    teacher_prob,
    train_soft,
    validate_teacher_record,
)


class TestTeacherContract(unittest.TestCase):
    def test_record_complete(self):
        item = build_stage_data()["s1-multi"][0]
        p, rec = teacher_prob("qwen-local", item)
        self.assertTrue(0.0 < p < 1.0)
        self.assertEqual(validate_teacher_record(rec), [])
        self.assertTrue(all(k in rec for k in REQUIRED_RECORD_KEYS))

    def test_invalid_rejected(self):
        self.assertTrue(validate_teacher_record({}))
        _, rec = teacher_prob("qwen-local",
                              build_stage_data()["s1-multi"][0])
        bad = dict(rec, confidence=2.0)
        self.assertTrue(validate_teacher_record(bad))
        bad2 = dict(rec)
        del bad2["prompt_hash"]
        self.assertTrue(validate_teacher_record(bad2))

    def test_prompt_hash_stable(self):
        self.assertEqual(prompt_hash("abc"), prompt_hash("abc"))
        self.assertNotEqual(prompt_hash("abc"), prompt_hash("abd"))

    def test_contaminated_prompt_refused(self):
        from .distillation import TEACHERS  # noqa
        item = {"state": canary_texts()[0], "question": "q?",
                "options": [], "answer": "o0", "y": 0, "id": "cx"}
        with self.assertRaises(ValueError):
            teacher_prob("qwen-local", item)

    def test_no_eval_only_in_outputs(self):
        scanner = ContaminationScanner()
        data = build_stage_data()["s2-semantic"][:10]
        labeled, _ = adjudicate(data)
        for it in labeled:
            hit, _ = scanner.scan(it["state"] + " " + it["question"])
            self.assertFalse(hit)


class TestAdjudication(unittest.TestCase):
    def test_counts_add_up(self):
        data = build_stage_data()["s2-semantic"][:12]
        labeled, counts = adjudicate(data)
        self.assertEqual(len(labeled), len(data))
        self.assertEqual(counts["agree"] + counts["disagree"], len(data))
        self.assertEqual(counts["adjudicated"], counts["disagree"])

    def test_deterministic(self):
        data = build_stage_data()["s2-semantic"][:8]
        l1, c1 = adjudicate(data)
        l2, c2 = adjudicate(data)
        self.assertEqual(c1, c2)
        self.assertEqual([x["soft"] for x in l1],
                         [x["soft"] for x in l2])

    def test_cost_within_phase(self):
        data = build_stage_data()["s2-semantic"][:8]
        labeled, _ = adjudicate(data)
        records = [r for it in labeled for r in it["records"]]
        cost, note = cost_per_1k(records)
        lo, hi = PHASE_BUDGET_EUR_PER_1K
        if cost is not None:
            self.assertTrue(lo <= cost <= hi)
        else:
            self.assertIn("not_incurred", note)


class TestAblation(unittest.TestCase):
    def test_identical_splits_both_arms(self):
        rep = run_distillation()
        for arm in ("hard", "soft"):
            for k in ("nll", "brier", "ece", "mean_conf"):
                self.assertIn(k, rep[arm])
        self.assertIn(rep["decision"],
                      ("adopt-soft-KL",
                       "keep-hard-labels (no measured gain; roadmap unblocked)"))

    def test_zero_shot_reported(self):
        rep = run_distillation()
        self.assertIn("nll", rep["zero_shot_hard"])
        self.assertIn("nll", rep["zero_shot_soft"])

    def test_soft_trains(self):
        data = build_stage_data()["s2-semantic"][6:12]
        labeled, _ = adjudicate(data)
        head = train_soft(labeled, 6060, steps=10)
        ev = evaluate_head(head, data)
        self.assertGreater(ev["mean_conf"], 0.5)


if __name__ == "__main__":
    unittest.main()
