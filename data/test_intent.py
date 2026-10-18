"""Tests for #T-teacher-intent (stdlib unittest)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from data.intent.email_triage import FORWARD_CASES, generate, to_jsonl
from data.schema import validate
from data.teacher_client import TeacherClient, agreement_report


class TestEmailTriage(unittest.TestCase):
    def test_generate_count_and_valid(self):
        rows = generate(500, seed=0)
        self.assertEqual(len(rows), 500)
        for r in rows:
            self.assertEqual(validate(r), [])
            q = r.questions[0]
            self.assertEqual(len(q.options), 4)
            self.assertIn(r.questions[0].answer, ("archivar", "responder", "urgente", "spam"))

    def test_deterministic(self):
        a = generate(200, seed=1)
        b = generate(200, seed=1)
        self.assertEqual([r.state for r in a], [r.state for r in b])
        self.assertEqual([r.questions[0].answer for r in a],
                         [r.questions[0].answer for r in b])
        c = generate(200, seed=2)
        self.assertNotEqual([r.state for r in a], [r.state for r in c])

    def test_splits_present(self):
        rows = generate(1000, seed=0)
        splits = {r.split for r in rows}
        self.assertEqual(splits, {"train", "calibration", "test"})

    def test_weights_match_gold(self):
        for r in generate(200, seed=3):
            q = r.questions[0]
            self.assertIsNotNone(q.weights)
            self.assertEqual(max(q.weights, key=q.weights.get), q.answer)
            self.assertTrue(0.0 <= q.teacher_conf <= 1.0)

    def test_to_jsonl_roundtrip(self):
        rows = generate(50, seed=0)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "et.jsonl"
            to_jsonl(rows, p)
            self.assertEqual(len(p.read_text().strip().split("\n")), 50)

    def test_forward_cases_shape(self):
        self.assertGreaterEqual(len(FORWARD_CASES), 10)
        self.assertTrue(all(isinstance(c, str) and c.strip() for c in FORWARD_CASES))


class TestTeacherClient(unittest.TestCase):
    def test_cache_second_pass_costs_zero(self):
        with tempfile.TemporaryDirectory() as d:
            cli = TeacherClient(cache_path=Path(d) / "labels.json")
            rows = generate(500, seed=7)
            texts = [r.state for r in rows]
            golds = [r.questions[0].answer for r in rows]
            cli.label(texts, gold_hint=golds)
            self.assertEqual(cli.calls_made + cli.cache_hits, 500)
            self.assertGreater(cli.calls_made, 0)  # dup states may hit cache early
            cli2 = TeacherClient(cache_path=Path(d) / "labels.json")
            cli2.label(texts, gold_hint=golds)
            self.assertEqual(cli2.calls_made, 0)
            self.assertEqual(cli2.cache_hits, 500)
            self.assertEqual(cli2.stats()["cost_eur"], 0.0)

    def test_agreement_report_keys(self):
        rep = agreement_report(["a", "b", "a"], ["a", "a", "a"])
        self.assertEqual(rep, {"n": 3, "agree": 2,
                               "agreement": round(2 / 3, 4)})


if __name__ == "__main__":
    unittest.main()
