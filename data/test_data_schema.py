"""Tests for #T-data-schema (stdlib unittest — no deps yet)."""
from __future__ import annotations

import unittest

from .leakage import LeakageDetector
from .registry import DatasetCard, Registry
from .schema import Example, Option, Question, is_valid, validate
from .seed import EVAL_ONLY_BENCHMARKS, seed_registry


def _choice() -> Example:
    return Example(
        state="Order #42: shoes, size 44, trail use",
        questions=[
            Question(
                id="q1",
                kind="choice",
                options=[Option("a", "keep"), Option("b", "refund"), Option("c", "unknown")],
                answer="a",
            ),
            Question(
                id="q2",
                kind="boolean",
                options=[Option("yes", "yes"), Option("no", "no")],
                answer="unknown",
            ),
        ],
    )


class TestSchema(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(validate(_choice()), [])

    def test_rejects_v2_kind(self):
        ex = _choice()
        ex.questions[0].kind = "score"
        self.assertTrue(any("V2" in e for e in validate(ex)))

    def test_rejects_boolean_3_options(self):
        ex = _choice()
        ex.questions[1].options.append(Option("m", "maybe"))
        self.assertTrue(any("boolean" in e for e in validate(ex)))

    def test_rejects_bad_answer_and_conf(self):
        ex = _choice()
        ex.questions[0].answer = "zzz"
        ex.questions[0].teacher_conf = 2.0
        errs = validate(ex)
        self.assertEqual(len(errs), 2)

    def test_rejects_empty_state(self):
        ex = _choice()
        ex.state = "  "
        self.assertFalse(is_valid(ex))


class TestRegistry(unittest.TestCase):
    def test_seed_registers(self):
        reg = seed_registry()
        card = reg.get("banking77")
        self.assertEqual(card.license, "CC-BY-4.0")

    def test_rejects_unpinned(self):
        reg = Registry()
        with self.assertRaises(ValueError):
            reg.register(
                DatasetCard("x", "src", "mir", "MIT", "train", "", "0" * 64, "t")
            )

    def test_fence(self):
        reg = seed_registry()
        ok, _ = reg.train_ok("banking77")
        self.assertTrue(ok)
        for bad in ("clinc150", "huffpost", "boolq"):
            ok, _ = reg.train_ok(bad)
            self.assertFalse(ok, bad)
        self.assertTrue(reg.check_mix(["banking77", "clinc150"]))

    def test_manifest_hash(self):
        reg = seed_registry()
        m = reg.manifest(["banking77", "helpsteer2"])
        self.assertEqual(len(m["manifest_sha256"]), 64)


class TestLeakage(unittest.TestCase):
    def test_exact_and_paraphrase(self):
        det = LeakageDetector(
            ["What is the capital of France?"] + EVAL_ONLY_BENCHMARKS
        )
        hit, _ = det.scan("What is the capital of France?")
        self.assertTrue(hit)
        hit, _ = det.scan("what is the CAPITAL of france??")
        self.assertTrue(hit)
        hit, _ = det.scan("Classify this support ticket about shoes")
        self.assertFalse(hit)


if __name__ == "__main__":
    unittest.main()
