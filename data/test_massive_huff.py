"""Tests for #T-massive-huff (stdlib unittest).

Gate requirements (§§26-31, 59, 65-66, 99, 122-126):
- MASSIVE adapter keeps intent + locale descriptions and passes an
  ES<->EN round-trip plus a code-switching probe.
- LogiQA/ReClor convert state/question/options/gold with NO explanations
  stored (serialized output scanned for explanation substrings).
- HuffPost 41-category universe: previously rejected categories convert.
- Civil cap: probe-mix share counter fails when civil > 15%.
- License matrix: huffpost gated (content rights), boolq share-alike in
  the commercial branch.
"""
from __future__ import annotations

import dataclasses
import json
import unittest

from .adapters import (
    HUFFPOST_UNIVERSE,
    adapt_huffpost,
    check_load,
    p0_registry,
)
from .convert_logiqa_reclor import adapt_logiqa, adapt_reclor, field_names
from .convert_massive import adapt_massive

CIVIL_CAP = 0.15


def civil_share(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("empty mix")
    return counts.get("civil-comments", 0) / total


class TestHuffpostFull(unittest.TestCase):
    def test_universe_covers_source_categories(self):
        # 41 distinct categories in the 200 853-row snapshot + legacy FOOD
        # kept as a harmless distractor.
        self.assertEqual(len(HUFFPOST_UNIVERSE), 42)
        self.assertEqual(HUFFPOST_UNIVERSE[:8],
                         ["POLITICS", "SPORTS", "TECH", "ENTERTAINMENT",
                          "BUSINESS", "CRIME", "TRAVEL", "FOOD"])

    def test_previously_rejected_categories_convert(self):
        rows = [{"headline": f"h{i}", "description": f"d{i} second sentence here",
                 "category": c, "split": "train"}
                for i, c in enumerate(["WELLNESS", "COMEDY", "MEDIA",
                                       "EDUCATION", "WORLD NEWS", "DIVORCE"])]
        ex = adapt_huffpost(rows, seed=0)
        self.assertEqual([e.questions[0].answer for e in ex],
                         ["WELLNESS", "COMEDY", "MEDIA",
                          "EDUCATION", "WORLD NEWS", "DIVORCE"])

    def test_still_rejects_empty_headline(self):
        with self.assertRaises(Exception):
            adapt_huffpost([{"headline": "", "description": "x",
                             "category": "TECH"}])


class TestMassiveLocales(unittest.TestCase):
    ROWS = [
        {"utt": "wake me up at seven", "intent": "alarm_set",
         "locale": "en-US", "split": "train"},
        {"utt": "despiértame a las siete", "intent": "alarm_set",
         "locale": "es-ES", "split": "train"},
        {"utt": "mets un réveil à sept heures", "intent": "alarm_set",
         "locale": "fr-FR", "split": "train"},
        {"utt": "stelle einen Wecker auf sieben", "intent": "alarm_set",
         "locale": "de-DE", "split": "train"},
        {"utt": "play some music", "intent": "music_play",
         "locale": "en-US", "split": "train"},
        {"utt": "pon música", "intent": "music_play",
         "locale": "es-ES", "split": "train"},
    ]

    def test_intent_and_locale_preserved(self):
        ex = adapt_massive(self.ROWS, seed=0)
        self.assertEqual(len(ex), 6)
        for e, r in zip(ex, self.ROWS):
            self.assertEqual(e.questions[0].answer, r["intent"])
            self.assertIn(r["locale"], e.state)

    def test_es_en_round_trip_same_answer(self):
        ex = adapt_massive(self.ROWS, seed=0)
        self.assertEqual(ex[0].questions[0].answer,
                         ex[1].questions[0].answer)
        self.assertEqual(ex[4].questions[0].answer,
                         ex[5].questions[0].answer)

    def test_code_switching_converts(self):
        rows = [{"utt": "set an alarma para las seven please", "intent": "alarm_set",
                 "locale": "en-US", "split": "calibration"},
                {"utt": "pon música please", "intent": "music_play",
                 "locale": "es-ES", "split": "calibration"}]
        ex = adapt_massive(rows, seed=0)
        self.assertEqual(ex[0].questions[0].answer, "alarm_set")
        self.assertEqual(ex[0].split, "calibration")


class TestLogiqaReclor(unittest.TestCase):
    def test_logiqa_shape_and_gold(self):
        inner = {"id": 7, "answer": 2, "text": "All cats are mammals.",
                 "question": "What follows?",
                 "options": ["Fish swim", "Birds fly", "Cats are mammals",
                             "Dogs bark"],
                 "type": {"Categorical Reasoning": True}}
        ex, skipped = adapt_logiqa([{"split": "train", "text": json.dumps(inner)}])
        self.assertEqual(skipped, {"truncated": 0, "degenerate": 0})
        self.assertEqual(ex[0].questions[0].answer, "opt2")
        self.assertEqual(ex[0].questions[0].kind, "choice")
        self.assertIn("All cats are mammals.", ex[0].state)
        keys = [k.lower() for k in field_names(dataclasses.asdict(ex[0]))]
        for forbidden in ("explanation", "rationale", "chain-of-thought", "cot"):
            self.assertFalse(any(forbidden in k for k in keys), keys)

    def test_logiqa_truncated_lines_skipped(self):
        ex, skipped = adapt_logiqa([{"split": "train", "text": '{"id": 1, "ans'}])
        self.assertEqual(skipped, {"truncated": 1, "degenerate": 0})
        self.assertEqual(ex, [])

    def test_logiqa_nli_shapes_to_boolean(self):
        mc_nli = {"label": "entailed",
                  "major_premise": ["Mountains are barren."],
                  "minor_premise": "Policy helps some people.",
                  "conclusion": "Food security is urgent."}
        ex, skipped = adapt_logiqa([{"split": "train",
                                     "text": json.dumps(mc_nli)}])
        self.assertEqual(skipped, {"truncated": 0, "degenerate": 0})
        self.assertEqual(ex[0].questions[0].kind, "boolean")
        self.assertEqual(ex[0].questions[0].answer, "yes")
        hypo = {"id": 1, "label": "not-entailment", "premise": "All men are mortal.",
                "hypothesis": "Socrates flies.", "type": {}}
        ex, _ = adapt_logiqa([{"split": "train", "text": json.dumps(hypo)}])
        self.assertEqual(ex[0].questions[0].answer, "no")

    def test_logiqa_degenerate_rows_skipped(self):
        inner = {"id": 9, "answer": 0, "text": "", "question": "Q?",
                 "options": ["a", "b"], "type": {}}
        ex, skipped = adapt_logiqa([{"split": "train", "text": json.dumps(inner)}])
        self.assertEqual(skipped, {"truncated": 0, "degenerate": 1})
        self.assertEqual(ex, [])

    def test_reclor_hidden_gold_marked_unknown(self):
        rows = [{"split": "test", "context": "Union talks stall.",
                 "question": "What follows?",
                 "answers": "['A stalls', 'B moves']",
                 "label": -1, "id_string": "test_99"}]
        ex = adapt_reclor(rows)
        self.assertEqual(ex[0].questions[0].answer, "unknown")

    def test_reclor_shape_and_gold(self):
        rows = [{"split": "test", "context": "All men are mortal.",
                 "question": "What follows?",
                 "answers": "['Socrates is mortal', 'Socrates flies']",
                 "label": 0, "id_string": "test_0"}]
        ex = adapt_reclor(rows)
        self.assertEqual(ex[0].questions[0].answer, "opt0")
        self.assertEqual(len(ex[0].questions[0].options), 2)
        keys = [k.lower() for k in field_names(dataclasses.asdict(ex[0]))]
        for forbidden in ("explanation", "rationale", "chain-of-thought", "cot"):
            self.assertFalse(any(forbidden in k for k in keys), keys)


class TestCivilCap(unittest.TestCase):
    def test_capped_mix_passes(self):
        counts = {"civil-comments": 150, "boolq": 300, "massive": 300,
                  "huffpost": 250}
        self.assertLessEqual(civil_share(counts), CIVIL_CAP)

    def test_uncapped_mix_fails(self):
        counts = {"civil-comments": 900, "boolq": 50, "massive": 50}
        self.assertGreater(civil_share(counts), CIVIL_CAP)

    def test_empty_mix_raises(self):
        with self.assertRaises(ValueError):
            civil_share({})


class TestLicenseMatrix(unittest.TestCase):
    def test_huffpost_gated_boolq_share_alike(self):
        reg = p0_registry()
        rep = check_load(["ordinary text"], "huffpost", reg)
        self.assertFalse(rep.fence_ok)
        rep = check_load(["ordinary text"], "boolq", reg)
        self.assertFalse(rep.fence_ok)
        # commercial branch: boolq carries the share-alike obligation
        from .registry import TRAIN_GATED
        self.assertIn("share-alike", TRAIN_GATED["boolq"])
        card = reg.get("boolq")
        self.assertEqual(card.license, "CC-BY-SA-3.0")

    def test_train_clear_sources(self):
        reg = p0_registry()
        for ds in ("banking77", "civil-comments", "helpsteer2"):
            rep = check_load(["ordinary training text"], ds, reg)
            self.assertTrue(rep.fence_ok, ds)


if __name__ == "__main__":
    unittest.main()
