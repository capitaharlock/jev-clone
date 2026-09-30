"""Contract tests for the one-sequence option-marker scorer."""

import unittest

from model import ce_scorer as CE
from model import listwise_scorer as LS


class TinyTokenizer:
    cls_token_id = 1
    sep_token_id = 2
    mask_token_id = 3
    pad_token_id = 0
    mask_token = "[MASK]"

    def encode(self, text, add_special_tokens=False):
        return [10 + (ord(char) % 41) for char in text if not char.isspace()]


class EncodingTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TinyTokenizer()

    def test_one_marker_per_option(self):
        got = LS.encode_texts(
            self.tokenizer, "state and question", ["beta", "alpha", "gamma"])
        self.assertEqual(len(got.marker_positions), 3)
        self.assertEqual(
            [got.input_ids[pos] for pos in got.marker_positions],
            [self.tokenizer.mask_token_id] * 3)

    def test_canonical_order_maps_back_to_input(self):
        got = LS.encode_texts(
            self.tokenizer, "prompt", ["beta", "alpha", "gamma"],
            canonical_order=True)
        self.assertEqual(got.encoded_to_input, [1, 0, 2])

    def test_sequence_too_long_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "complete options need"):
            LS.encode_texts(
                self.tokenizer, "x" * 30, ["a" * 20, "b" * 20],
                max_length=24)

    def test_state_is_cut_before_any_option(self):
        got = LS.encode_texts(
            self.tokenizer, "question", ["alpha", "beta"],
            state="x" * 100, max_length=50)
        self.assertTrue(got.truncated)
        self.assertEqual(len(got.marker_positions), 2)
        self.assertEqual(
            [got.input_ids[pos] for pos in got.marker_positions],
            [self.tokenizer.mask_token_id] * 2)

    def test_decision_uses_the_shared_render_contract(self):
        decision = CE.Decision(
            state="Two offers are available.",
            question="Which one is cheaper?",
            candidates=(
                CE.Candidate("c1", "Alpha costs 20 euros"),
                CE.Candidate("c2", "Beta costs 10 euros"),
            ),
            family=CE.COMPARISON,
            gold="c2",
        )
        got = LS.encode_decision(self.tokenizer, decision)
        self.assertEqual(len(got.marker_positions), 2)
        self.assertLess(max(got.marker_positions), len(got.input_ids))

    def test_flat_pair_adapter_keeps_decision_boundaries(self):
        groups = LS.ListwisePairScorer._groups([
            ("premise one", "option a"),
            ("premise one", "option b"),
            ("premise two", "option c"),
            ("premise two", "option d"),
        ])
        self.assertEqual(groups, [
            ("premise one", ["option a", "option b"]),
            ("premise two", ["option c", "option d"]),
        ])

    def test_pair_format_is_recovered_without_the_context_block(self):
        state, question, options = LS.ListwisePairScorer._parts(
            "State: facts\nQuestion: Which?\nOptions under comparison:\n- A\n- B",
            ["The answer to this question is: A.",
             "The answer to this question is: B."])
        self.assertEqual(state, "facts")
        self.assertEqual(question, "Which?")
        self.assertEqual(options, ["A", "B"])


if __name__ == "__main__":
    unittest.main()
