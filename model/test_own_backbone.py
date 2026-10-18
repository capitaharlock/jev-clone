import unittest

import torch

from model.own_backbone import ByteTokenizer, OwnDecisionModel, OwnModelConfig


def tiny_model() -> OwnDecisionModel:
    torch.manual_seed(7)
    return OwnDecisionModel(OwnModelConfig(
        max_length=64,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        pointer_layers=1,
        dropout=0.0,
    )).eval()


def inputs(model: OwnDecisionModel, options: list[str]):
    tok = model.tokenizer
    device = next(model.parameters()).device
    state = tok.batch(["the gate is open"], 64, device)
    question = tok.batch(["what is the gate state?"], 64, device)
    option_ids, option_tokens = tok.batch(options, 64, device)
    return (
        *state,
        *question,
        option_ids.unsqueeze(0),
        option_tokens.unsqueeze(0),
        torch.ones((1, len(options)), dtype=torch.bool, device=device),
    )


class ByteTokenizerTest(unittest.TestCase):

    def test_uses_fixed_utf8_vocabulary(self):
        tokenizer = ByteTokenizer()
        encoded = tokenizer.encode("á", 8)
        self.assertEqual(encoded[0], tokenizer.BOS)
        self.assertEqual(encoded[-1], tokenizer.EOS)
        self.assertTrue(all(0 <= token < tokenizer.VOCAB_SIZE
                            for token in encoded))


class OwnDecisionModelTest(unittest.TestCase):

    def test_lineage_has_no_external_checkpoint(self):
        model = tiny_model()
        self.assertFalse(model.lineage()["external_checkpoint"])
        self.assertEqual(model.lineage()["initialization"], "random")
        self.assertEqual(model.lineage()["tokenizer"], "utf-8-bytes")

    def test_dynamic_option_count(self):
        model = tiny_model()
        with torch.no_grad():
            two = model(*inputs(model, ["open", "closed"]))
            four = model(*inputs(model, ["open", "closed", "jammed",
                                         "unknown"]))
        self.assertEqual(tuple(two.shape), (1, 3))
        self.assertEqual(tuple(four.shape), (1, 5))

    def test_option_permutation_preserves_aligned_scores(self):
        model = tiny_model()
        options = ["open", "closed", "jammed"]
        order = [2, 0, 1]
        with torch.no_grad():
            base = model(*inputs(model, options))[0]
            shuffled = model(*inputs(model, [options[i] for i in order]))[0]
        restored = torch.empty_like(shuffled[:-1])
        for new_index, old_index in enumerate(order):
            restored[old_index] = shuffled[new_index]
        torch.testing.assert_close(base[:-1], restored, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(base[-1], shuffled[-1],
                                   atol=1e-5, rtol=1e-5)

    def test_default_model_is_first_capacity_rung(self):
        model = OwnDecisionModel()
        count = model.lineage()["parameters"]
        self.assertGreater(count, 12_000_000)
        self.assertLess(count, 25_000_000)


if __name__ == "__main__":
    unittest.main()
