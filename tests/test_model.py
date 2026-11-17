import tempfile
import unittest
from pathlib import Path

import torch

from model.backbone import ByteTokenizer, DecisionModel, ModelConfig
from model.checkpoint import load_model, save
from model.inference import batch_tensors, predict

SMALL = ModelConfig(max_length=512, d_model=64, n_heads=4, n_layers=2,
                    d_ff=128, pointer_layers=1, dropout=0.0)
CPU = torch.device("cpu")


def row(options: list[str], question: str = "Which has the most sugar?",
        answer: str = "c0") -> dict:
    return {"state": "Comparing biscuits on the shelf.", "question": question,
            "candidates": [{"id": f"c{i}", "text": t}
                           for i, t in enumerate(options)],
            "answer": answer}


ROWS = [row(["sugars 3 g", "sugars 12.5 g", "sugars 7 g"]),
        row(["energy 120 kcal, fat 2 g", "energy 40 kcal, fat 9 g",
             "energy 410 kcal, fat 0.5 g, of which saturates 0.1 g"]),
        row(["a", "a much longer option text that pads the batch " * 3, "b"]),
        row(["x", "y", "z", "w"], answer="c3")]


class TokenizerTest(unittest.TestCase):

    def test_bytes_round_trip_with_markers(self):
        tokens = ByteTokenizer().encode("sal 1 g", 32)
        self.assertEqual(tokens[0], ByteTokenizer.BOS)
        self.assertEqual(tokens[-1], ByteTokenizer.EOS)
        self.assertEqual(bytes(tokens[1:-1]).decode(), "sal 1 g")

    def test_non_ascii_text_is_utf8(self):
        tokens = ByteTokenizer().encode("azúcar", 32)
        self.assertEqual(bytes(tokens[1:-1]).decode("utf-8"), "azúcar")


class ModelTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.model = DecisionModel(SMALL).eval()

    def test_published_shape_has_about_seventeen_million_parameters(self):
        count = DecisionModel(ModelConfig()).lineage()["parameters"]
        self.assertEqual(count, 16_959_746)

    def test_no_parameter_is_shaped_by_a_label_count(self):
        allowed = {1, SMALL.d_model, SMALL.d_ff, 3 * SMALL.d_model,
                   4 * SMALL.d_model, SMALL.vocab_size, SMALL.max_length}
        for name, parameter in self.model.named_parameters():
            self.assertTrue(set(parameter.shape) <= allowed, name)

    def test_logits_cover_options_plus_unknown(self):
        inputs, _ = batch_tensors(self.model, ROWS[:3], CPU)
        self.assertEqual(tuple(self.model(*inputs).shape), (3, 4))

    def test_a_row_scores_the_same_alone_and_in_a_batch(self):
        batched = predict(self.model, ROWS, CPU, batch_size=8)
        for alone, together in zip((predict(self.model, [r], CPU)[0]
                                    for r in ROWS), batched):
            for key, value in alone["probabilities"].items():
                self.assertAlmostEqual(value, together["probabilities"][key],
                                       places=5)

    def test_option_count_is_free(self):
        for k in (2, 5, 9):
            result = predict(self.model, [row([f"o {i}" for i in range(k)])],
                             CPU)[0]
            self.assertEqual(len(result["probabilities"]), k)
            self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0,
                                   places=4)

    def test_long_inputs_stay_inside_the_context(self):
        long = row(["energy 1 kcal " * 200] * 2 + ["fat 2 g " * 300],
                   question="q " * 400)
        long["state"] = "s " * 900
        inputs, _ = batch_tensors(self.model, [long], CPU)
        self.assertEqual(self.model(*inputs).shape[1], 4)

    def test_budgets_fit_the_context(self):
        for k in range(1, 27):
            question, state, option = self.model._budgets(1536, k)
            self.assertLessEqual(question + state + 1 + k * (option + 1), 1536)

    def test_checkpoint_round_trip_is_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            save(self.model, Path(folder))
            loaded = load_model(Path(folder))
            for (name, a), (_, b) in zip(self.model.state_dict().items(),
                                         loaded.state_dict().items()):
                self.assertTrue(torch.equal(a, b), name)

    def test_position_table_can_grow_on_load(self):
        with tempfile.TemporaryDirectory() as folder:
            save(self.model, Path(folder))
            grown = load_model(Path(folder), max_length=1024)
            self.assertEqual(grown.position_embed.weight.shape[0], 1024)
            self.assertTrue(torch.equal(
                grown.position_embed.weight[:512],
                self.model.position_embed.weight))


if __name__ == "__main__":
    unittest.main()
