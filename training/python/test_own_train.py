import json
import tempfile
import unittest
from pathlib import Path

import torch

from model.own_backbone import OwnDecisionModel, OwnModelConfig
from training.python.own_train import (
    batch_tensors, leaked_texts, read_episodes, row_tensors, split_unseen)


def episode(answer: str = "a") -> dict:
    return {
        "state": "the gate is open",
        "question": "what is the gate state?",
        "candidates": [
            {"id": "a", "text": "open"},
            {"id": "b", "text": "closed"},
        ],
        "answer": answer,
    }


class OwnTrainTest(unittest.TestCase):

    def test_reader_rejects_rows_without_valid_gold(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "rows.jsonl"
            rows = [episode(), episode("unknown"), episode("missing"),
                    {"state": "", "question": "q"}]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            got = read_episodes([path])
        self.assertEqual([row["answer"] for row in got], ["a", "unknown"])

    def test_row_tensors_maps_option_and_unknown_gold(self):
        model = OwnDecisionModel(OwnModelConfig(
            max_length=64, d_model=32, n_heads=4, n_layers=1, d_ff=64,
            pointer_layers=1, dropout=0.0))
        inputs, option_gold = row_tensors(
            model, episode(), 64, torch.device("cpu"))
        _, unknown_gold = row_tensors(
            model, episode("unknown"), 64, torch.device("cpu"))
        self.assertEqual(option_gold, 0)
        self.assertEqual(unknown_gold, 2)
        self.assertEqual(tuple(model(*inputs).shape), (1, 3))


def labelled(name: str, gold: str, *texts: str) -> dict:
    return {
        "state": f"state {name}",
        "question": f"question {name}",
        "candidates": [{"id": f"c{i}", "text": text}
                       for i, text in enumerate(texts, 1)],
        "answer": gold,
    }


class UnseenSplitTest(unittest.TestCase):

    def test_held_out_text_never_reaches_training(self):
        rows = [labelled(str(i), "c1", f"left {i}", f"right {i}")
                for i in range(8)]
        for seed in range(30):
            try:
                train, unseen, held = split_unseen(rows, 0.5, seed)
                break
            except ValueError:
                continue
        else:
            self.fail("no seed separated seen and unseen rows")
        self.assertGreater(held, 0)
        self.assertEqual(leaked_texts(train, unseen), [])
        train_text = {text for row in train
                      for candidate in row["candidates"]
                      for text in [candidate["text"]]}
        for row in unseen:
            self.assertTrue(all(candidate["text"] not in train_text
                                for candidate in row["candidates"]))

    def test_batch_accepts_equal_option_counts(self):
        model = OwnDecisionModel(OwnModelConfig(
            max_length=32, d_model=32, n_heads=4, n_layers=1, d_ff=64,
            pointer_layers=1, dropout=0.0))
        rows = [labelled("a", "c1", "alpha", "beta"),
                labelled("b", "c2", "gamma", "delta")]
        inputs, gold = batch_tensors(model, rows, 32, torch.device("cpu"))
        self.assertEqual(tuple(model(*inputs).shape), (2, 3))
        self.assertEqual(gold.tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
