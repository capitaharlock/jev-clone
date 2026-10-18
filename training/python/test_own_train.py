import json
import tempfile
import unittest
from pathlib import Path

import torch

from model.own_backbone import OwnDecisionModel, OwnModelConfig
from training.python.own_train import read_episodes, row_tensors


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


if __name__ == "__main__":
    unittest.main()
