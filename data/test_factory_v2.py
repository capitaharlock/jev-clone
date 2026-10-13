import json
import tempfile
import unittest
from pathlib import Path

from data import episode_contract as EC
from data import factory_v2 as F


class FactoryV2Test(unittest.TestCase):

    def test_source_release_converts_to_train_episode(self):
        row = {
            "id": "gsm8k-train-1",
            "state": "Three groups of four contain twelve items.",
            "question": "Which numerical result answers the problem?",
            "candidates": [
                {"id": "c1", "text": "7"},
                {"id": "c2", "text": "12"},
            ],
            "gold": {"candidate_ids": ["c2"]},
            "capability": "quantitative_reasoning",
            "domain": "mathematics",
            "language": "en",
            "group_id": "gsm8k-train-1",
            "provenance": {
                "source_id": "gsm8k",
                "revision": "pinned",
            },
        }
        episode = F.convert(row)
        self.assertTrue(EC.is_valid(episode))
        self.assertEqual(episode["family"], EC.COMPARISON)
        self.assertEqual(episode["origin"], "factory:gsm8k")
        self.assertEqual(episode["split"], "train")

    def test_missing_release_is_an_empty_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(F.load_training(Path(tmp) / "missing.jsonl"), [])


if __name__ == "__main__":
    unittest.main()
