import json
import tempfile
import unittest
from pathlib import Path

import tools.own_loop as loop


class OwnLoopTest(unittest.TestCase):

    def test_rotation_follows_batch_order(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            for name in ("batch-0002", "batch-0001"):
                path = base / name
                path.mkdir()
                (path / "episodes.jsonl").write_text("{}\n")
            files = loop.episode_files(base)
        self.assertEqual([path.parent.name for path in files],
                         ["batch-0001", "batch-0002"])

    def test_promotion_requires_a_clean_gain(self):
        candidate = {"holdout_wilson_lower": 0.40, "holdout_chance": 0.30,
                     "checks": {"holdout_clean": True,
                                "unseen_beats_chance": True,
                                "own_weights": True}}
        self.assertTrue(loop.promotable(candidate, 0.05))
        self.assertFalse(loop.promotable(candidate, 0.20))
        candidate["checks"]["holdout_clean"] = False
        self.assertFalse(loop.promotable(candidate, 0.0))

    def test_state_write_replaces_the_file(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "state.json"
            loop.write_json(path, {"cycle": 1})
            self.assertEqual(json.loads(path.read_text())["cycle"], 1)
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
