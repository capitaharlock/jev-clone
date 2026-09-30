"""Tests de `data.stream`: numeracion de lotes y pausa, sin Ollama."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data import stream as S


class Batches(unittest.TestCase):
    def test_next_batch_is_one_when_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(S, "RULE_ROOT", Path(tmp)):
                self.assertEqual(S.list_rule_batches(), [])
                self.assertEqual(S.next_rule_batch(), 1)

    def test_next_batch_skips_holes_by_max(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for n in (1, 3):
                d = root / f"batch-{n:04d}"
                d.mkdir()
                (d / "episodes.jsonl").write_text("{}\n")
            with patch.object(S, "RULE_ROOT", root):
                self.assertEqual(S.list_rule_batches(), [1, 3])
                self.assertEqual(S.next_rule_batch(), 4)

    def test_training_flag_pauses(self):
        with tempfile.TemporaryDirectory() as tmp:
            flag = Path(tmp) / "TRAINING"
            with patch.object(S, "TRAINING", flag):
                self.assertFalse(S.training_paused())
                flag.write_text("1\n")
                self.assertTrue(S.training_paused())


class Heartbeat(unittest.TestCase):
    def test_write_heartbeat_is_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "producer.json"
            with patch.object(S, "HEARTBEAT", path):
                S.write_heartbeat({"action": "idle", "n": 0})
            doc = json.loads(path.read_text())
            self.assertEqual(doc["action"], "idle")


if __name__ == "__main__":
    unittest.main()
