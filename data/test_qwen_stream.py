"""Batch numbering for the Qwen producer. No server calls."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data import qwen_stream as Q


class NextBatch(unittest.TestCase):
    def test_starts_at_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(Q, "QWEN_ROOT", Path(tmp)):
                self.assertEqual(Q.next_batch(), (1, False))

    def test_resumes_an_unfinished_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            done = root / "batch-0003"
            done.mkdir()
            (done / "manifest.json").write_text(
                json.dumps({"status": "published"}))
            pending = root / "batch-0004"
            pending.mkdir()
            (pending / "manifest.json").write_text(
                json.dumps({"status": "planned"}))
            with patch.object(Q, "QWEN_ROOT", root):
                self.assertEqual(Q.next_batch(), (4, True))

    def test_advances_after_published_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for n in (1, 2):
                path = root / f"batch-{n:04d}"
                path.mkdir()
                (path / "manifest.json").write_text(
                    json.dumps({"status": "published"}))
            with patch.object(Q, "QWEN_ROOT", root):
                self.assertEqual(Q.next_batch(), (3, False))


if __name__ == "__main__":
    unittest.main()
