import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import research_loop as R


class RelevanceTest(unittest.TestCase):

    def test_decision_reranker_scores_above_unrelated_item(self):
        useful, matched = R.relevance({
            "title": "Listwise decision reranker with calibrated abstention",
            "summary": "", "license": "Apache-2.0"})
        unrelated, _ = R.relevance({
            "title": "Image segmentation", "summary": "",
            "license": "unknown"})
        self.assertGreaterEqual(useful, 10)
        self.assertGreater(useful, unrelated)
        self.assertIn("listwise", matched)


class ScheduleTest(unittest.TestCase):

    def test_missing_run_is_due(self):
        self.assertTrue(R.due({}, "daily"))
        self.assertTrue(R.due({}, "audit"))
        self.assertTrue(R.due({}, "weekly"))

    def test_recent_run_is_not_due(self):
        state = {"daily": {
            "last_completed_utc": R.utcnow(),
            "next_due_utc": R.next_due(R.utcnow(), "daily")}}
        self.assertFalse(R.due(state, "daily"))


class ScanTest(unittest.TestCase):

    def test_partial_scan_keeps_evidence_and_errors(self):
        good = lambda _: [{
            "id": "source:1", "source": "test", "kind": "paper",
            "title": "Decision calibration", "summary": "",
            "url": "https://example.test/1", "published": None,
            "updated": R.utcnow(), "revision": "abc",
            "license": "MIT",
        }]

        def bad(_):
            raise TimeoutError("offline")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(R, "SOURCES", {"good": good, "bad": bad}), \
                    patch.object(R, "OUT", root), \
                    patch.object(R, "STATE", root / "state.json"), \
                    patch.object(R, "EVENTS", root / "events.jsonl"), \
                    patch.object(R, "FINDINGS", root / "findings.jsonl"), \
                    patch.object(R, "RUNS", root / "runs"), \
                    patch.object(R, "REPORTS", root / "reports"):
                run = R.run_scan("daily", limit=2)
                self.assertEqual(run["status"], "partial")
                self.assertEqual(run["new_findings"], 1)
                self.assertEqual(run["sources"]["bad"]["status"], "error")
                retry = R.parse_utc(run["next_due_utc"])
                self.assertLessEqual(
                    (retry - R.datetime.now(R.timezone.utc)).total_seconds(),
                    3600)
                finding = json.loads(
                    (root / "findings.jsonl").read_text().splitlines()[0])
                self.assertEqual(finding["id"], "source:1")
                self.assertTrue(list((root / "reports").glob("*.md")))


if __name__ == "__main__":
    unittest.main()
