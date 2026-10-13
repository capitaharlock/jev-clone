import unittest

from tools import benchmark_refine as R


def row():
    return {
        "id": "v2-test",
        "state": "Sensor A has value 12. Sensor B has value 40.",
        "question": "Which sensor has value 40?",
        "candidates": [
            {"id": "c1", "text": "Sensor A"},
            {"id": "c2", "text": "Sensor B"},
        ],
        "gold": {"candidate_ids": ["c2"]},
        "provenance": {
            "source_id": "benchmark-v2-program",
            "revision": "program-v1",
            "license": "CC0-1.0",
            "content_sha256": "0" * 64,
        },
    }


class RefineTest(unittest.TestCase):

    def test_rewrite_must_keep_numbers_and_mentioned_options(self):
        ok, _ = R.acceptable(
            row(), "The report gives Sensor A a value of 12 and Sensor B 40.")
        self.assertTrue(ok)
        ok, reason = R.acceptable(
            row(), "According to the rewritten report, Sensor B has the "
            "larger reading and should be selected.")
        self.assertFalse(ok)
        self.assertIn("missing:", reason)

    def test_publish_never_changes_gold(self):
        source = row()
        out = R.publish(
            source, "The report gives Sensor A a value of 12 and Sensor B 40.")
        self.assertEqual(out["gold"], source["gold"])
        self.assertTrue(out["refinement"]["gold_unchanged"])
        self.assertNotEqual(out["provenance"]["content_sha256"], "0" * 64)


if __name__ == "__main__":
    unittest.main()
