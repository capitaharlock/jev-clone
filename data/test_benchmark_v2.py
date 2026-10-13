import json
import tempfile
import unittest
from pathlib import Path

from data import benchmark_v2 as B


class BenchmarkV2AdapterTest(unittest.TestCase):

    def row(self, capability="quantitative_reasoning"):
        return {
            "id": "v2-test-1",
            "state": "Alpha has a total of 12. Beta has a total of 20.",
            "question": "Which option has a total of 20?",
            "candidates": [
                {"id": "c1", "text": "Alpha"},
                {"id": "c2", "text": "Beta"},
            ],
            "gold": {"candidate_ids": ["c2"], "method": "program",
                     "confidence": 1.0, "verifier": "test"},
            "capability": capability,
            "domain": "mathematics",
            "language": "en",
            "group_id": "v2-test-group",
            "provenance": {"revision": "benchmark-v2-program-v1"},
        }

    def test_converts_every_capability_to_a_supported_family(self):
        for capability, family in B.CAPABILITY_TO_FAMILY.items():
            episode = B.convert(self.row(capability))
            self.assertEqual(episode["family"], family)
            self.assertEqual(episode["capability"], capability)
            self.assertEqual(episode["domain"], "mathematics")

    def test_loader_limits_each_capability(self):
        rows = []
        for capability in B.CAPABILITY_TO_FAMILY:
            for index in range(3):
                row = self.row(capability)
                row["id"] = f"v2-{capability}-{index}"
                row["group_id"] = row["id"]
                rows.append(row)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dev.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            episodes = B.load_battery(path, limit_per_capability=2)
        self.assertEqual(len(episodes), 20)


if __name__ == "__main__":
    unittest.main()
