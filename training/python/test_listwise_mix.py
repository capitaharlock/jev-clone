from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from data import episode_contract as EC
from training.python import listwise_mix as LM


class AlignedNumericMixTest(unittest.TestCase):
    def test_rotates_complete_groups_with_balanced_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch = root / "batch-0001"
            batch.mkdir()
            rows = []
            for family, form in LM.ALIGNED_FORMS.items():
                for lang in ("en", "es"):
                    for group_number in range(2):
                        group = f"{family}-{lang}-{group_number}"
                        for variant in range(4):
                            rows.append({
                                "id": f"{group}-{variant}",
                                "family": family,
                                "lang": lang,
                                "variant_group": group,
                                "candidates": [
                                    {"id": "c1", "text": "one"},
                                    {"id": "c2", "text": "two"},
                                ],
                                "rule_trace": {"form": form},
                            })
            (batch / "episodes.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows))

            first, first_meta = LM.aligned_numeric_rows(
                10, root, rows_per_cell={2: 4}, start_cycle=10)
            second, second_meta = LM.aligned_numeric_rows(
                11, root, rows_per_cell={2: 4}, start_cycle=10)

            self.assertEqual(len(first), 16)
            self.assertEqual(len(second), 16)
            self.assertEqual(
                first_meta["rows_per_family_lang_k"],
                {
                    f"{family}/{lang}/K2": 4
                    for family in (EC.COMPARISON, EC.PRIORITY)
                    for lang in ("en", "es")
                })
            self.assertTrue(
                {row["id"] for row in first}.isdisjoint(
                    row["id"] for row in second))
            self.assertEqual(first_meta["round"], 0)
            self.assertEqual(second_meta["round"], 1)


if __name__ == "__main__":
    unittest.main()
