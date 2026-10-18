"""The arena board names every registered cut and the source of each score."""
from __future__ import annotations

import unittest

from eval.arena_board import PUBLIC, build_board


class Board(unittest.TestCase):
    def test_covers_the_decision_battery_and_public_cuts(self):
        rows = build_board()
        ids = {row["id"] for row in rows}
        self.assertIn("battery-dev", ids)
        self.assertIn("battery-dev:priority_decision", ids)
        self.assertIn("battery-dev:k=8", ids)
        for key, _split, _title, _protocol in PUBLIC:
            self.assertIn(key, ids)

    def test_jev_cells_name_their_source(self):
        for row in build_board():
            self.assertIn(row["jev_kind"], {"measured", "cited", "absent"})
            self.assertIn(row["clone_kind"], {"measured", "unscored"})
            self.assertTrue(row["protocol"])
            if row["jev_kind"] == "absent":
                self.assertIsNone(row["jev"])
            if row["jev_kind"] == "cited":
                self.assertIsNotNone(row["jev"])


if __name__ == "__main__":
    unittest.main()
