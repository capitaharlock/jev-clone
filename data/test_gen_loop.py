"""Tests for tools/data_gen_loop.py — the budgeted loop CLI (#T-gen-schemas).

The loop used to append email templates forever; what is asserted here is
that a run ENDS when its quota is full, writes rows that validate as
universal-schema V1, and publishes its diversity instead of its rate.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.schema import Example, Option, Question, validate  # noqa: E402
from tools import data_gen_loop as gl  # noqa: E402


def run_loop(tmp, *extra):
    out = Path(tmp) / "schemas.jsonl"
    code = gl.main(["--per-cell", "2", "--seed", "31", "--quiet",
                    "--domains", "support-routing,logistics,devops-release",
                    "--languages", "es,en", "--ks", "3,6",
                    "--out", str(out), *extra])
    return code, out


def rows_of(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def as_example(row):
    return Example(
        state=row["state"], split=row["split"],
        questions=[Question(id=q["id"], kind=q["kind"],
                            options=[Option(**o) for o in q["options"]],
                            answer=q["answer"]) for q in row["questions"]])


class LoopTest(unittest.TestCase):
    def test_run_fills_its_quota_and_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out = run_loop(tmp)
            self.assertEqual(code, 0)
            rows = rows_of(out)
            self.assertEqual(len(rows), 3 * 2 * 2 * 2)  # dom x lang x K x quota

    def test_rows_are_valid_universal_schema_with_generator_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out = run_loop(tmp)
            for row in rows_of(out):
                self.assertEqual(validate(as_example(row)), [])
                gen = row["gen"]
                self.assertIn(gen["domain"],
                              ("support-routing", "logistics", "devops-release"))
                self.assertIn(gen["language"], ("es", "en"))
                self.assertIn(gen["k"], (3, 6))
                self.assertEqual(len(row["questions"][0]["options"]), gen["k"])
                self.assertIn(gen["question"], row["state"])
                self.assertEqual(gen["unknown"],
                                 row["questions"][0]["answer"] == "unknown")

    def test_diversity_is_published_not_a_row_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out = run_loop(tmp)
            report = json.loads((out.parent / "diversity.json").read_text())
            self.assertEqual(len(report["domains"]), 3)
            self.assertEqual(len(report["languages"]), 2)
            self.assertEqual(sorted(report["k_distribution"]), ["3", "6"])
            self.assertEqual(report["unique_skeletons"], report["rows"])
            self.assertIn("dedup", report)
            self.assertIn("rejections", report)
            self.assertGreater(len(report["skeleton_curve"]), 1)
            history = (out.parent / "diversity-history.jsonl").read_text()
            self.assertEqual(len(history.strip().splitlines()), 1)

    def test_splits_are_by_skeleton_never_by_row_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out = run_loop(tmp)
            rows = rows_of(out)
            by_skeleton = {}
            for r in rows:
                by_skeleton.setdefault(r["gen"]["skeleton"], set()).add(r["split"])
            self.assertTrue(all(len(v) == 1 for v in by_skeleton.values()))
            self.assertTrue(set(r["split"] for r in rows) <= {
                "train", "calibration", "test"})

    def test_unknown_domain_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                gl.main(["--domains", "email-triage", "--quiet",
                         "--out", str(Path(tmp) / "x.jsonl")])


if __name__ == "__main__":
    unittest.main()
