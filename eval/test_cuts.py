"""Unit tests for eval.cuts — the dev/test separation (rule R7).

The cuts are sealed into a tempdir here, never into `artifacts/splits/`, so
the tests can prove the seal refuses to move without touching the seal the
repo published.
"""
import json
import tempfile
import unittest
from pathlib import Path

from . import cuts as K


class TestTheTwoCuts(unittest.TestCase):
    def test_the_dev_cut_is_not_the_reserved_one(self):
        self.assertFalse(K.DEV.reserved)
        self.assertTrue(K.TEST.reserved)
        self.assertNotEqual(K.DEV.split, K.TEST.split)

    def test_no_row_belongs_to_both(self):
        """The whole point: an arm chosen on dev was not chosen on test."""
        dev, test = set(K.cut_ids(K.DEV)), set(K.cut_ids(K.TEST))
        self.assertEqual(dev & test, set())
        self.assertEqual(len(dev), K.DEV_ROWS)
        self.assertEqual(len(test), 3080)

    def test_the_dev_cut_is_a_seeded_sample_not_the_first_n_rows(self):
        ids = K.cut_ids(K.DEV)
        head = K.split_ids(K.DEV)[:K.DEV_ROWS]
        self.assertNotEqual(set(ids), set(head))

    def test_the_same_seed_draws_the_same_rows(self):
        self.assertEqual(K.cut_ids(K.DEV), K.cut_ids(K.DEV))

    def test_another_seed_draws_other_rows(self):
        other = K.Cut(**{**K.DEV.__dict__, "seed": K.DEV_SEED + 1})
        self.assertNotEqual(set(K.cut_ids(K.DEV)), set(K.cut_ids(other)))


class TestSealing(unittest.TestCase):
    def test_sealing_writes_the_ids_and_a_manifest_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            man = K.seal(K.DEV, splits_dir=Path(tmp))
            base = Path(tmp) / K.DEV.split_name
            self.assertTrue((base / "banking77-dev.ids").exists())
            self.assertEqual(man["counts"]["rows"], K.DEV_ROWS)
            self.assertEqual(len(man["manifest_sha256"]), 64)
            self.assertEqual(man["reserved"], False)

    def test_a_second_seal_verifies_instead_of_rewriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = K.seal(K.DEV, splits_dir=Path(tmp))
            again = K.seal(K.DEV, splits_dir=Path(tmp))
            self.assertEqual(first["manifest_sha256"],
                             again["manifest_sha256"])

    def test_a_moved_fence_raises_instead_of_resealing_quietly(self):
        with tempfile.TemporaryDirectory() as tmp:
            K.seal(K.DEV, splits_dir=Path(tmp))
            path = Path(tmp) / K.DEV.split_name / "manifest.json"
            doc = json.loads(path.read_text())
            doc["files"]["banking77-dev.ids"]["sha256"] = "0" * 64
            path.write_text(json.dumps(doc))
            with self.assertRaises(ValueError) as cm:
                K.seal(K.DEV, splits_dir=Path(tmp))
            self.assertIn("disagrees", str(cm.exception))

    def test_force_reseals_on_purpose(self):
        with tempfile.TemporaryDirectory() as tmp:
            K.seal(K.DEV, splits_dir=Path(tmp))
            path = Path(tmp) / K.DEV.split_name / "manifest.json"
            doc = json.loads(path.read_text())
            doc["files"]["banking77-dev.ids"]["sha256"] = "0" * 64
            path.write_text(json.dumps(doc))
            man = K.seal(K.DEV, splits_dir=Path(tmp), force=True)
            self.assertEqual(man["counts"]["rows"], K.DEV_ROWS)

    def test_the_repo_seal_is_on_disk_and_still_agrees_with_the_corpus(self):
        """`python3 -m eval.cuts seal` was run; it must still verify."""
        for cut in K.CUTS.values():
            with self.subTest(cut=cut.name):
                self.assertTrue(cut.manifest_path.exists(),
                                "run `python3 -m eval.cuts seal`")
                K.seal(cut)  # raises if the rows moved


class TestSamplesMatchTheSeal(unittest.TestCase):
    def test_the_sealed_ids_name_the_samples_the_scorer_builds(self):
        """A seal that named other rows than the ones scored would be
        decoration. 40 rows is enough to prove the id scheme lines up."""
        rows = K.samples(K.DEV, limit=40)
        self.assertEqual(len(rows), 40)
        ids = set(K.sealed(K.DEV)["ids"])
        for s in rows:
            self.assertIn(f"{s.dataset}:{s.row_id}:{s.question_id}", ids)

    def test_every_row_carries_the_whole_label_space(self):
        rows = K.samples(K.DEV, limit=10)
        self.assertTrue(all(len(s.options) == 77 for s in rows))
        self.assertTrue(all(s.options[s.gold_index]["id"] == s.answer
                            for s in rows))


class TestTheSealSurvivesAFreshClone(unittest.TestCase):
    """Git carries the manifest, not the `.ids` list (`.gitignore`)."""

    def test_the_ids_are_recomputed_and_checked_against_the_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            K.seal(K.DEV, splits_dir=Path(tmp))
            (Path(tmp) / K.DEV.split_name / "banking77-dev.ids").unlink()
            doc = K.sealed(K.DEV, splits_dir=Path(tmp))
            self.assertEqual(doc["rows"], K.DEV_ROWS)
            self.assertEqual(set(doc["ids"]), set(K.cut_ids(K.DEV)))

    def test_a_corpus_that_moved_is_refused_not_rescored(self):
        with tempfile.TemporaryDirectory() as tmp:
            K.seal(K.DEV, splits_dir=Path(tmp))
            base = Path(tmp) / K.DEV.split_name
            (base / "banking77-dev.ids").unlink()
            doc = json.loads((base / "manifest.json").read_text())
            doc["files"]["banking77-dev.ids"]["sha256"] = "0" * 64
            (base / "manifest.json").write_text(json.dumps(doc))
            with self.assertRaises(ValueError):
                K.sealed(K.DEV, splits_dir=Path(tmp))


class TestLedger(unittest.TestCase):
    def test_a_query_is_appended_never_merged(self):
        doc = K.ledger()
        before = len(doc["queries"])
        grown = K.record_query("ckpt/x", "because", write=False)
        self.assertEqual(len(grown["queries"]), before + 1)
        self.assertEqual(grown["queries"][-1]["reason"], "because")
        self.assertEqual(len(K.ledger()["queries"]), before)

    def test_the_published_ledger_documents_the_queries_already_made(self):
        """R7 asks for the ones made BEFORE the ledger existed too."""
        doc = K.ledger()
        self.assertTrue(doc["queries"], "seed the ledger first")
        old = [q for q in doc["queries"] if "recorded" in q]
        self.assertTrue(old, "the pre-ledger queries are not written down")
        for q in doc["queries"]:
            with self.subTest(when=q["when"]):
                self.assertTrue(q["checkpoint"])
                self.assertTrue(q["reason"])
                self.assertTrue(q["when"])


if __name__ == "__main__":
    unittest.main()


class TestExternalCut(unittest.TestCase):
    """`external_cut` returns an episode-v1 cut BY SHA (#T-ingest-laya)."""

    def _publish(self, tmp: Path, split: str = "test", rows: int = 3,
                 eval_only: bool = True) -> Path:
        base = tmp / "typed-decisions" / split
        base.mkdir(parents=True)
        lines = "".join(json.dumps({"id": f"td-x-{i}", "state": "s",
                                    "eval_only": eval_only}) + "\n"
                        for i in range(rows))
        (base / "episodes.jsonl").write_text(lines, encoding="utf-8")
        from eval import splits as S
        (base / "manifest.json").write_text(json.dumps({
            "dataset": "LocalLLaMA/typed-decisions", "revision": "abc",
            "seed": 1, "eval_only": eval_only,
            "files": {"episodes.jsonl": {
                "rows": rows,
                "sha256": S.sha256_file(base / "episodes.jsonl")}}}))
        return base

    def test_the_cut_comes_back_with_its_sha_and_reserved_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self._publish(Path(tmp))
            cut = K.external_cut("typed-decisions", "test", external_dir=Path(tmp))
            self.assertEqual(cut["rows"], 3)
            self.assertTrue(cut["reserved"])
            self.assertEqual(cut["revision"], "abc")
            self.assertEqual(len(cut["sha256"]), 64)
            again = K.external_cut("typed-decisions", "test", sha256=cut["sha256"],
                                   external_dir=Path(tmp))
            self.assertEqual(again["episodes"], cut["episodes"])
            self.assertTrue((base / "episodes.jsonl").exists())

    def test_a_moved_file_or_another_sha_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self._publish(Path(tmp))
            with self.assertRaises(ValueError):
                K.external_cut("typed-decisions", "test", sha256="0" * 64,
                               external_dir=Path(tmp))
            with (base / "episodes.jsonl").open("a") as fh:
                fh.write(json.dumps({"id": "td-x-9"}) + "\n")
            with self.assertRaises(ValueError):
                K.external_cut("typed-decisions", "test", external_dir=Path(tmp))

    def test_a_cut_that_was_never_converted_says_how_to_get_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as cm:
                K.external_cut("typed-decisions", "test", external_dir=Path(tmp))
            self.assertIn("convert", str(cm.exception))

    @unittest.skipUnless((K.EXTERNAL_DIR / "typed-decisions" / "test"
                          / "manifest.json").exists(),
                         "typed-decisions not converted on this machine")
    def test_the_published_test_cut_is_reserved_and_verifies(self):
        cut = K.external_cut("typed-decisions", "test")
        self.assertTrue(cut["reserved"])
        self.assertEqual(cut["rows"], cut["rows"])
        self.assertEqual(cut["rows"], len(cut["episodes"]))
        self.assertTrue(all(ep["eval_only"] for ep in cut["episodes"]))
