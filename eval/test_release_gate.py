"""Unit tests for eval.release_gate — the release criterion applied
mechanically (#T-release-gate).

What these tests police is the property the whole task exists for: the
verdict must be a FUNCTION OF THE DOCUMENT. The test the task names is
`test_the_verdict_is_recomputed_from_the_markdown`: the same evidence, two
different `release-criteria.md`, two different answers — which is only
possible if no threshold lives in the code.

The rest guard the honesty rules: absent evidence is a FAILED criterion (not
a skipped one), an unsigned criterion still produces a verdict, and every
threshold must carry its reason.
"""
import json
import tempfile
import unittest
from pathlib import Path

from . import gate_rules as G
from . import release_gate as RG

REAL = RG.CRITERIA_MD


def write_criteria(tmp: str, criteria: dict,
                   signed_by: str = "pending-operator",
                   signed_utc=None, name: str = "criteria.md") -> Path:
    utc = "null" if signed_utc is None else signed_utc
    path = Path(tmp) / name
    path.write_text(
        "# test criterion\n\n"
        f"{RG.MACHINE_MARKER}\n"
        "```json\n" + json.dumps(criteria, indent=2) + "\n```\n\n"
        f"{RG.SIGNATURE_MARKER}\n"
        "```yaml\nsignature:\n"
        f"  signed_by: {signed_by}\n  signed_utc: {utc}\n"
        "  criteria_version: 1\n```\n")
    return path


class TestCriteriaParsing(unittest.TestCase):
    def test_the_real_document_parses(self):
        loaded = RG.load_criteria()
        self.assertEqual(loaded["criteria_version"], 1)
        self.assertEqual(loaded["criteria_path"],
                         ".meshkore/docs/release-criteria.md")
        self.assertEqual(len(loaded["criteria_sha"]), 64)

    def test_the_sha_is_the_sha_of_the_whole_file(self):
        self.assertEqual(RG.load_criteria()["criteria_sha"],
                         G.sha256_file(REAL))

    def test_editing_the_document_changes_the_sha(self):
        crit = RG.load_criteria()["criteria"]
        with tempfile.TemporaryDirectory() as tmp:
            a = RG.load_criteria(write_criteria(tmp, crit, name="a.md"))
            crit2 = json.loads(json.dumps(crit))
            crit2["thresholds"]["unseen_accuracy_all_min"]["value"] = 0.1
            b = RG.load_criteria(write_criteria(tmp, crit2, name="b.md"))
            self.assertNotEqual(a["criteria_sha"], b["criteria_sha"])

    def test_every_threshold_carries_its_reason(self):
        for name, spec in RG.load_criteria()["criteria"]["thresholds"].items():
            with self.subTest(threshold=name):
                self.assertTrue(str(spec.get("why", "")).strip())

    def test_a_threshold_without_a_reason_is_rejected(self):
        crit = json.loads(json.dumps(RG.load_criteria()["criteria"]))
        crit["thresholds"]["unseen_ece_max"].pop("why")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                RG.load_criteria(write_criteria(tmp, crit))

    def test_a_document_without_the_machine_block_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.md"
            path.write_text("# no block here\n")
            with self.assertRaises(ValueError):
                RG.load_criteria(path)


class TestSignature(unittest.TestCase):
    def test_the_shipped_criterion_is_an_unsigned_proposal(self):
        loaded = RG.load_criteria()
        self.assertFalse(loaded["criteria_signed"])
        self.assertEqual(loaded["signature"]["signed_by"], "pending-operator")
        self.assertIsNone(loaded["signature"]["signed_utc"])

    def test_an_unsigned_criterion_still_produces_a_verdict(self):
        """Signing pins the criterion; it does not enable it."""
        gate = RG.run(write=False, mark=False)
        self.assertFalse(gate["criteria_signed"])
        self.assertIn(gate["verdict"], ("GO", "NO-GO"))
        self.assertTrue(gate["criteria_results"])

    def test_an_operator_signature_is_recognised(self):
        crit = RG.load_criteria()["criteria"]
        with tempfile.TemporaryDirectory() as tmp:
            path = write_criteria(tmp, crit, signed_by="ricart",
                                  signed_utc="2026-09-22T09:00:00Z")
            loaded = RG.load_criteria(path)
            self.assertTrue(loaded["criteria_signed"])
            self.assertEqual(loaded["signature"]["signed_by"], "ricart")

    def test_a_signed_by_without_a_date_does_not_count(self):
        crit = RG.load_criteria()["criteria"]
        with tempfile.TemporaryDirectory() as tmp:
            loaded = RG.load_criteria(
                write_criteria(tmp, crit, signed_by="ricart"))
            self.assertFalse(loaded["criteria_signed"])


class TestVerdictComesFromTheDocument(unittest.TestCase):
    """The test the task names: no threshold constant in the code."""

    def _evaluate(self, criteria):
        unseen = json.loads((RG.ROOT / criteria["evidence"]["unseen"]).read_text())
        latency = json.loads(
            (RG.ROOT / criteria["evidence"]["latency"]["artifact"]).read_text())
        return {r["criterion"]: r
                for r in RG.evaluate(criteria, unseen, latency, None)}

    def test_the_verdict_is_recomputed_from_the_markdown(self):
        crit = RG.load_criteria()["criteria"]
        strict = self._evaluate(crit)
        self.assertFalse(strict["unseen_accuracy_all_min"]["pass"])

        relaxed = json.loads(json.dumps(crit))
        relaxed["thresholds"]["unseen_accuracy_all_min"]["value"] = 0.10
        with tempfile.TemporaryDirectory() as tmp:
            loaded = RG.load_criteria(write_criteria(tmp, relaxed))
        after = self._evaluate(loaded["criteria"])

        # SAME evidence, DIFFERENT document, DIFFERENT answer.
        self.assertTrue(after["unseen_accuracy_all_min"]["pass"])
        self.assertEqual(strict["unseen_accuracy_all_min"]["measured"],
                         after["unseen_accuracy_all_min"]["measured"])
        self.assertEqual(after["unseen_accuracy_all_min"]["threshold"], 0.10)

    def test_tightening_the_document_flips_a_passing_criterion(self):
        crit = json.loads(json.dumps(RG.load_criteria()["criteria"]))
        self.assertTrue(self._evaluate(crit)["unseen_ece_max"]["pass"])
        crit["thresholds"]["unseen_ece_max"]["value"] = 0.001
        self.assertFalse(self._evaluate(crit)["unseen_ece_max"]["pass"])

    def test_the_verdict_names_the_document_it_applied(self):
        gate = RG.run(write=False, mark=False)
        self.assertEqual(gate["criteria_sha"], G.sha256_file(REAL))
        self.assertEqual(gate["criteria_path"],
                         ".meshkore/docs/release-criteria.md")


class TestEvidenceRules(unittest.TestCase):
    def test_absent_evidence_is_a_failed_criterion_not_a_skipped_one(self):
        crit = RG.load_criteria()["criteria"]
        results = {r["criterion"]: r for r in RG.evaluate(crit, None, None, None)}
        for name, r in results.items():
            with self.subTest(criterion=name):
                self.assertFalse(r["pass"])
                self.assertEqual(r["evidence"], "ABSENT")

    def test_the_teacher_kappa_has_no_artifact_and_therefore_fails(self):
        gate = RG.run(write=False, mark=False)
        r = next(x for x in gate["criteria_results"]
                 if x["criterion"] == "teacher_cohen_kappa_min")
        self.assertFalse(r["pass"])
        self.assertEqual(r["evidence"], "ABSENT")
        self.assertIn("teacher_cohen_kappa_min", gate["absent_evidence"])

    def test_a_teacher_artifact_below_the_floor_still_fails(self):
        crit = RG.load_criteria()["criteria"]
        results = {r["criterion"]: r for r in RG.evaluate(
            crit, None, None, {"cohen_kappa": 0.0})}
        r = results["teacher_cohen_kappa_min"]
        self.assertFalse(r["pass"])
        self.assertEqual(r["evidence"], "present")

    def test_an_unattributed_latency_cannot_sustain_a_release(self):
        """40 ms is under the 150 ms budget, and it still fails: the
        artifact names no model_version (rule C2)."""
        gate = RG.run(write=False, mark=False)
        r = next(x for x in gate["criteria_results"]
                 if x["criterion"] == "latency_p95_max_ms")
        self.assertTrue(r["under_budget"])
        self.assertFalse(r["model_version_present"])
        self.assertFalse(r["pass"])

    def test_the_exempt_cut_is_read_from_the_evidence_not_the_code(self):
        unseen = json.loads((RG.ROOT / "artifacts" / "gates"
                             / "T-unseen-labels" / "gate.json").read_text())
        self.assertTrue(RG._cut_is_exempt(unseen, "boolq"))
        self.assertIsNone(RG._cut_is_exempt(unseen, "huffpost"))
        crit = RG.load_criteria()["criteria"]
        r = {x["criterion"]: x for x in RG.evaluate(crit, unseen, None, None)}
        self.assertNotIn("boolq", r["unseen_accuracy_cut_min"]["measured"])
        self.assertIn("huffpost", r["unseen_accuracy_cut_min"]["measured"])


class TestPublishedVerdict(unittest.TestCase):
    """What the gate actually wrote to disk."""

    def setUp(self):
        self.gate = json.loads(RG.GATE_PATH.read_text())

    def test_the_gate_artifact_exists_with_verdict_and_criteria_sha(self):
        self.assertIn(self.gate["verdict"], ("GO", "NO-GO"))
        self.assertEqual(self.gate["criteria_sha"], G.sha256_file(REAL))
        self.assertFalse(self.gate["criteria_signed"])

    def test_the_published_verdict_is_a_no_go(self):
        """Not an assertion about quality: an assertion that the gate
        publishes what it measured. Eight criteria fail on real numbers."""
        self.assertEqual(self.gate["verdict"], "NO-GO")
        self.assertFalse(self.gate["pass"])
        self.assertTrue(self.gate["failed_criteria"])

    def test_the_gate_is_not_uniformly_red(self):
        """A gate that fails everything proves nothing about the gate."""
        passed = [r for r in self.gate["criteria_results"] if r["pass"]]
        self.assertTrue(passed, "no criterion passes: the gate may be rigged")
        self.assertIn("unseen_ece_max", [r["criterion"] for r in passed])

    def test_the_numbers_are_the_ones_t_unseen_labels_published(self):
        unseen = json.loads((RG.ROOT / "artifacts" / "gates"
                             / "T-unseen-labels" / "gate.json").read_text())
        by = {r["criterion"]: r for r in self.gate["criteria_results"]}
        self.assertEqual(by["unseen_accuracy_all_min"]["measured"],
                         unseen["headline"]["overall"]["accuracy_unseen"])
        self.assertEqual(by["seen_unseen_accuracy_drop_max"]["measured"],
                         unseen["headline"]["overall"]["accuracy_drop"])

    def test_the_coherence_audit_is_published_beside_it(self):
        audit = json.loads(RG.AUDIT_PATH.read_text())
        self.assertFalse(audit["pass"])
        self.assertIn("T-release", audit["invalid"])
        self.assertEqual(audit["criteria_sha"], self.gate["criteria_sha"])

    def test_incoherent_evidence_blocks_the_release(self):
        self.assertIn("T-pointer-head",
                      self.gate["coherence"]["blocking_this_release"])


if __name__ == "__main__":
    unittest.main()
