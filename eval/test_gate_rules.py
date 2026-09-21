"""Unit tests for eval.gate_rules — the coherence rules (#T-release-gate).

Nothing here touches the real `artifacts/gates/` tree except to read it: every
rule is exercised against SYNTHETIC gate directories built in a tempdir, so a
rule can be proved to fire without waiting for the repo to publish a bad
number.

The test the task names explicitly is `test_synthetic_green_with_zero_kappa_
fails_the_runner`: an artifact with `pass: true` and `cohen_kappa: 0.0` must
make the gate runner FAIL — error, not warning.
"""
import json
import tempfile
import unittest
from pathlib import Path

from . import gate_rules as G

SEALED = "a" * 64
COHERENCE = {
    "cohen_kappa_floor": {"value": 0.10, "why": "test"},
    "require_model_version": {"value": True, "why": "test"},
    "require_sealed_split": {"value": True, "why": "test"},
}


def gate_dir(tmp: str, name: str, **files) -> Path:
    d = Path(tmp) / name
    d.mkdir(parents=True, exist_ok=True)
    for fname, doc in files.items():
        (d / f"{fname}.json").write_text(json.dumps(doc, indent=2))
    return d


def rules_of(result: dict) -> set:
    return {e["rule"] for e in result["errors"]}


class TestGreenClaim(unittest.TestCase):
    def test_top_level_pass_is_authoritative(self):
        doc = {"pass": False, "checks": {"a": {"pass": True}},
               "verdict": "GO anyway"}
        claim = G.green_claim(doc)
        self.assertFalse(claim["green"])
        self.assertEqual(claim["by"], "pass")

    def test_nested_verdict_decides_when_there_is_no_pass(self):
        """The `T-release/release.json` shape: no `pass`, a nested GO."""
        doc = {"numbers": {"cohen_kappa": 0.0, "cold_verdict": "GO"}}
        claim = G.green_claim(doc)
        self.assertTrue(claim["green"])
        self.assertEqual(claim["by"], "/numbers/cold_verdict")

    def test_a_failing_verdict_is_not_green(self):
        self.assertFalse(G.green_claim({"verdict": "FAIL [x]"})["green"])

    def test_numbers_without_a_claim_are_not_green(self):
        claim = G.green_claim({"p95_ms": 12.0, "accuracy": 0.9})
        self.assertFalse(claim["green"])
        self.assertIsNone(claim["by"])


class TestC1Kappa(unittest.TestCase):
    def test_synthetic_green_with_zero_kappa_fails_the_runner(self):
        """The test the task names: pass:true + cohen_kappa 0.0 => ERROR."""
        with tempfile.TemporaryDirectory() as tmp:
            d = gate_dir(tmp, "T-synthetic", gate={
                "pass": True, "model_version": "m-1",
                "split_sha256": SEALED,
                "iaa": {"cohen_kappa": 0.0, "exact_agree_rate": 0.852}})
            r = G.check_gate_dir(d, COHERENCE, Path(tmp))
            self.assertFalse(r["pass"])
            self.assertIn("C1", rules_of(r))
            err = next(e for e in r["errors"] if e["rule"] == "C1")
            self.assertEqual(err["severity"], "error")
            self.assertEqual(err["offending"][0]["cohen_kappa"], 0.0)

    def test_the_runner_exits_non_zero_on_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = gate_dir(tmp, "T-synthetic", gate={
                "pass": True, "model_version": "m-1",
                "split_sha256": SEALED, "cohen_kappa": 0.0})
            self.assertEqual(G.main(["check", str(d)]), 1)

    def test_kappa_just_above_the_floor_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = gate_dir(tmp, "T-ok", gate={
                "pass": True, "model_version": "m-1",
                "split_sha256": SEALED, "cohen_kappa": 0.11})
            self.assertTrue(G.check_gate_dir(d, COHERENCE, Path(tmp))["pass"])

    def test_the_floor_comes_from_the_criteria_not_from_the_code(self):
        loose = dict(COHERENCE, cohen_kappa_floor={"value": -1.0, "why": "t"})
        with tempfile.TemporaryDirectory() as tmp:
            d = gate_dir(tmp, "T-synthetic", gate={
                "pass": True, "model_version": "m-1",
                "split_sha256": SEALED, "cohen_kappa": 0.0})
            self.assertIn("C1", rules_of(G.check_gate_dir(d, COHERENCE,
                                                          Path(tmp))))
            self.assertNotIn("C1", rules_of(G.check_gate_dir(d, loose,
                                                             Path(tmp))))

    def test_a_red_gate_may_publish_a_zero_kappa(self):
        """The rules police GREEN claims, not bad numbers. A bad number
        published as bad is exactly what this repo wants."""
        with tempfile.TemporaryDirectory() as tmp:
            d = gate_dir(tmp, "T-honest", gate={
                "pass": False, "cohen_kappa": 0.0})
            self.assertTrue(G.check_gate_dir(d, COHERENCE, Path(tmp))["pass"])


class TestC2ModelVersion(unittest.TestCase):
    def test_a_metric_without_model_version_fails_a_green(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = gate_dir(tmp, "T-nomv", gate={
                "pass": True, "split_sha256": SEALED, "accuracy": 0.9})
            self.assertIn("C2", rules_of(G.check_gate_dir(d, COHERENCE,
                                                          Path(tmp))))

    def test_latency_counts_as_a_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = gate_dir(tmp, "T-lat", gate={"pass": True, "p95_ms": 40.07})
            self.assertIn("C2", rules_of(G.check_gate_dir(d, COHERENCE,
                                                          Path(tmp))))

    def test_a_sibling_file_may_carry_the_model_version(self):
        """The unit is the gate DIRECTORY: one publication, several files."""
        with tempfile.TemporaryDirectory() as tmp:
            d = gate_dir(tmp, "T-pair",
                         gate={"pass": True, "model_version": "m-1",
                               "split_sha256": SEALED},
                         report={"accuracy": 0.9})
            self.assertNotIn("C2", rules_of(G.check_gate_dir(d, COHERENCE,
                                                             Path(tmp))))

    def test_an_empty_model_version_is_no_model_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = gate_dir(tmp, "T-blank", gate={
                "pass": True, "model_version": "  ",
                "split_sha256": SEALED, "accuracy": 0.9})
            self.assertIn("C2", rules_of(G.check_gate_dir(d, COHERENCE,
                                                          Path(tmp))))

    def test_a_green_that_publishes_no_metric_is_fine(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = gate_dir(tmp, "T-plain", gate={"pass": True, "commit": "abc"})
            self.assertTrue(G.check_gate_dir(d, COHERENCE, Path(tmp))["pass"])


class TestC3SealedSplit(unittest.TestCase):
    def test_a_quality_metric_needs_a_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = gate_dir(tmp, "T-unsealed", gate={
                "pass": True, "model_version": "m-1", "seed": 7, "ece": 0.02})
            self.assertIn("C3", rules_of(G.check_gate_dir(d, COHERENCE,
                                                          Path(tmp))))

    def test_a_seed_alone_is_not_a_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            seeded = gate_dir(tmp, "T-seed", gate={
                "pass": True, "model_version": "m-1", "seed": 20260921,
                "accuracy": 0.9})
            sealed = gate_dir(tmp, "T-sealed", gate={
                "pass": True, "model_version": "m-1", "seed": 20260921,
                "split_sha256": SEALED, "accuracy": 0.9})
            self.assertIn("C3", rules_of(G.check_gate_dir(seeded, COHERENCE,
                                                          Path(tmp))))
            self.assertNotIn("C3", rules_of(G.check_gate_dir(sealed, COHERENCE,
                                                             Path(tmp))))

    def test_a_passing_group_split_sealed_check_is_a_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = gate_dir(tmp, "T-grp", gate={
                "pass": True, "model_version": "m-1", "accuracy": 0.9,
                "criteria": {"group_split_sealed": {"pass": True}}})
            self.assertNotIn("C3", rules_of(G.check_gate_dir(d, COHERENCE,
                                                             Path(tmp))))

    def test_latency_alone_needs_no_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = gate_dir(tmp, "T-lat", gate={
                "pass": True, "model_version": "m-1", "p95_ms": 40.07})
            self.assertNotIn("C3", rules_of(G.check_gate_dir(d, COHERENCE,
                                                             Path(tmp))))

    def test_a_referenced_manifest_that_is_missing_breaks_the_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = gate_dir(tmp, "T-ghost", gate={
                "pass": True, "model_version": "m-1", "accuracy": 0.9,
                "splits": {"x": {"manifest": "artifacts/splits/x/manifest.json",
                                 "manifest_sha256": SEALED}}})
            r = G.check_gate_dir(d, COHERENCE, Path(tmp))
            self.assertIn("C3", rules_of(r))
            self.assertTrue(any("broken" in e for e in r["errors"]))

    def test_a_stale_manifest_digest_breaks_the_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            man = Path(tmp) / "artifacts" / "splits" / "x" / "manifest.json"
            man.parent.mkdir(parents=True)
            man.write_text('{"rows": 10}')
            d = gate_dir(tmp, "T-stale", gate={
                "pass": True, "model_version": "m-1", "accuracy": 0.9,
                "splits": {"x": {"manifest": "artifacts/splits/x/manifest.json",
                                 "manifest_sha256": SEALED}}})
            r = G.check_gate_dir(d, COHERENCE, Path(tmp))
            self.assertIn("C3", rules_of(r))
            broken = next(e for e in r["errors"] if "broken" in e)
            self.assertEqual(broken["broken"][0]["why"], "sha256 mismatch")


class TestScanAndMarking(unittest.TestCase):
    def _tree(self, tmp):
        gate_dir(tmp, "T-bad", gate={"pass": True, "model_version": "m-1",
                                     "split_sha256": SEALED,
                                     "cohen_kappa": 0.0})
        gate_dir(tmp, "T-good", gate={"pass": True, "model_version": "m-1",
                                      "split_sha256": SEALED,
                                      "accuracy": 0.9})
        gate_dir(tmp, "T-empty")

    def test_scan_marks_the_offender_and_leaves_the_artifact_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._tree(tmp)
            before = (Path(tmp) / "T-bad" / "gate.json").read_text()
            rep = G.scan(Path(tmp), COHERENCE, mark=True,
                         criteria_sha="deadbeef", root=Path(tmp))
            self.assertFalse(rep["pass"])
            self.assertEqual(rep["invalid"], ["T-bad"])
            marker = Path(tmp) / "T-bad" / G.MARKER_NAME
            self.assertTrue(marker.exists())
            # marked, NOT deleted
            self.assertTrue((Path(tmp) / "T-bad" / "gate.json").exists())
            self.assertEqual((Path(tmp) / "T-bad" / "gate.json").read_text(),
                             before)
            mark = json.loads(marker.read_text())
            self.assertEqual(mark["criteria_sha"], "deadbeef")
            self.assertIn("C1", {e["rule"] for e in mark["errors"]})

    def test_a_clean_gate_is_not_marked(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._tree(tmp)
            G.scan(Path(tmp), COHERENCE, mark=True, root=Path(tmp))
            self.assertFalse((Path(tmp) / "T-good" / G.MARKER_NAME).exists())

    def test_a_stale_marker_is_removed_once_the_gate_is_fixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._tree(tmp)
            G.scan(Path(tmp), COHERENCE, mark=True, root=Path(tmp))
            (Path(tmp) / "T-bad" / "gate.json").write_text(json.dumps({
                "pass": True, "model_version": "m-1",
                "split_sha256": SEALED, "cohen_kappa": 0.7}))
            rep = G.scan(Path(tmp), COHERENCE, mark=True, root=Path(tmp))
            self.assertTrue(rep["pass"])
            self.assertFalse((Path(tmp) / "T-bad" / G.MARKER_NAME).exists())

    def test_the_marker_is_never_audited_as_a_gate_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._tree(tmp)
            G.scan(Path(tmp), COHERENCE, mark=True, root=Path(tmp))
            r = G.check_gate_dir(Path(tmp) / "T-bad", COHERENCE, Path(tmp))
            self.assertNotIn(G.MARKER_NAME, r["files"])


class TestPublishedGates(unittest.TestCase):
    """The rules applied to what this repo has actually published."""

    def test_t_release_is_invalid(self):
        r = G.check_gate_dir(G.GATES_DIR / "T-release", COHERENCE)
        self.assertFalse(r["pass"])
        self.assertIn("C1", rules_of(r))
        self.assertIn({"file": "release.json", "at": "/numbers/cohen_kappa",
                       "value": 0.0}, r["cohen_kappa"])

    def test_t_release_is_marked_on_disk_and_still_has_its_numbers(self):
        d = G.GATES_DIR / "T-release"
        self.assertTrue((d / G.MARKER_NAME).exists(),
                        "run `python3 -m eval.release_gate gate` first")
        release = json.loads((d / "release.json").read_text())
        self.assertEqual(release["numbers"]["cohen_kappa"], 0.0)
        self.assertEqual(release["numbers"]["exact_agree_rate"], 0.852)

    def test_the_honest_red_gates_are_not_flagged(self):
        for task in ("T-unseen-labels", "T-train-real"):
            with self.subTest(task=task):
                r = G.check_gate_dir(G.GATES_DIR / task, COHERENCE)
                self.assertTrue(r["pass"], r["errors"])


if __name__ == "__main__":
    unittest.main()
