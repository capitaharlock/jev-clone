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
    "require_chance_next_to_accuracy": {"value": True, "why": "test"},
}
#: the three companions R2 demands beside every accuracy
R2 = {"chance": 0.012987, "cardinality": 77,
      "accuracy_ci95": [0.009002, 0.016888]}


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
                "pass": True, "split_sha256": SEALED, "accuracy": 0.9, **R2})
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
                         report={"accuracy": 0.9, **R2})
            self.assertNotIn("C2", rules_of(G.check_gate_dir(d, COHERENCE,
                                                             Path(tmp))))

    def test_an_empty_model_version_is_no_model_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = gate_dir(tmp, "T-blank", gate={
                "pass": True, "model_version": "  ",
                "split_sha256": SEALED, "accuracy": 0.9, **R2})
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
                "accuracy": 0.9, **R2})
            sealed = gate_dir(tmp, "T-sealed", gate={
                "pass": True, "model_version": "m-1", "seed": 20260921,
                "split_sha256": SEALED, "accuracy": 0.9, **R2})
            self.assertIn("C3", rules_of(G.check_gate_dir(seeded, COHERENCE,
                                                          Path(tmp))))
            self.assertNotIn("C3", rules_of(G.check_gate_dir(sealed, COHERENCE,
                                                             Path(tmp))))

    def test_a_passing_group_split_sealed_check_is_a_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = gate_dir(tmp, "T-grp", gate={
                "pass": True, "model_version": "m-1", "accuracy": 0.9, **R2,
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
                "pass": True, "model_version": "m-1", "accuracy": 0.9, **R2,
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
                "pass": True, "model_version": "m-1", "accuracy": 0.9, **R2,
                "splits": {"x": {"manifest": "artifacts/splits/x/manifest.json",
                                 "manifest_sha256": SEALED}}})
            r = G.check_gate_dir(d, COHERENCE, Path(tmp))
            self.assertIn("C3", rules_of(r))
            broken = next(e for e in r["errors"] if "broken" in e)
            self.assertEqual(broken["broken"][0]["why"], "sha256 mismatch")


class TestC4AccuracyNeedsItsChance(unittest.TestCase):
    """R2 — an accuracy with no chance rate beside it is not a result."""

    def test_a_bare_accuracy_fails_the_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = gate_dir(tmp, "T-bare", gate={
                "pass": True, "model_version": "m-1",
                "split_sha256": SEALED, "accuracy": 0.215})
            r = G.check_gate_dir(d, COHERENCE, Path(tmp))
            self.assertFalse(r["pass"])
            self.assertIn("C4", rules_of(r))
            err = next(e for e in r["errors"] if e["rule"] == "C4")
            self.assertEqual(err["severity"], "error")
            self.assertEqual(sorted(err["offending"][0]["missing"]),
                             ["accuracy_ci95", "cardinality", "chance"])

    def test_each_companion_is_required_on_its_own(self):
        for drop in ("chance", "cardinality", "accuracy_ci95"):
            with self.subTest(missing=drop), \
                    tempfile.TemporaryDirectory() as tmp:
                doc = {"pass": True, "model_version": "m-1",
                       "split_sha256": SEALED, "accuracy": 0.215, **R2}
                doc.pop(drop)
                d = gate_dir(tmp, "T-partial", gate=doc)
                r = G.check_gate_dir(d, COHERENCE, Path(tmp))
                self.assertIn("C4", rules_of(r))
                err = next(e for e in r["errors"] if e["rule"] == "C4")
                self.assertEqual(err["offending"][0]["missing"], [drop])

    def test_a_complete_accuracy_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = gate_dir(tmp, "T-full", gate={
                "pass": True, "model_version": "m-1",
                "split_sha256": SEALED, "accuracy": 0.012338, **R2})
            self.assertTrue(G.check_gate_dir(d, COHERENCE, Path(tmp))["pass"])

    def test_a_red_gate_does_not_escape_the_rule(self):
        """C1-C3 police green claims. A bare accuracy misleads whatever
        verdict sits beside it, so C4 does not wait for one."""
        with tempfile.TemporaryDirectory() as tmp:
            d = gate_dir(tmp, "T-red", gate={
                "pass": False, "model_version": "m-1", "accuracy": 0.05})
            self.assertIn("C4", rules_of(G.check_gate_dir(d, COHERENCE,
                                                          Path(tmp))))

    def test_an_enclosing_object_may_carry_the_companions(self):
        """A per-dataset breakdown is read with its cut's K and chance."""
        with tempfile.TemporaryDirectory() as tmp:
            d = gate_dir(tmp, "T-nested", gate={
                "pass": True, "model_version": "m-1",
                "split_sha256": SEALED,
                "unseen": {"accuracy": 0.093, **R2,
                           "per_dataset": {"huffpost": {"accuracy": 0.078,
                                                        "n": 38}}}})
            self.assertTrue(G.check_gate_dir(d, COHERENCE, Path(tmp))["pass"])

    def test_a_sibling_cut_does_not_lend_its_chance(self):
        """`seen` publishing K does not make `unseen`'s bare accuracy
        readable: the companions must ENCLOSE the number, not neighbour it."""
        with tempfile.TemporaryDirectory() as tmp:
            d = gate_dir(tmp, "T-sibling", gate={
                "pass": True, "model_version": "m-1",
                "split_sha256": SEALED,
                "seen": {"accuracy": 0.62, **R2},
                "unseen": {"accuracy": 0.093, "n": 3008}})
            r = G.check_gate_dir(d, COHERENCE, Path(tmp))
            self.assertIn("C4", rules_of(r))
            off = next(e for e in r["errors"] if e["rule"] == "C4")
            self.assertEqual([o["at"] for o in off["offending"]],
                             ["/unseen"])

    def test_a_cited_third_party_number_is_exempt(self):
        """R3 keeps citations honest; C4 is about OUR measurements."""
        with tempfile.TemporaryDirectory() as tmp:
            d = gate_dir(tmp, "T-cite", gate={
                "pass": True, "model_version": "m-1",
                "split_sha256": SEALED, "same_rows": False,
                "references": [{"who": "Jev (teacher)", "accuracy": 0.924,
                                "source": "github.com/...",
                                "caveat": "public benchmark"}]})
            self.assertTrue(G.check_gate_dir(d, COHERENCE, Path(tmp))["pass"])

    def test_a_scalar_is_not_an_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = gate_dir(tmp, "T-scalar", gate={
                "pass": True, "model_version": "m-1",
                "split_sha256": SEALED, "accuracy": 0.215,
                "chance": 0.2, "cardinality": 5, "accuracy_ci95": 0.19})
            self.assertIn("C4", rules_of(G.check_gate_dir(d, COHERENCE,
                                                          Path(tmp))))

    def test_mean_k_counts_as_the_cardinality_of_a_variable_k_cut(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = gate_dir(tmp, "T-meank", gate={
                "pass": True, "model_version": "m-1",
                "split_sha256": SEALED, "accuracy": 0.093,
                "chance": 0.2067, "mean_k": 4.3684,
                "accuracy_ci95": [0.081, 0.106]})
            self.assertTrue(G.check_gate_dir(d, COHERENCE, Path(tmp))["pass"])

    def test_the_rule_can_be_turned_off_only_from_the_criteria(self):
        off = dict(COHERENCE,
                   require_chance_next_to_accuracy={"value": False,
                                                    "why": "t"})
        with tempfile.TemporaryDirectory() as tmp:
            d = gate_dir(tmp, "T-bare", gate={
                "pass": True, "model_version": "m-1",
                "split_sha256": SEALED, "accuracy": 0.215})
            self.assertIn("C4", rules_of(G.check_gate_dir(d, COHERENCE,
                                                          Path(tmp))))
            self.assertNotIn("C4", rules_of(G.check_gate_dir(d, off,
                                                             Path(tmp))))

    def test_the_runner_exits_non_zero_on_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = gate_dir(tmp, "T-bare", gate={
                "pass": True, "model_version": "m-1",
                "split_sha256": SEALED, "accuracy": 0.215})
            self.assertEqual(G.main(["check", str(d)]), 1)


class TestScanAndMarking(unittest.TestCase):
    def _tree(self, tmp):
        gate_dir(tmp, "T-bad", gate={"pass": True, "model_version": "m-1",
                                     "split_sha256": SEALED,
                                     "cohen_kappa": 0.0})
        gate_dir(tmp, "T-good", gate={"pass": True, "model_version": "m-1",
                                      "split_sha256": SEALED,
                                      "accuracy": 0.9, **R2})
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

    def test_the_honest_red_gate_is_not_flagged(self):
        """#T-unseen-labels publishes 30+ accuracies, every one of them with
        its chance, its mean K and its interval: red, and coherent."""
        r = G.check_gate_dir(G.GATES_DIR / "T-unseen-labels", COHERENCE)
        self.assertTrue(r["pass"], r["errors"])
        self.assertGreater(r["accuracy_claims"]["published"], 20)
        self.assertEqual(r["accuracy_claims"]["without_companions"], 0)

    def test_the_full_space_artifact_is_not_flagged(self):
        """Its two teacher numbers are citations; ours carries R2 in full."""
        r = G.check_gate_dir(G.GATES_DIR / "T-teacher-probe", COHERENCE)
        self.assertTrue(r["pass"], r["errors"])
        self.assertEqual(r["accuracy_claims"]["without_companions"], 0)

    def test_t_train_real_trips_c4_on_its_per_dataset_breakdown(self):
        """The rule earns its keep on the real tree: the 2026-09-21 gate
        publishes `checks.unseen_beats_chance.accuracy` and a per-dataset
        split of it with chance and CI but no K, so the reader cannot tell
        0.1836 on banking77 from its own chance. `compose_gate` now emits
        `mean_k` there; the artifact on disk predates the fix and stays as
        published — marked, never rewritten."""
        r = G.check_gate_dir(G.GATES_DIR / "T-train-real", COHERENCE)
        self.assertEqual(rules_of(r), {"C4"})
        err = next(e for e in r["errors"] if e["rule"] == "C4")
        self.assertTrue(all(o["missing"] == ["cardinality"]
                            for o in err["offending"]))


class TestReadingCoherence(unittest.TestCase):
    """C6 — a gate whose reading contradicts its numbers FAILS (#T-battery-
    metrics). Synthetic dirs only; the real corrections are asserted below."""

    FIG = {"accuracy": 0.0, "accuracy_ci95": [0.0, 0.003827],
           "chance": 0.012987, "cardinality": 77, "n": 1000,
           "accuracy_options_only": 0.009,
           "accuracy_options_only_ci95": [0.004742, 0.017016]}

    def test_a_reading_that_contradicts_its_interval_fails_the_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = gate_dir(tmp, "T-synthetic", gate={
                "pass": False, "primary": {"arm": dict(self.FIG)},
                "verdict": {"reading": "the primary interval contains chance"}})
            r = G.check_gate_dir(d, COHERENCE)
            self.assertFalse(r["pass"])
            self.assertIn("C6", rules_of(r))
            self.assertEqual(r["readings"]["incoherent"], 1)

    def test_the_corrected_sentence_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = gate_dir(tmp, "T-synthetic", gate={
                "pass": False, "primary": {"arm": dict(self.FIG)},
                "verdict": {"reading":
                            "the primary interval excludes chance from "
                            "below; the interval that contains chance is "
                            "the forced one, `unknown` out of the race"}})
            self.assertTrue(G.check_gate_dir(d, COHERENCE)["pass"])

    def test_a_metrics_suite_report_missing_a_metric_fails_the_runner(self):
        from . import metrics_suite as M
        rows = [{"row_id": "a", "family": "f1", "variant_group": "g1",
                 "k": 4, "gold_index": 0, "pred": 0,
                 "probs": [0.7, 0.1, 0.1, 0.1, 0.05]}]
        rep = M.report(rows, cut={"name": "synthetic"},
                       calibration={"temperature": 1.0, "fitted_on": "dev",
                                    "verified_on": "test"},
                       permutation={"pass": True}, tracking={"pass": False})
        del rep["macro"]
        with tempfile.TemporaryDirectory() as tmp:
            d = gate_dir(tmp, "T-synthetic", suite=rep)
            r = G.check_gate_dir(d, COHERENCE)
            self.assertIn("C5", rules_of(r))
            self.assertEqual(r["metrics_suite"]["incomplete"], 1)


class TestSuiteIsTheOnlyArithmetic(unittest.TestCase):
    """C7 — a PILOT gate that computes its own figure FAILS (#T-battery-
    metrics). C5 only reaches a document already stamped `jev.metrics.v1`;
    this is the rule for the run that stamps nothing and does the arithmetic
    itself."""

    HAND = {"pass": True, "model_version": "ckpt-42",
            "split_sha256": SEALED,
            "measured": {"accuracy": 0.71, **R2}}

    def suite_report(self):
        from . import metrics_suite as M
        rows = [{"row_id": "a", "family": "f1", "variant_group": "g1",
                 "k": 4, "gold_index": 0, "pred": 0,
                 "probs": [0.7, 0.1, 0.1, 0.1, 0.05]},
                {"row_id": "b", "family": "f1", "variant_group": "g1",
                 "k": 4, "gold_index": 1, "pred": 1,
                 "probs": [0.1, 0.7, 0.1, 0.1, 0.05]}]
        return M.report(rows, cut={"name": "battery-dev", "seal": SEALED},
                        model_version="ckpt-42", task="T-battery-sealed",
                        calibration={"temperature": 1.0, "fitted_on": "dev",
                                     "verified_on": "test"},
                        permutation={"pass": True}, tracking={"pass": False})

    def test_a_pilot_gate_computing_its_own_accuracy_fails_the_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = gate_dir(tmp, "T-battery-sealed", gate=dict(self.HAND))
            r = G.check_gate_dir(d, COHERENCE)
            self.assertFalse(r["pass"])
            self.assertIn("C7", rules_of(r))
            self.assertEqual(r["metrics_suite"]["hand_computed"], 1)
            err = next(e for e in r["errors"] if e["rule"] == "C7")
            self.assertEqual(err["severity"], "error")
            self.assertEqual(err["offending"][0]["at"], "/measured/accuracy")

    def test_the_same_figure_published_through_the_suite_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = gate_dir(tmp, "T-battery-sealed", metrics=self.suite_report())
            r = G.check_gate_dir(d, COHERENCE)
            self.assertNotIn("C7", rules_of(r), r["errors"])
            self.assertEqual(r["metrics_suite"]["reports"], ["metrics.json"])
            self.assertEqual(r["metrics_suite"]["hand_computed"], 0)

    def test_a_gate_outside_the_pilot_is_not_retro_flagged(self):
        """The historical gates are what the suite replaces, not offenders."""
        with tempfile.TemporaryDirectory() as tmp:
            d = gate_dir(tmp, "T-some-2026-05-gate", gate=dict(self.HAND))
            r = G.check_gate_dir(d, COHERENCE)
            self.assertNotIn("C7", rules_of(r))
            self.assertFalse(r["metrics_suite"]["pilot"])

    def test_a_quoted_third_party_number_is_exempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = gate_dir(tmp, "T-preflight-refs", gate={
                "pass": False,
                "reference": {"who": "ModernBERT-base-zeroshot-v2.0",
                              "source": "model card",
                              "accuracy": 0.63, **R2}})
            r = G.check_gate_dir(d, COHERENCE)
            self.assertNotIn("C7", rules_of(r), r["errors"])
            self.assertEqual(r["metrics_suite"]["hand_computed"], 0)


class TestPilotCensusOnDisk(unittest.TestCase):
    """What the pilot publishes TODAY, measured rather than asserted."""

    def test_no_pilot_gate_on_disk_computes_its_own_figure(self):
        census = G.pilot_census(coherence=COHERENCE)
        self.assertEqual(census["offenders"], [])
        self.assertTrue(census["pass"])

    def test_the_only_pilot_gate_on_disk_publishes_no_figure_yet(self):
        census = G.pilot_census(coherence=COHERENCE)
        self.assertEqual([g["gate"] for g in census["on_disk"]],
                         ["T-battery-dev"])
        self.assertEqual(census["publishing_no_figure"], ["T-battery-dev"])
        self.assertIn("T-battery-sealed", census["absent"])


class TestCorrectedReadingsOnDisk(unittest.TestCase):
    """The three readings #T-battery-metrics corrected, as published."""

    def test_the_two_named_gates_now_pass_c6(self):
        for task in ("T-fullspace-objective", "T-bigk-optsets"):
            r = G.check_gate_dir(G.GATES_DIR / task, COHERENCE)
            self.assertNotIn("C6", rules_of(r), f"{task}: {r['errors']}")
            self.assertEqual(r["readings"]["incoherent"], 0, task)

    def test_the_corrections_keep_the_wrong_sentence_on_the_record(self):
        doc = json.loads((G.GATES_DIR / "T-fullspace-objective"
                          / "gate.json").read_text())
        self.assertIn("contains chance",
                      doc["verdict"]["reading_corrected"]["was"])
        # and the numbers it was read off are untouched
        self.assertEqual(doc["primary"]["arm"]["accuracy_ci95"],
                         [0.0, 0.003827])
        self.assertEqual(doc["primary"]["arm"]["chance"], 0.012987)


if __name__ == "__main__":
    unittest.main()
