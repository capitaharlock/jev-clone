"""Unit tests for eval.unseen — the primary metric (#T-unseen-labels).

Nothing here needs torch: the model-scoring path is one function
(`eval.calib.entries_from_samples`) and it is exercised by the real gate
run. What these tests police is the PROTOCOL, which is where a metric like
this actually breaks:

* the unseen label space is really absent from training, by text;
* the row partition is a GROUP split, so a template cannot sit on both
  sides of the fence (finding C);
* the cross-lingual arm reports itself unmet instead of quietly passing;
* the gate's `pass` is the AND of its criteria and cannot be talked up.
"""
import json
import os
import tempfile
import unittest

from data.optset import ALLOWED_DATASETS, assert_trainable
from training.python import train_decision as T

from . import unseen as U


class TestLocaleProtocol(unittest.TestCase):
    def test_locale_of_reads_the_massive_prefix(self):
        self.assertEqual(U.locale_of("[it-IT] svegliami alle cinque"), "it-IT")
        self.assertEqual(U.locale_of("no prefix here"), "")
        self.assertEqual(U.locale_of(""), "")

    def test_four_train_locales_two_eval_locales_no_overlap(self):
        self.assertEqual(len(U.TRAIN_LOCALES), 4)
        self.assertEqual(len(U.EVAL_LOCALES), 2)
        self.assertEqual(set(U.TRAIN_LOCALES) | set(U.EVAL_LOCALES),
                         set(U.MASSIVE_LOCALES))
        self.assertFalse(set(U.TRAIN_LOCALES) & set(U.EVAL_LOCALES))

    def test_eval_locales_are_the_mechanical_choice(self):
        """Fixed before measuring: no picking the flattering pair later."""
        self.assertEqual(list(U.EVAL_LOCALES),
                         sorted(U.MASSIVE_LOCALES)[-2:])


class TestGroupPartition(unittest.TestCase):
    def test_side_is_stable_and_only_two_sided(self):
        for i in range(200):
            side = U.eval_side(f"row number {i} about parcels", "banking77")
            self.assertIn(side, ("test", "calibration"))

    def test_same_skeleton_never_straddles_the_fence(self):
        """The finding-C invariant: surface variation is not a new group."""
        pairs = [
            ("my card arrived on 3 May and the fee was 12 EUR",
             "my card arrived on 9 June and the fee was 40 EUR"),
            ("Jordi, your transfer of $15 failed at 10:30",
             "Marta, your transfer of $99 failed at 18:45"),
        ]
        for a, b in pairs:
            self.assertEqual(U.eval_side(a, "banking77"),
                             U.eval_side(b, "banking77"),
                             f"{a!r} and {b!r} are the same template")

    def test_digits_alone_do_not_make_a_new_group(self):
        """2000 rows off one template are ONE group, hence one side."""
        sides = {U.eval_side(f"utterance {i} <num> parcels", "massive")
                 for i in range(2000)}
        self.assertEqual(len(sides), 1)

    def test_partition_splits_roughly_seventy_thirty(self):
        sides = [U.eval_side(f"a distinct sentence about topic w{i}",
                             "massive") for i in range(2000)]
        frac = sides.count("test") / len(sides)
        self.assertGreater(frac, 0.6)
        self.assertLess(frac, 0.8)

    def test_seal_is_idempotent_and_disjoint(self):
        sealed = U.seal_metric_split("banking77")
        self.assertEqual(sealed["shared_groups_metric_vs_calibration"], 0)
        again = U.seal_metric_split("banking77")
        self.assertEqual(sealed["manifest_sha256"], again["manifest_sha256"])
        self.assertEqual(sorted(sealed["files"]),
                         ["calibration.ids", "test.ids"])
        for meta in sealed["files"].values():
            self.assertEqual(len(meta["sha256"]), 64)


class TestUnseenTextAbsence(unittest.TestCase):
    """Gate criterion 1, on the real corpus."""

    @classmethod
    def setUpClass(cls):
        cls.holdout = T.build_holdout()

    def test_holdout_matches_the_published_protocol(self):
        h = self.holdout
        self.assertEqual(len(h["huffpost"].unseen), 11)
        self.assertEqual(len(h["huffpost"].seen), 30)
        self.assertGreaterEqual(len(h["banking77"].unseen), 2)
        self.assertGreaterEqual(len(h["massive"].unseen), 2)
        self.assertEqual(h["boolq"].unseen, [])
        self.assertIn("two-label pool", h["boolq"].exempt)

    def test_no_unseen_label_text_in_training(self):
        rep = U.unseen_text_absence(self.holdout, rows_cap=3000)
        self.assertTrue(rep["pass"], rep["hits"])
        self.assertEqual(rep["n_hits"], 0)
        self.assertGreater(rep["banned_texts"], 0)
        self.assertGreater(rep["checked"]["pool_labels"], 0)
        self.assertGreater(rep["checked"]["row_golds"], 0)
        self.assertGreater(rep["checked"]["samples"], 0)
        self.assertIn("TEXT", rep["compared_by"])

    def test_the_check_catches_a_label_left_in_the_pool(self):
        """Negative control: if the restriction fails, criterion 1 fails."""
        broken = {}
        for name, h in self.holdout.items():
            broken[name] = T.DatasetHoldout(
                name, seen=list(h.seen), unseen=list(h.seen[:1]),
                texts=dict(h.texts), exempt=h.exempt)
        rep = U.unseen_text_absence(broken, rows_cap=200)
        self.assertFalse(rep["pass"])
        self.assertTrue(rep["hits"]["distractor_pool"])
        self.assertGreater(rep["n_hits"], 0)

    def test_state_mentions_are_informational_only(self):
        rep = U.unseen_text_absence(self.holdout, rows_cap=2000)
        self.assertIn("informational_state_mentions", rep)
        self.assertTrue(rep["pass"])
        self.assertIn("never gating",
                      rep["informational_state_mentions"]["note"])


class TestEvalOnlyBenchmarks(unittest.TestCase):
    def test_firewall_still_refuses_them_for_training(self):
        for name in U.EVAL_ONLY:
            self.assertNotIn(name, ALLOWED_DATASETS)
            with self.assertRaises(ValueError):
                assert_trainable([name])

    def test_reclor_is_scored_on_the_labelled_split(self):
        self.assertEqual(U.EVAL_ONLY_SPLIT["reclor"], "calibration")
        self.assertIn("hidden", U.EVAL_ONLY_SPLIT_NOTE["reclor"])

    def test_samples_have_a_real_gold_and_the_asked_cardinality(self):
        samples = U.eval_only_samples("reclor", 4, limit=25)
        self.assertEqual(len(samples), 25)
        for s in samples:
            self.assertEqual(len(s.options), 4)
            self.assertEqual(s.options[s.gold_index]["id"], s.answer)
            self.assertNotEqual(s.answer, "unknown")
            self.assertFalse(s.is_unknown)

    def test_gold_position_is_reshuffled_not_inherited(self):
        samples = U.eval_only_samples("logiqa", 4, limit=200)
        positions = {s.gold_index for s in samples}
        self.assertEqual(positions, {0, 1, 2, 3})

    def test_sampling_is_deterministic(self):
        a = U.eval_only_samples("reclor", 4, limit=10)
        b = U.eval_only_samples("reclor", 4, limit=10)
        self.assertEqual([s.question_id for s in a],
                         [s.question_id for s in b])


class TestCrossLingualStatus(unittest.TestCase):
    def _gate(self, holdout: dict) -> dict:
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "gate.json")
            with open(p, "w") as f:
                json.dump({"run_id": "r1", "holdout": holdout}, f)
            from pathlib import Path
            return U.cross_lingual_status(Path(p))

    def test_unmet_when_the_run_carved_no_locales(self):
        rep = self._gate({"massive": {"unseen": ["x"]}})
        self.assertFalse(rep["pass"])
        self.assertIn("all 6 MASSIVE locales", rep["blocker"])
        self.assertEqual(rep["protocol"]["eval_locales"],
                         list(U.EVAL_LOCALES))

    def test_met_when_the_run_carved_exactly_the_eval_locales(self):
        rep = self._gate({"locales": {"unseen": list(U.EVAL_LOCALES)}})
        self.assertTrue(rep["pass"])
        self.assertIsNone(rep["blocker"])

    def test_unmet_when_the_run_carved_a_different_pair(self):
        rep = self._gate({"locales": {"unseen": ["en-US"]}})
        self.assertFalse(rep["pass"])

    def test_missing_training_gate_is_not_a_pass(self):
        from pathlib import Path
        rep = U.cross_lingual_status(Path("/nonexistent/gate.json"))
        self.assertFalse(rep["pass"])


def _cut(n=10, acc=0.5, ece=0.2, brier=0.6):
    m = {"n": n, "accuracy": acc, "accuracy_ci95": [acc - 0.1, acc + 0.1],
         "chance": 0.2, "accuracy_options_only": acc,
         "accuracy_options_only_ci95": [acc - 0.1, acc + 0.1],
         "chance_options_only": 0.25, "ece": ece, "brier": brier,
         "nll": 1.0, "mean_k": 4.0, "abstain_rate": 0.1}
    return {"raw": m, "n": n}


class TestReportShape(unittest.TestCase):
    def test_pair_reports_the_drop_and_the_rise(self):
        pair = U._pair(_cut(acc=0.6, ece=0.1), _cut(acc=0.2, ece=0.4))
        self.assertAlmostEqual(pair["accuracy_drop"], 0.4)
        self.assertAlmostEqual(pair["ece_rise"], 0.3)
        # unseen 0.2 with a CI reaching down to 0.1 does NOT clear the 0.2
        # chance rate: the pair must say so, not round it up
        self.assertFalse(pair["unseen_beats_chance"])
        better = U._pair(_cut(acc=0.6), _cut(acc=0.45))
        self.assertTrue(better["unseen_beats_chance"])

    def test_pair_is_empty_without_both_halves(self):
        self.assertEqual(U._pair(_cut(), {"n": 0}), {})

    def test_headline_carries_both_halves_of_every_cut(self):
        table = {"banking77": {"seen": _cut(), "unseen": _cut(acc=0.1),
                               "pair": U._pair(_cut(), _cut(acc=0.1))}}
        head = U.headline(table)
        self.assertEqual(len(head["cuts"]["banking77"]["accuracy"]), 2)
        self.assertEqual(len(head["cuts"]["banking77"]["n"]), 2)


class TestGateComposition(unittest.TestCase):
    def _compose(self, **over):
        table = {"banking77": {"seen": _cut(), "unseen": _cut(acc=0.1)},
                 "boolq": {"seen": _cut(), "unseen": {"n": 0},
                           "exempt": "two-label pool"}}
        table["banking77"]["pair"] = U._pair(table["banking77"]["seen"],
                                             table["banking77"]["unseen"])
        table["ALL"] = dict(table["banking77"])
        args = {
            "model_version": "mv-1",
            "ckpt_dir": "artifacts/checkpoints/x",
            "table": table, "eval_only": {},
            "absence": {"pass": True},
            "cross": {"pass": True, "protocol": {}, "blocker": None},
            "seals": {"banking77": {
                "shared_groups_metric_vs_calibration": 0}},
            "cal": {"model_version": "mv-1",
                    "predictor": {"name": "pointer-decision-head"},
                    "global": {"temperature": 1.5}},
            "holdout": T.build_holdout(), "n_scored": 42,
        }
        args.update(over)
        return U.compose_gate(**args)

    def test_all_criteria_green_passes(self):
        gate = self._compose()
        self.assertTrue(gate["pass"])
        self.assertEqual(gate["failed_criteria"], [])
        self.assertTrue(gate["verdict"].startswith("PASS"))
        self.assertEqual(gate["task"], "T-unseen-labels")
        self.assertEqual(gate["model_version"], "mv-1")

    def test_missing_model_version_fails_the_gate(self):
        gate = self._compose(model_version="", cal={
            "predictor": {"name": "pointer-decision-head"}, "global": {}})
        self.assertFalse(gate["pass"])
        self.assertIn("metrics_from_trained_checkpoint",
                      gate["failed_criteria"])

    def test_a_parameterless_scorer_fails_the_gate(self):
        gate = self._compose(cal={
            "model_version": "mv-1", "global": {},
            "predictor": {"name": "cosine-char3-softmax"}})
        self.assertFalse(gate["pass"])
        self.assertIn("metrics_from_trained_checkpoint",
                      gate["failed_criteria"])

    def test_leaked_label_fails_the_gate(self):
        gate = self._compose(absence={"pass": False, "n_hits": 3})
        self.assertFalse(gate["pass"])
        self.assertIn("unseen_text_absent_from_train",
                      gate["failed_criteria"])

    def test_unmet_cross_lingual_arm_fails_the_gate_but_keeps_the_table(self):
        gate = self._compose(cross={"pass": False, "protocol": {},
                                    "blocker": "no locale holdout"})
        self.assertFalse(gate["pass"])
        self.assertEqual(gate["failed_criteria"], ["cross_lingual_holdout"])
        self.assertTrue(gate["verdict"].startswith("FAIL"))
        self.assertIn("unseen accuracy", gate["verdict"])
        self.assertIn("banking77", gate["table"])

    def test_a_group_straddling_the_fence_fails_the_gate(self):
        gate = self._compose(seals={"banking77": {
            "shared_groups_metric_vs_calibration": 7}})
        self.assertFalse(gate["pass"])
        self.assertIn("group_split_sealed", gate["failed_criteria"])

    def test_an_unpaired_cut_fails_the_gate(self):
        table = {"huffpost": {"seen": _cut(), "unseen": {"n": 0}}}
        gate = self._compose(table=table)
        self.assertFalse(gate["pass"])
        self.assertIn("paired_seen_unseen_table", gate["failed_criteria"])

    def test_exempt_cut_does_not_fail_the_pairing(self):
        gate = self._compose()
        pairing = gate["criteria"]["paired_seen_unseen_table"]
        self.assertTrue(pairing["pass"])
        self.assertEqual(pairing["cuts"]["boolq"]["exempt"],
                         "two-label pool")


class TestPublishedGateArtifact(unittest.TestCase):
    """The artifact the operator reads. Skipped until the gate has run."""

    def setUp(self):
        if not U.GATE_PATH.exists():
            self.skipTest("gate.json not generated in this checkout")
        self.gate = json.loads(U.GATE_PATH.read_text())

    def test_carries_model_version_and_the_paired_table(self):
        self.assertTrue(self.gate["model_version"])
        self.assertEqual(self.gate["calibration"]["model_version"],
                         self.gate["model_version"])
        for name in ("banking77", "huffpost", "massive", "ALL"):
            cut = self.gate["table"][name]
            self.assertGreater(cut["seen"]["n"], 0, name)
            self.assertGreater(cut["unseen"]["n"], 0, name)
            self.assertIn("ece", cut["seen"]["raw"])
            self.assertIn("unknown", cut["unseen"]["risk_coverage"])

    def test_eval_only_numbers_carry_date_and_model_version(self):
        for name, rep in self.gate["eval_only_results"].items():
            self.assertTrue(rep["model_version"], name)
            self.assertRegex(rep["date"], r"^\d{4}-\d{2}-\d{2}$")
            self.assertIn("firewall", rep["never_in_train"])

    def test_pass_is_the_and_of_the_criteria(self):
        failed = [k for k, v in self.gate["criteria"].items()
                  if not v.get("pass")]
        self.assertEqual(sorted(failed), self.gate["failed_criteria"])
        self.assertEqual(self.gate["pass"], not failed)


if __name__ == "__main__":
    unittest.main()
