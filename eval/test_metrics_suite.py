"""Unit tests for eval.metrics_suite — #T-battery-metrics.

Everything here is synthetic: rows are built by hand so a rule can be shown
to fire without a checkpoint, a GPU or a corpus. The four the task names
explicitly:

* a verdict missing any mandatory metric, or the chance of its K, or the n
  of its cut, makes the runner FAIL — error, not warning;
* a `reading` claiming "contains chance" while its interval excludes chance
  makes the runner FAIL;
* joint counterfactual success is computed by `variant_group` and is 0 when
  only one half of the pair is right;
* permutation invariance and state/question tracking are two properties and
  the report never lets one stand in for the other.
"""
import json
import unittest

from . import metrics_suite as M

CUT = {"name": "synthetic-dev", "reserved": False, "split": "train"}
CAL = {"temperature": 1.3, "fitted_on": "dev", "verified_on": "test",
       "threshold": 0.4, "strategy": "maxprob"}
PERM = {"max_abs_prob_delta": 4.7e-07, "tolerance": 1e-05, "n_perms": 24,
        "n_rows": 64, "pass": True, "source": "T-option-text 2026-09-24"}
TRACK = {"n": 64, "followed": 9, "rate": 0.140625, "pass": False,
         "source": "T-option-text 2026-09-24"}


def row(rid, family="f1", group=None, k=4, gold=0, pred=0, probs=None):
    probs = probs or ([0.7] + [0.1] * (k - 1) + [0.05])
    return {"row_id": rid, "family": family,
            "variant_group": group or f"g-{rid}", "k": k,
            "gold_index": gold, "pred": pred, "probs": probs}


def full(rows, **kw):
    kw.setdefault("calibration", CAL)
    kw.setdefault("permutation", PERM)
    kw.setdefault("tracking", TRACK)
    return M.report(rows, cut=dict(CUT), model_version="test-mv", **kw)


def rules(errors) -> set:
    return {e["rule"] for e in errors}


def missing(errors) -> set:
    return {e.get("missing") for e in errors}


class ShapeTest(unittest.TestCase):
    def test_a_complete_report_passes_the_runner(self):
        rep = full([row("a", group="p1", pred=0), row("b", group="p1", pred=0),
                    row("c", family="f2", pred=1)])
        self.assertEqual(M.check(rep), [])
        self.assertEqual(M.require(rep), rep)

    def test_every_figure_carries_n_k_chance_and_interval(self):
        rep = full([row("a"), row("b", family="f2")])
        for where in (rep["ranking"], rep["abstention"], rep["counterfactual"],
                      rep["by_family"]["f1"]["ranking"]):
            for key in M.FIGURE_KEYS:
                self.assertIn(key, where, key)
            self.assertEqual(len(where["accuracy_ci95"]), 2)

    def test_the_two_accuracies_are_published_apart_and_together(self):
        # 87.7 % abstention on a head whose ranking is at chance: the pair is
        # the whole point — the 0.0 alone reads as prudence.
        rows = [row(f"r{i}", pred=4) for i in range(877)]
        rows += [row(f"s{i}", pred=1) for i in range(123)]
        rep = full(rows)
        self.assertEqual(rep["abstention"]["accuracy"], 0.0)
        self.assertEqual(rep["abstention"]["abstain_rate"], 0.877)
        self.assertEqual(rep["ranking"]["accuracy"], 1.0)
        self.assertEqual(rep["abstention"]["coverage"]["coverage"], 0.123)
        self.assertEqual(
            rep["abstention"]["precision_among_answered"]["accuracy"], 0.0)
        self.assertEqual(M.check(rep), [])

    def test_chance_is_written_by_k_and_for_the_cut(self):
        rep = full([row("a", k=4), row("b", k=8, family="f2")])
        self.assertEqual(set(rep["chance"]["by_k"]), {"4", "8"})
        self.assertEqual(rep["chance"]["by_k"]["8"]["chance"], 0.125)
        self.assertEqual(rep["chance"]["chance"], round((0.25 + 0.125) / 2, 6))
        self.assertEqual(rep["chance"]["mean_k"], 6.0)


class MandatoryMetricTest(unittest.TestCase):
    """The task's first gate test: anything missing FAILS the runner."""

    def test_a_missing_section_fails_the_runner(self):
        for section in M.MANDATORY_SECTIONS:
            rep = full([row("a")])
            del rep[section]
            errors = M.check(rep)
            self.assertIn("C5", rules(errors), section)
            self.assertIn(section, missing(errors), section)
            with self.assertRaises(M.IncoherentReport):
                M.require(rep)

    def test_a_figure_without_its_chance_fails_the_runner(self):
        rep = full([row("a")])
        rep["ranking"]["chance"] = None
        self.assertIn("chance", missing(M.check(rep)))

    def test_a_cut_without_its_n_fails_the_runner(self):
        rep = full([row("a")])
        rep["cut"]["n"] = None
        self.assertIn("n", missing(M.check(rep)))

    def test_a_figure_without_its_interval_fails_the_runner(self):
        rep = full([row("a")])
        rep["abstention"]["accuracy_ci95"] = 0.5
        self.assertIn("accuracy_ci95", missing(M.check(rep)))

    def test_coverage_and_precision_travel_together(self):
        rep = full([row("a")])
        del rep["abstention"]["coverage"]
        self.assertIn("coverage", missing(M.check(rep)))

    def test_a_calibration_verified_on_its_own_fit_is_not_one(self):
        rep = full([row("a")], calibration={"temperature": 1.0,
                                            "fitted_on": "dev",
                                            "verified_on": "dev"})
        self.assertIn("verification", missing(M.check(rep)))

    def test_an_uncalibrated_report_fails_the_runner(self):
        rep = M.report([row("a")], cut=dict(CUT), permutation=PERM,
                       tracking=TRACK)
        self.assertIn("fitted_on/verified_on", missing(M.check(rep)))

    def test_nll_and_brier_and_ece_are_published_together(self):
        rep = full([row("a")])
        for key in ("nll", "brier", "ece"):
            self.assertIsNotNone(rep["calibration"][key], key)
        broken = full([row("a")])
        broken["calibration"]["brier"] = None
        self.assertIn("brier", missing(M.check(broken)))

    def test_softmax_weights_may_not_be_renamed_into_truth(self):
        rep = full([row("a")])
        rep["calibration"]["p_true"] = 0.7
        errors = M.check(rep)
        self.assertIn("C5", rules(errors))
        self.assertTrue(any("absolute probability" in e["why"]
                            for e in errors))


class MacroByFamilyTest(unittest.TestCase):
    def test_macro_sees_the_family_a_global_mean_hides(self):
        rows = [row(f"a{i}", family="big", pred=0) for i in range(90)]
        rows += [row(f"b{i}", family="small", pred=1) for i in range(10)]
        rep = full(rows)
        self.assertEqual(rep["abstention"]["accuracy"], 0.9)
        self.assertEqual(rep["by_family"]["small"]["abstention"]["accuracy"],
                         0.0)
        self.assertEqual(rep["macro"]["accuracy"], 0.5)
        self.assertEqual(rep["macro"]["worst_family"], "small")
        self.assertEqual(rep["macro"]["n_families"], 2)

    def test_an_empty_family_breakdown_fails_the_runner(self):
        rep = full([row("a")])
        rep["by_family"] = {}
        self.assertIn("families", missing(M.check(rep)))


class CounterfactualTest(unittest.TestCase):
    """The task's third gate test: one half right is a zero."""

    def test_one_half_of_a_pair_right_scores_zero(self):
        rep = full([row("base", group="p1", pred=0),
                    row("variant", group="p1", pred=2)])
        cf = rep["counterfactual"]
        self.assertEqual(cf["n"], 1)
        self.assertEqual(cf["hits"], 0)
        self.assertEqual(cf["accuracy"], 0.0)
        self.assertEqual(cf["groups_with_one_half_only"], 1)
        # and the per-row accuracy is still 0.5: the pair is what disagrees
        self.assertEqual(rep["abstention"]["accuracy"], 0.5)

    def test_both_halves_right_scores_one(self):
        cf = full([row("base", group="p1", pred=0),
                   row("variant", group="p1", pred=0)])["counterfactual"]
        self.assertEqual((cf["hits"], cf["accuracy"]), (1, 1.0))

    def test_chance_of_a_pair_is_the_product_of_its_halves(self):
        cf = full([row("base", group="p1", k=4, pred=0),
                   row("variant", group="p1", k=4, pred=3)])["counterfactual"]
        self.assertEqual(cf["chance"], 0.0625)

    def test_unpaired_groups_are_counted_not_scored(self):
        cf = full([row("lonely", group="solo", pred=0),
                   row("base", group="p1", pred=0),
                   row("variant", group="p1", pred=0)])["counterfactual"]
        self.assertEqual((cf["n"], cf["unpaired_groups"]), (1, 1))

    def test_an_abstention_on_one_half_also_scores_zero(self):
        cf = full([row("base", group="p1", pred=0),
                   row("variant", group="p1", pred=4)])["counterfactual"]
        self.assertEqual(cf["hits"], 0)


class ControlsAreDistinctTest(unittest.TestCase):
    def test_permutation_and_tracking_are_two_properties(self):
        rep = full([row("a")])
        perm, track = rep["permutation_invariance"], rep["tracking"]
        self.assertTrue(perm["pass"])
        self.assertFalse(track["pass"])
        self.assertEqual(perm["distinct_from"], M.TRACKING_WHAT)
        self.assertEqual(track["distinct_from"], M.PERMUTATION_WHAT)
        self.assertIn("never stands in for one", rep["controls_are_distinct"])
        self.assertEqual(M.check(rep), [])

    def test_an_uninvented_control_is_absent_and_fails(self):
        rep = full([row("a")], permutation=None)
        self.assertFalse(rep["permutation_invariance"]["measured"])
        self.assertIn("permutation_invariance", missing(M.check(rep)))

    def test_a_missing_tracking_control_fails_on_its_own(self):
        rep = full([row("a")], tracking=None)
        self.assertIn("tracking", missing(M.check(rep)))


class ReadingCoherenceTest(unittest.TestCase):
    """The task's second gate test: the text must match the numbers."""

    FIG = {"accuracy": 0.0, "accuracy_ci95": [0.0, 0.003827],
           "chance": 0.012987, "cardinality": 77, "n": 1000,
           "accuracy_options_only": 0.009,
           "accuracy_options_only_ci95": [0.004742, 0.017016]}

    def test_contains_chance_while_the_interval_excludes_it_fails(self):
        doc = {"primary": {"arm": dict(self.FIG)},
               "verdict": {"reading": "the primary interval contains chance"}}
        errors = M.reading_errors(doc)
        self.assertEqual([e["rule"] for e in errors], ["C6"])
        self.assertEqual(errors[0]["claim"], "contains_chance")
        self.assertEqual(errors[0]["accuracy_ci95"], [0.0, 0.003827])

    def test_the_same_sentence_about_the_forced_interval_passes(self):
        doc = {"primary": {"arm": dict(self.FIG)},
               "verdict": {"reading": "the forced interval, `unknown` taken "
                                      "out of the race, contains chance"}}
        self.assertEqual(M.reading_errors(doc), [])

    def test_below_chance_that_is_really_below_chance_passes(self):
        doc = {"primary": {"arm": dict(self.FIG)},
               "verdict": {"reading": "the primary interval excludes chance "
                                      "from below"}}
        self.assertEqual(M.reading_errors(doc), [])

    def test_beats_chance_while_the_interval_contains_it_fails(self):
        doc = {"headline": {"accuracy": 0.009, "chance": 0.012987, "n": 1000,
                            "cardinality": 77,
                            "accuracy_ci95": [0.004742, 0.017016],
                            "reading": "this arm beats chance"}}
        errors = M.reading_errors(doc)
        self.assertEqual(errors[0]["claim"], "above_chance")

    def test_a_negated_claim_is_read_as_its_complement(self):
        node = {"accuracy": 0.009, "chance": 0.012987, "n": 1000,
                "cardinality": 77, "accuracy_ci95": [0.004742, 0.017016]}
        ok = dict(node, reading="this arm does not beat chance")
        self.assertEqual(M.reading_errors({"h": ok}), [])
        bad = dict(node, reading="this arm does not contain chance")
        self.assertEqual(M.reading_errors({"h": bad})[0]["claim"],
                         "contains_chance")

    def test_a_cause_discarded_beside_an_r9_disclaimer_fails(self):
        doc = {"verdict": {
            "r9": "a NO-GO at this budget LIMITS SPEND AND DOES NOT "
                  "ESTABLISH CAUSE",
            "reading": "the CARDINALITY OF THE OBJECTIVE is discarded as the "
                       "cause of the transfer failure"}}
        errors = M.reading_errors(doc)
        self.assertEqual(errors[0]["claim"], "cause_discarded")

    def test_prose_about_another_seed_does_not_lift_the_r9_bar(self):
        # every one of these artifacts already SAYS a causal claim would
        # need another seed; reading that sentence as evidence of one is how
        # the claim got published.
        doc = {"verdict": {
            "r9": "does not establish cause; any causal claim needs the "
                  "winning arm repeated on another seed",
            "reading": "the objective is discarded as the cause"}}
        self.assertEqual(M.reading_errors(doc)[0]["claim"], "cause_discarded")

    def test_a_declared_replication_lifts_the_r9_bar(self):
        doc = {"verdict": {
            "r9": "does not establish cause on its own", "seeds": 3,
            "reading": "the objective is discarded as the cause"}}
        self.assertEqual(M.reading_errors(doc), [])

    def test_the_observed_benefit_wording_is_not_a_causal_claim(self):
        doc = {"verdict": {
            "r9": "does not establish cause",
            "reading": "what this NO-GO eliminates is the OBSERVED BENEFIT "
                       "of that arm at one seed and one budget"}}
        self.assertEqual(M.reading_errors(doc), [])

    def test_a_claim_with_no_numbers_to_check_is_left_alone(self):
        self.assertEqual(
            M.reading_errors({"verdict": {"reading": "it beats chance"}}), [])

    def test_check_carries_the_reading_rule(self):
        rep = full([row("a")])
        rep["verdict"] = {"reading": "the ranking beats chance"}
        rep["ranking"]["accuracy_ci95"] = [0.0, 0.5]
        self.assertIn("C6", rules(M.check(rep)))


class RowShapeTest(unittest.TestCase):
    def test_a_row_without_a_family_or_a_group_is_refused(self):
        for field in ("family", "variant_group", "k", "probs"):
            raw = row("a")
            del raw[field]
            with self.assertRaises(M.IncoherentReport):
                M.report([raw], cut=dict(CUT))

    def test_the_weight_vector_must_match_k(self):
        with self.assertRaises(M.IncoherentReport):
            M.report([row("a", k=4, probs=[0.5, 0.5])], cut=dict(CUT))

    def test_an_unanswerable_row_is_right_only_when_abstained(self):
        rows = [dict(row("u", pred=4), gold_index=None),
                dict(row("v", pred=0), gold_index=None)]
        rep = full(rows)
        self.assertEqual(rep["abstention"]["hits"], 1)
        self.assertEqual(rep["abstention"]["unanswerable_rows"]["n"], 2)
        # and they are out of the forced figure: nothing correct to rank
        self.assertEqual(rep["ranking"]["n"], 0)
        self.assertEqual(rep["ranking"]["excluded_unanswerable_rows"], 2)


class CliTest(unittest.TestCase):
    def test_check_exits_non_zero_on_an_incomplete_report(self):
        import tempfile
        rep = full([row("a")])
        del rep["counterfactual"]
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as fh:
            json.dump(rep, fh)
            path = fh.name
        self.assertEqual(M.main(["check", path]), 1)


if __name__ == "__main__":
    unittest.main()
