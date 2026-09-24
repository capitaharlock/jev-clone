"""Unit tests for eval.fullspace — `beats_chance` and the K sweep.

Two defects of #T-eval-cardinality are nailed down here, both of them in the
*reading* of numbers rather than in the numbers themselves:

* `beats_chance` used to be decided with `flo`, the lower bound of the FORCED
  ranking (`accuracy_options_only`, `unknown` taken out of the race). A head
  that abstained on every row would publish `accuracy: 0.0` next to
  `beats_chance: true`. The artifact on disk has `abstain_rate: 0`, so no
  published verdict changes — the field was simply wrong about what it meant.
* `sweep_reading()` used to say "the head clears chance up to K=40", which
  describes a progression. The intervals do not carry one: K=5 and K=20 do
  not clear their chance rate while K=8 and K=40 do.

No model is loaded: the scorer is stubbed, so every case here is about the
arithmetic and the wording, which is what was broken.
"""
import unittest
from unittest import mock

from data.optset import Sample

from . import fullspace as F


def samples(n: int, k: int) -> list:
    """`n` rows of `k` options each, gold always first."""
    options = [{"id": f"l{i}", "text": f"label {i}"} for i in range(k)]
    return [Sample(dataset="banking77", row_id=f"r{i}", question_id="q",
                   state="state", question="question", options=list(options),
                   answer="l0", gold_index=0) for i in range(n)]


def entries(preds: list, k: int) -> list:
    """One scored entry per prediction index — `k` means `unknown`."""
    out = []
    for pred in preds:
        probs = [0.1] * (k + 1)
        probs[pred] = 0.9
        out.append({"probs": probs, "pred": pred, "cardinality": k})
    return out


def score_with(preds: list, k: int) -> dict:
    rows = samples(len(preds), k)
    with mock.patch("eval.calib.entries_from_samples",
                    return_value=entries(preds, k)):
        return F.score(object(), rows)


class TestBeatsChance(unittest.TestCase):
    def test_total_abstention_does_not_beat_chance(self):
        """The case the task names: every row abstains, the ranking is
        perfect underneath, and the published accuracy is 0."""
        rep = score_with([5] * 200, 5)
        self.assertEqual(rep["abstain_rate"], 1.0)
        self.assertEqual(rep["accuracy"], 0.0)
        self.assertEqual(rep["hits"], 0)
        self.assertFalse(rep["beats_chance"])
        # the ranking underneath is perfect and says so, separately
        self.assertEqual(rep["accuracy_options_only"], 1.0)
        self.assertTrue(rep["ranking_beats_chance"])

    def test_the_two_fields_are_decided_by_their_own_interval(self):
        rep = score_with([0] * 200, 5)
        self.assertTrue(rep["beats_chance"])
        self.assertTrue(rep["ranking_beats_chance"])
        self.assertEqual(rep["beats_chance_decided_by"],
                         "accuracy_ci95[0] > chance")

    def test_a_head_at_chance_claims_nothing(self):
        rep = score_with([i % 5 for i in range(200)], 5)
        self.assertEqual(rep["accuracy"], 0.2)
        self.assertEqual(rep["chance"], 0.2)
        self.assertFalse(rep["beats_chance"])
        self.assertFalse(rep["ranking_beats_chance"])

    def test_the_counts_are_published_beside_the_rates(self):
        """A rate with no numerator cannot be re-derived or pooled."""
        rep = score_with([0, 0, 1, 2, 5], 5)
        self.assertEqual(rep["n"], 5)
        self.assertEqual(rep["hits"], 2)
        self.assertEqual(rep["hits_options_only"], 3)
        self.assertEqual(rep["cardinality"], 5)
        self.assertEqual(rep["abstain_rate"], 0.2)

    def test_partial_abstention_splits_the_two_verdicts(self):
        """90 % abstention over a perfect ranking: the ranking clears
        chance, the number we publish does not."""
        preds = [5] * 180 + [0] * 20
        rep = score_with(preds, 5)
        self.assertEqual(rep["accuracy"], 0.1)
        self.assertFalse(rep["beats_chance"])
        self.assertTrue(rep["ranking_beats_chance"])


class TestSweepReading(unittest.TestCase):
    #: the shape the published artifact has: K=5 and K=20 do not clear
    PUBLISHED = {
        "5": {"n": 1000, "hits": 215, "accuracy": 0.215, "chance": 0.2,
              "accuracy_ci95": [0.190653, 0.241528], "beats_chance": False},
        "8": {"n": 1000, "hits": 151, "accuracy": 0.151, "chance": 0.125,
              "accuracy_ci95": [0.130146, 0.174525], "beats_chance": True},
        "20": {"n": 1000, "hits": 61, "accuracy": 0.061, "chance": 0.05,
               "accuracy_ci95": [0.04778, 0.07758], "beats_chance": False},
        "40": {"n": 1000, "hits": 36, "accuracy": 0.036, "chance": 0.025,
               "accuracy_ci95": [0.026116, 0.049436], "beats_chance": True},
        "77": {"n": 1000, "hits": 11, "accuracy": 0.011, "chance": 0.012987,
               "accuracy_ci95": [0.006153, 0.019589], "beats_chance": False},
    }

    def test_the_published_sweep_is_read_as_out_of_order(self):
        reading = F.sweep_reading(self.PUBLISHED)
        self.assertIn("OUT OF ORDER", reading)
        self.assertIn("[8, 40]", reading)
        self.assertIn("[5, 20, 77]", reading)

    def test_no_progression_is_claimed(self):
        """The sentence the external review struck out must not come back."""
        reading = F.sweep_reading(self.PUBLISHED).lower()
        for phrase in ("up to k=40", "holds", "still clears chance at k",
                       "stops clearing"):
            self.assertNotIn(phrase, reading)

    def test_the_counts_and_the_chance_rates_travel_with_it(self):
        reading = F.sweep_reading(self.PUBLISHED)
        self.assertIn("215/1000", reading)
        self.assertIn("chance 0.2", reading)
        self.assertIn("CI [0.190653, 0.241528]", reading)

    def test_a_monotone_sweep_is_not_called_out_of_order(self):
        sweep = {k: dict(v) for k, v in self.PUBLISHED.items()}
        sweep["5"]["beats_chance"] = True
        sweep["20"]["beats_chance"] = True
        sweep["77"]["beats_chance"] = False
        reading = F.sweep_reading(sweep)
        self.assertNotIn("OUT OF ORDER", reading)
        self.assertIn("[5, 8, 20, 40]", reading)

    def test_no_cardinality_clearing_is_said_plainly(self):
        sweep = {k: dict(v, beats_chance=False)
                 for k, v in self.PUBLISHED.items()}
        self.assertIn("no cardinality", F.sweep_reading(sweep))

    def test_an_empty_sweep_reads_as_nothing(self):
        self.assertIn("not enough points", F.sweep_reading({}))


class TestReferencesAreCitations(unittest.TestCase):
    def test_every_reference_declares_itself_a_citation(self):
        """C4 of `eval.gate_rules` exempts citations; the exemption has to
        be claimed in the artifact, not inferred by the reader."""
        for dataset, refs in F.REFERENCES.items():
            for ref in refs:
                with self.subTest(dataset=dataset, who=ref["who"]):
                    self.assertTrue(ref["citation"])
                    self.assertTrue(ref["source"])
                    self.assertTrue(ref["caveat"])


if __name__ == "__main__":
    unittest.main()
