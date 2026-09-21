"""Tests for #T-optset-sampler (stdlib unittest).

The three gate checks are called through the SAME functions the gate uses,
so `artifacts/gates/T-optset-sampler/gate.json` cannot claim something this
suite does not check:

1. no emitted sample has an option set equal to the dataset's global label
   space (the HuffPost 41-category bug);
2. the gold sits at a uniform position — chi-square over >= 10k samples;
3. the `unknown` fraction and the hard/easy mix are COUNTED, not assumed.

Run:
    python3 -m unittest data.test_optset -v
"""
from __future__ import annotations

import os
import unittest

from .optset import (
    ALLOWED_DATASETS,
    MIN_CHI2_SAMPLES,
    OptionSetSampler,
    Sample,
    SamplerConfig,
    assert_trainable,
    chi2_sf,
    dataset_path,
    difficulty,
    epoch_shuffle_report,
    firewall_report,
    gold_position_chi2,
    label_pool,
    question_text,
)

CORPUS_READY = all(os.path.exists(dataset_path(d)) for d in ALLOWED_DATASETS)
_SAMPLER: OptionSetSampler | None = None
_SAMPLES: list[Sample] = []


def setUpModule() -> None:
    """One sampler for the whole suite: ~3k rows/dataset -> >10k samples."""
    global _SAMPLER, _SAMPLES
    if not CORPUS_READY:
        return
    _SAMPLER = OptionSetSampler(rows_per_dataset=3000)
    _SAMPLES = list(_SAMPLER.epoch(0))


needs_corpus = unittest.skipUnless(
    CORPUS_READY, "converted P0 corpus missing from artifacts/data-prefetch")


class ConfigAndFirewall(unittest.TestCase):
    """Structure: runs without the corpus."""

    def test_config_rejects_impossible_knobs(self):
        for bad in ({"k_min": 1}, {"k_min": 9, "k_max": 4},
                    {"hard_fraction": 1.5}, {"unknown_fraction": 1.0},
                    {"hard_top_ratio": 0.0}, {"cos_weight": -0.1}):
            with self.assertRaises(ValueError):
                SamplerConfig(**bad)

    def test_quarantined_and_eval_only_corpora_are_refused(self):
        for blocked in ("synth-loop", "logiqa", "reclor"):
            with self.assertRaises(ValueError):
                assert_trainable([blocked])
        report = firewall_report()
        self.assertTrue(report["all_blocked"], report)
        self.assertEqual(report["allowed"], list(ALLOWED_DATASETS))

    def test_allowlist_is_the_clean_p0_corpus(self):
        self.assertEqual(ALLOWED_DATASETS,
                         ("banking77", "massive", "huffpost", "boolq"))
        self.assertEqual(assert_trainable(["banking77"]), ["banking77"])

    def test_sibling_label_beats_a_random_one(self):
        """Hard negatives must actually rank plausible distractors first."""
        self.assertGreater(
            difficulty("card_arrival", "card_delivery_estimate"),
            difficulty("card_arrival", "exchange_rate"))
        self.assertGreater(difficulty("PARENTS", "PARENTING"),
                           difficulty("PARENTS", "WORLDPOST"))

    def test_question_text_follows_the_head_convention(self):
        self.assertEqual(question_text({"id": "q0"}), "q0")
        self.assertEqual(question_text({"id": "q0", "text": "why?"}), "why?")

    def test_chi2_sf_matches_known_values(self):
        self.assertAlmostEqual(chi2_sf(3.841459, 1), 0.05, places=4)
        self.assertAlmostEqual(chi2_sf(11.070498, 5), 0.05, places=4)
        self.assertAlmostEqual(chi2_sf(0.0, 3), 1.0, places=9)


@needs_corpus
class GateCheck1NoGlobalLabelSpace(unittest.TestCase):
    """The HuffPost bug cannot reproduce."""

    def test_no_sample_equals_the_global_label_space(self):
        violations = _SAMPLER.global_space_violations(_SAMPLES)
        self.assertEqual(violations, [], violations[:3])

    def test_k_stays_far_below_the_pool_for_non_binary_datasets(self):
        per = _SAMPLER.composition(_SAMPLES)["per_dataset"]
        for name in ("banking77", "massive", "huffpost"):
            self.assertLess(per[name]["max_k"], per[name]["pool_size"],
                            f"{name}: K reached the global label space")
            self.assertLessEqual(per[name]["max_k"], SamplerConfig().k_max)

    def test_only_a_binary_pool_is_exempt(self):
        exempt = _SAMPLER.binary_exempt_datasets()
        self.assertEqual(exempt, ["boolq"])
        self.assertEqual(len(_SAMPLER.pool_ids["boolq"]), 2)

    def test_k_is_variable_not_fixed(self):
        ks = {s.k for s in _SAMPLES}
        self.assertGreaterEqual(len(ks), 5, sorted(ks))


@needs_corpus
class GateCheck2GoldPositionUniform(unittest.TestCase):
    """No positional prior — the 0.435 LogiQA shortcut cannot exist here."""

    def test_chi_square_over_10k_samples(self):
        report = gold_position_chi2(_SAMPLES)
        self.assertGreaterEqual(report["n_samples"], MIN_CHI2_SAMPLES,
                                "need >= 10k answerable samples")
        self.assertGreater(report["pooled_p"], 0.01, report)
        for k, stats in report["per_k"].items():
            self.assertGreater(stats["p"], 0.01, f"K={k}: {stats}")

    def test_the_test_catches_a_planted_prior(self):
        """Gold always last -> the same check must reject it."""
        rigged = []
        for s in _SAMPLES[:2000]:
            if s.is_unknown or s.k < 2:
                continue
            clone = Sample(dataset=s.dataset, row_id=s.row_id,
                           question_id=s.question_id, state=s.state,
                           question=s.question, options=list(s.options),
                           answer=s.answer, gold_index=s.k - 1)
            rigged.append(clone)
        report = gold_position_chi2(rigged)
        self.assertLess(report["pooled_p"], 1e-6, report)


@needs_corpus
class GateCheck3MixIsCounted(unittest.TestCase):
    """`unknown` and hard/easy are measured by counting, never assumed."""

    def test_unknown_fraction_matches_the_configuration(self):
        comp = _SAMPLER.composition(_SAMPLES)
        counted = sum(1 for s in _SAMPLES if s.is_unknown)
        self.assertEqual(counted,
                         round(comp["pct_unknown"] / 100.0 * len(_SAMPLES)))
        self.assertAlmostEqual(comp["pct_unknown"],
                               100.0 * _SAMPLER.config.unknown_fraction,
                               delta=2.0)

    def test_unknown_rows_really_drop_the_gold(self):
        checked = 0
        for s in _SAMPLES:
            if not s.is_unknown:
                continue
            checked += 1
            self.assertEqual(s.gold_index, s.k)
            self.assertEqual(s.answer, "unknown")
            # the withheld gold must really be absent from the options
            self.assertIsNotNone(s.dropped_gold)
            self.assertNotIn(s.dropped_gold, s.option_ids())
        self.assertGreater(checked, 0)

    def test_gold_is_present_exactly_once_in_answerable_rows(self):
        for s in _SAMPLES[:5000]:
            if s.is_unknown:
                continue
            ids = s.option_ids()
            self.assertEqual(ids.count(s.answer), 1)
            self.assertEqual(ids[s.gold_index], s.answer)
            self.assertEqual(len(set(ids)), len(ids))

    def test_hard_and_easy_counts_add_up_to_the_distractors(self):
        for s in _SAMPLES[:5000]:
            n_distract = s.k - (0 if s.is_unknown else 1)
            self.assertEqual(s.n_hard + s.n_easy, n_distract)

    def test_hard_mix_is_near_target_where_the_pool_allows_it(self):
        per = _SAMPLER.composition(_SAMPLES)["per_dataset"]
        for name in ("banking77", "massive", "huffpost"):
            self.assertAlmostEqual(per[name]["pct_hard_negatives"],
                                   100.0 * _SAMPLER.config.hard_fraction,
                                   delta=6.0, msg=f"{name}: {per[name]}")
        # boolq has exactly one possible distractor: counted, not targeted.
        self.assertEqual(per["boolq"]["pct_hard_negatives"], 100.0)

    def test_hard_distractors_are_more_plausible_than_easy_ones(self):
        index = _SAMPLER.index["banking77"]
        gold = "card_arrival"
        hard, easy = index.buckets(gold)
        worst_hard = min(difficulty(gold, index.text[i]) for i in hard)
        best_easy = max(difficulty(gold, index.text[i]) for i in easy)
        self.assertGreaterEqual(worst_hard, best_easy)


@needs_corpus
class EpochShuffleAndContract(unittest.TestCase):

    def test_option_order_is_reshuffled_every_epoch(self):
        report = epoch_shuffle_report(_SAMPLER, (0, 1), limit=1500)
        self.assertTrue(report["set_stable"], report)
        self.assertGreaterEqual(report["reordered_rate"], 0.5, report)
        self.assertGreater(report["gold_moved"], 0, report)

    def test_epochs_are_reproducible(self):
        a = [s.option_ids() for n, s in enumerate(_SAMPLER.epoch(3)) if n < 200]
        b = [s.option_ids() for n, s in enumerate(_SAMPLER.epoch(3)) if n < 200]
        self.assertEqual(a, b)

    def test_batches_do_not_pad_k(self):
        batch = next(_SAMPLER.batches(epoch=0, batch_size=32))
        self.assertEqual(len(batch), 32)
        self.assertGreater(len({s.k for s in batch}), 1)

    def test_emitted_contract_is_what_the_head_consumes(self):
        for s in _SAMPLES[:500]:
            self.assertTrue(s.state and s.question)
            self.assertGreaterEqual(s.k, 1)
            self.assertTrue(all({"id", "text"} <= set(o) for o in s.options))
            self.assertEqual(s.gold_target(), s.gold_index)
            # the head returns [K + 1] logits; the target must index them
            self.assertLess(s.gold_target(), s.k + 1)

    def test_pool_is_the_full_label_space_of_the_file(self):
        self.assertEqual(len(label_pool("huffpost")), 41)
        self.assertEqual(len(label_pool("boolq")), 2)


if __name__ == "__main__":
    unittest.main()
