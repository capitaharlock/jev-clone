"""Tests for #T-train-real (stdlib unittest).

Two layers, the same rule as `model/test_decision_head.py`:

* structure — runs without torch: the holdout algebra, the metric
  functions and the question canonicalisation;
* behaviour — needs torch + the sha256-verified backbone, and is SKIPPED
  (never faked) when they are absent. It trains a real (tiny) run and
  checks the properties the gate claims: batched encoding equals the
  single-row encoder, no held-out label text reaches training, the
  checkpoint reloads cold and answers inside the latency budget, and the
  same seed reproduces the same losses.

Run:
    .venv-train/bin/python -m unittest training.python.test_train_decision -v
"""
from __future__ import annotations

import json
import os
import shutil
import unittest

from data import mix
from data.optset import SamplerConfig

from . import train_decision as td

HAVE_STACK = td.HAVE_TORCH and os.path.exists(
    os.path.join(td.PREFETCH_DIR, "banking77.jsonl"))
SKIP = "real stack or converted corpus absent (never stubbed)"


class TestMetrics(unittest.TestCase):
    def test_canonical_question_strips_the_row_index(self):
        self.assertEqual(td.canonical_question("banking77-intent-8412"),
                         "banking77-intent")
        self.assertEqual(td.canonical_question("massive-intent-0"),
                         "massive-intent")

    def test_canonical_question_is_stable_for_prose(self):
        self.assertEqual(td.canonical_question("huffpost-cat"), "huffpost-cat")
        self.assertEqual(td.canonical_question("what is the gate state?"),
                         "what is the gate state?")
        self.assertEqual(td.canonical_question(""), "")

    def test_wilson_interval_brackets_the_estimate(self):
        lo, hi = td.wilson_interval(50, 100)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        # A small n must not produce a confident interval.
        lo_small, hi_small = td.wilson_interval(2, 4)
        self.assertGreater(hi_small - lo_small, hi - lo)

    def test_ece_is_zero_when_calibrated_and_large_when_not(self):
        calibrated = [(1.0, True)] * 50 + [(0.0, False)] * 50
        self.assertLess(td.expected_calibration_error(calibrated), 1e-9)
        overconfident = [(1.0, False)] * 50
        self.assertGreater(td.expected_calibration_error(overconfident), 0.9)

    def test_brier_is_zero_on_a_perfect_one_hot(self):
        self.assertAlmostEqual(td.brier_score([0.0, 1.0, 0.0], 1), 0.0)
        self.assertAlmostEqual(td.brier_score([1.0, 0.0, 0.0], 1), 2.0)

    def test_normalise_label_compares_texts_not_ids(self):
        self.assertEqual(td.normalise_label("ARTS & CULTURE"), "arts culture")
        self.assertEqual(td.normalise_label("card_arrival"), "card arrival")


class TestHoldout(unittest.TestCase):
    POOL = [{"id": f"l{i}", "text": t} for i, t in enumerate([
        "card_arrival", "card_delivery_estimate", "exchange_rate",
        "exchange_charge", "lost_or_stolen_card", "lost_or_stolen_phone",
        "pending_transfer", "failed_transfer", "atm_support",
        "country_support", "top_up_reverted", "verify_identity"])]

    def test_sibling_holdout_takes_pairs_and_respects_the_quota(self):
        seen, unseen = td.sibling_holdout(self.POOL, 0.5)
        self.assertEqual(len(unseen), 6)
        self.assertEqual(len(seen), 6)
        self.assertFalse(set(seen) & set(unseen))
        text = {o["id"]: o["text"] for o in self.POOL}
        held = {text[i] for i in unseen}
        # The rule is "hardest pairs first": at least one obvious sibling
        # pair must have been taken whole.
        pairs = [("exchange_rate", "exchange_charge"),
                 ("lost_or_stolen_card", "lost_or_stolen_phone"),
                 ("card_arrival", "card_delivery_estimate")]
        self.assertTrue(any(a in held and b in held for a, b in pairs),
                        f"no sibling pair held out: {sorted(held)}")

    def test_sibling_holdout_is_deterministic(self):
        self.assertEqual(td.sibling_holdout(self.POOL, 0.5),
                         td.sibling_holdout(self.POOL, 0.5))

    def test_a_tiny_pool_holds_nothing_out(self):
        seen, unseen = td.sibling_holdout(
            [{"id": "yes", "text": "yes"}, {"id": "no", "text": "no"}], 0.5)
        self.assertEqual(unseen, [])
        self.assertEqual(seen, ["no", "yes"])

    @unittest.skipUnless(HAVE_STACK, SKIP)
    def test_real_holdout_matches_the_unseen_labels_protocol(self):
        holdout = td.build_holdout()
        self.assertEqual(len(holdout["huffpost"].unseen), 11)
        self.assertEqual(len(holdout["huffpost"].seen), 30)
        self.assertEqual(holdout["boolq"].unseen, [])
        self.assertIn("two-label pool", holdout["boolq"].exempt)
        for name in ("banking77", "massive", "huffpost"):
            h = holdout[name]
            self.assertFalse(set(h.seen) & set(h.unseen))
            self.assertGreaterEqual(len(h.unseen), 2)


@unittest.skipUnless(HAVE_STACK, SKIP)
class TestBatchedEncoding(unittest.TestCase):
    """The batched encoder must BE `model.encoder.encode_state`."""

    @classmethod
    def setUpClass(cls):
        from model.encoder import load_backbone
        cls.backbone = load_backbone("ettin-68m", "cpu")

    def test_batched_encoding_matches_the_single_row_encoder(self):
        import torch

        from model.encoder import encode_state
        states = ["I am still waiting on my card?",
                  "the harbor gate is open and the ledger is balanced",
                  "a much longer passage " * 20]
        tokens, mask, n_tok = td.encode_states(self.backbone, states, 256)
        self.assertEqual(tokens.shape[0], len(states))
        self.assertEqual(tokens.shape[1] % td.PAD_MULTIPLE, 0)
        total = 0
        for i, state in enumerate(states):
            one = encode_state(self.backbone, state, 256)
            t = int(one["n_tokens"])
            total += t
            self.assertEqual(int(mask[i].sum()), t)
            diff = (tokens[i, :t] - one["tokens"][0, :t]).abs().max()
            self.assertLess(float(diff), 2e-3,
                            f"row {i} diverges from encode_state")
        self.assertEqual(n_tok, total)
        self.assertFalse(torch.is_inference(tokens))


@unittest.skipUnless(HAVE_STACK, SKIP)
class TestTrainingRun(unittest.TestCase):
    """A real, tiny, end-to-end run — no gate write, no placeholders."""

    RUN = "test-train-real"
    ROWS = 120
    SAMPLES = 256

    @classmethod
    def setUpClass(cls):
        cls.summary = td.train(max_samples=cls.SAMPLES, batch_size=32,
                               rows_per_dataset=cls.ROWS, eval_samples=120,
                               log_every=1, run_id=cls.RUN, device="cpu",
                               write_gate=False)
        cls.run_dir = os.path.join(td.RUNS_DIR, cls.RUN)
        cls.ckpt = os.path.join(td.ROOT, cls.summary["checkpoint"])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.run_dir, ignore_errors=True)
        shutil.rmtree(os.path.join(td.CKPT_DIR, cls.RUN), ignore_errors=True)
        shutil.rmtree(os.path.join(td.CKPT_DIR, cls.RUN + "-b"),
                      ignore_errors=True)
        shutil.rmtree(os.path.join(td.RUNS_DIR, cls.RUN + "-b"),
                      ignore_errors=True)

    def records(self, kind):
        with open(os.path.join(self.run_dir, "metrics.jsonl")) as fh:
            return [r for r in (json.loads(x) for x in fh)
                    if r.get("t") == kind]

    def test_metrics_jsonl_carries_brier_and_ece_from_step_one(self):
        steps = self.records("step")
        self.assertGreaterEqual(len(steps), 4)
        self.assertEqual(steps[0]["step"], 1)
        for key in ("loss", "accuracy", "brier", "ece", "tokens", "lr"):
            self.assertIn(key, steps[0], f"step 1 has no {key}")
        self.assertGreater(steps[0]["tokens"], 0)

    def test_one_multi_dataset_model_not_one_pickle_per_dataset(self):
        run = self.records("run")[0]
        # the mixture, not the P0 four: #T-corpus-rebalance caps every
        # source at 15 % and brings the rest in at that share
        self.assertEqual(run["datasets"], sorted(mix.SOURCES))
        self.assertTrue(run["mix"]["pass"], run["mix"])
        self.assertGreater(run["mix"]["dynamic_option_share"], 0.5)
        stages = self.records("stage")
        self.assertEqual(len(stages), 1, "one checkpoint, not one per dataset")
        self.assertTrue(os.path.exists(
            os.path.join(self.ckpt, "model.safetensors")))
        self.assertTrue(os.path.exists(
            os.path.join(self.ckpt, "tokenizer.json")))

    def test_manifest_carries_the_runtime_cache_key_components(self):
        with open(os.path.join(self.ckpt, "manifest.json")) as fh:
            manifest = json.load(fh)
        self.assertTrue(manifest["model_version"].startswith("jev-dec-"))
        self.assertEqual(len(manifest["tokenizer_hash"]), 64)
        self.assertEqual(len(manifest["weights_sha256"]), 64)
        self.assertIn("CacheKey", manifest["runtime_cache_key"])
        # the loop stops at the first batch boundary past max_samples, and
        # a source whose quota is smaller than the batch ends short
        self.assertGreaterEqual(manifest["samples_seen"], self.SAMPLES)
        self.assertLess(manifest["samples_seen"], self.SAMPLES + 32)
        self.assertGreater(manifest["tokens_seen"], 0)

    def test_the_head_stays_label_free_after_training(self):
        engine, _ = td.load_checkpoint(self.ckpt, "cpu")
        report = engine.head.label_free_report()
        self.assertTrue(report["label_free"], report["offenders"])

    def test_checkpoint_reloads_cold_and_answers_inside_the_budget(self):
        config = SamplerConfig(seed=1789)
        holdout = td.build_holdout()
        samplers = td.train_samplers(holdout, config, self.ROWS)
        probe = td.probe_samples(samplers, 12)
        engine, _ = td.load_checkpoint(self.ckpt, "cpu")
        latency = td.measure_decision_latency(engine, probe, n=12)
        self.assertLess(latency["p95_ms"], td.LATENCY_BUDGET_MS)
        result = engine.score(probe[0].state,
                              td.canonical_question(probe[0].question),
                              probe[0].options)
        self.assertEqual(len(result["probs"]) + 1, probe[0].k + 1)

    def test_no_held_out_label_text_reaches_training(self):
        holdout = td.build_holdout()
        samplers = td.train_samplers(holdout, SamplerConfig(seed=1789),
                                     self.ROWS)
        report = td.holdout_cleanliness(holdout, samplers, 2000)
        self.assertTrue(report["pass"], report["hits"])
        self.assertGreater(report["checked_samples"], 0)
        self.assertGreater(report["banned_texts"], 40)

    def test_same_seed_and_manifest_reproduce_the_same_losses(self):
        again = td.train(max_samples=self.SAMPLES, batch_size=32,
                         rows_per_dataset=self.ROWS, eval_samples=120,
                         log_every=1, run_id=self.RUN + "-b", device="cpu",
                         write_gate=False)
        with open(os.path.join(td.RUNS_DIR, self.RUN + "-b",
                               "metrics.jsonl")) as fh:
            other = [r for r in (json.loads(x) for x in fh)
                     if r.get("t") == "step"]
        mine = self.records("step")
        self.assertEqual([r["loss"] for r in mine],
                         [r["loss"] for r in other])
        self.assertEqual(self.summary["tokens_seen"], again["tokens_seen"])

    def test_a_budget_larger_than_the_mixture_aborts(self):
        """#T-mix-1m: repetition is never the silent fallback."""
        with self.assertRaises(td.MixShortfallError) as ctx:
            td.train(max_samples=1_000_000, batch_size=32,
                     rows_per_dataset=self.ROWS, eval_samples=120,
                     log_every=1, run_id=self.RUN + "-short", device="cpu",
                     write_gate=False)
        message = str(ctx.exception)
        self.assertIn("epochs over the same corpus", message)
        self.assertIn("--allow-repeat", message)
        shutil.rmtree(os.path.join(td.RUNS_DIR, self.RUN + "-short"),
                      ignore_errors=True)

    def test_allow_repeat_publishes_the_epochs_it_took(self):
        """Repetition is allowed out loud, and then it is a number."""
        run = self.RUN + "-rep"
        budget = self.SAMPLES * 8
        try:
            summary = td.train(max_samples=budget, batch_size=32,
                               rows_per_dataset=20, eval_samples=120,
                               log_every=8, run_id=run, device="cpu",
                               write_gate=False, allow_repeat=True,
                               stages=(budget,))
            run_dir = os.path.join(td.RUNS_DIR, run)
            with open(os.path.join(run_dir, "run.json")) as fh:
                record = json.load(fh)
            with open(os.path.join(run_dir, "mix.json")) as fh:
                mix_manifest = json.load(fh)
            self.assertTrue(record["allow_repeat"])
            self.assertGreater(record["epochs_over_corpus"], 1.0)
            self.assertFalse(record["budget"]["covers_budget"])
            self.assertEqual(mix_manifest["budget"]["epochs_over_corpus"],
                             record["epochs_over_corpus"])
            # the metrics log carries it too: line 0 is what a reader of a
            # published curve checks before trusting its x axis
            with open(os.path.join(run_dir, "metrics.jsonl")) as fh:
                line0 = json.loads(fh.readline())
            self.assertEqual(line0["epochs_over_corpus"],
                             record["epochs_over_corpus"])
            self.assertGreater(summary["epochs_run"], 1)
        finally:
            shutil.rmtree(os.path.join(td.RUNS_DIR, run), ignore_errors=True)
            shutil.rmtree(os.path.join(td.CKPT_DIR, run), ignore_errors=True)

    def test_a_covered_budget_reports_one_pass(self):
        run = self.records("run")[0]
        self.assertTrue(run["budget"]["covers_budget"])
        self.assertFalse(run["allow_repeat"])
        self.assertLessEqual(run["epochs_over_corpus"], 1.0)
        self.assertGreaterEqual(run["epoch_samples"], run["max_samples"])

    def test_the_mixture_obeys_the_corpus_guardrails(self):
        """#T-corpus-rebalance, enforced where the batches are built."""
        with open(os.path.join(self.run_dir, "mix.json")) as fh:
            manifest = json.load(fh)
        composition = manifest["composition"]
        for dataset, n in composition["by_dataset"].items():
            self.assertLessEqual(n / composition["rows"],
                                 mix.MAX_DATASET_FRACTION + 1e-9, dataset)
        for family, n in composition["by_family"].items():
            self.assertLessEqual(n / composition["rows"],
                                 mix.MAX_FAMILY_FRACTION + 1e-9, family)
        self.assertGreater(composition["dynamic_option_share"], 0.5)
        self.assertEqual(len(manifest["members_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
