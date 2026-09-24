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
import math
import os
import random
import shutil
import unittest
from types import SimpleNamespace

from data import mix
from data.optset import DistractorIndex, Sample, SamplerConfig, UNKNOWN_ID

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


class TestHeadWidthFlag(unittest.TestCase):
    """`--d-model` must reach `train()` (#T-antiscale-diag axis 3)."""

    def _capture(self, extra):
        seen = {}
        real = td.train
        td.train = lambda **kw: (seen.update(kw), {"steps": 0})[1]
        try:
            self.assertEqual(
                td.main(["train_decision", "train", "--run-id",
                         "flag-probe"] + extra), 0)
        finally:
            td.train = real
        return seen

    def test_d_model_flag_reaches_train(self):
        self.assertEqual(self._capture(["--d-model", "512"])["d_model"], 512)

    def test_d_model_defaults_to_the_published_head(self):
        self.assertEqual(self._capture([])["d_model"], td.DEFAULT_D_MODEL)
        self.assertEqual(td.DEFAULT_D_MODEL, 256)


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
        # source at 15 % and brings the rest in at that share. Experimental
        # sources (#T-labelspace-div) are registered but never scanned by
        # default, so the default run is `default_datasets()`, not SOURCES.
        self.assertEqual(run["datasets"], sorted(mix.default_datasets()))
        self.assertTrue(set(mix.SOURCES) - set(run["datasets"])
                        <= {d for d, s in mix.SOURCES.items()
                            if s.experimental})
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

    def test_the_run_and_the_manifest_declare_the_cardinality_regime(self):
        """R4 (#T-eval-cardinality): a checkpoint that does not say what
        cardinality it trained at turns every later comparison into a
        guess — a K<=8 verdict read next to a full-space one."""
        run = self.records("run")[0]
        with open(os.path.join(self.ckpt, "manifest.json")) as fh:
            manifest = json.load(fh)
        for where, doc in (("run.json", run), ("manifest", manifest)):
            with self.subTest(where=where):
                regime = doc["cardinality_regime"]
                self.assertEqual(regime["mode"], "sampled options")
                self.assertEqual((regime["k_min"], regime["k_max"]),
                                 (SamplerConfig().k_min,
                                  SamplerConfig().k_max))
                self.assertIn("denominator",
                              regime["loss_normalised_over"])
                self.assertIn("R4", regime["rule"])
        self.assertEqual(run["cardinality_regime"],
                         manifest["cardinality_regime"])

    def test_the_stage_eval_is_labelled_a_diagnostic_with_its_n_and_k(self):
        """It shares its regime with the objective, so it may not head a
        report — and it says so where it is published (R1/R2)."""
        stage = self.records("stage")[0]
        for side in ("seen", "unseen"):
            with self.subTest(side=side):
                report = stage[side]
                self.assertEqual(report["role"], "diagnostic")
                self.assertIn(f"n={report['n']}", report["why_diagnostic"])
                self.assertIn(f"mean K={report['mean_k']}",
                              report["why_diagnostic"])
                self.assertIsNotNone(report["chance"])
                self.assertIsNotNone(report["accuracy_ci95"])

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


class TestGenObjectiveFlags(unittest.TestCase):
    """The four #T-gen-objective flags must reach `train()` componibly."""

    def _capture(self, extra):
        seen = {}
        real = td.train
        td.train = lambda **kw: (seen.update(kw), {"steps": 0})[1]
        try:
            self.assertEqual(
                td.main(["train_decision", "train", "--run-id",
                         "flag-probe"] + extra), 0)
        finally:
            td.train = real
        return seen

    def test_all_four_flags_reach_train(self):
        seen = self._capture(["--label-dropout", "0.2",
                              "--episodic-resample",
                              "--contrastive-weight", "0.5",
                              "--contrastive-tau", "0.1",
                              "--prior-penalty", "1.0"])
        self.assertEqual(seen["label_dropout"], 0.2)
        self.assertTrue(seen["episodic_resample"])
        self.assertEqual(seen["contrastive_weight"], 0.5)
        self.assertEqual(seen["contrastive_tau"], 0.1)
        self.assertEqual(seen["prior_penalty"], 1.0)

    def test_all_four_default_off(self):
        seen = self._capture([])
        self.assertEqual(seen["label_dropout"], 0.0)
        self.assertFalse(seen["episodic_resample"])
        self.assertEqual(seen["contrastive_weight"], 0.0)
        self.assertEqual(seen["prior_penalty"], 0.0)


class TestGenObjective(unittest.TestCase):
    """Unit tests for the four #T-gen-objective candidates, synthetic."""

    POOL = [{"id": f"l{i}", "text": f"label number {i}"} for i in range(12)]

    def _sampler(self):
        index = DistractorIndex(list(self.POOL), SamplerConfig())
        return SimpleNamespace(index={"d": index},
                               pools={"d": list(self.POOL)}, rows={"d": []})

    def _sample(self, gold="l0", distractors=("l1", "l2", "l3"),
                unknown=False):
        by_id = {o["id"]: o["text"] for o in self.POOL}
        by_id[gold] = f"label number {gold}"
        options = [{"id": i, "text": by_id[i]} for i in distractors]
        if unknown:
            return Sample(dataset="d", row_id="d-0", question_id="q",
                          state="s", question="which?", options=options,
                          answer=UNKNOWN_ID, gold_index=len(options),
                          dropped_gold=gold, n_hard=1, n_easy=2)
        options.append({"id": gold, "text": by_id[gold]})
        return Sample(dataset="d", row_id="d-0", question_id="q",
                      state="s", question="which?", options=options,
                      answer=gold, gold_index=len(options) - 1,
                      n_hard=1, n_easy=2)

    def test_batched_packing_matches_the_row_layout(self):
        """`pack_options`/`batch_gold`/`batch_prior_penalty` vs the row path."""
        import torch
        batch = [self._sample(distractors=("l1", "l2")),
                 self._sample(distractors=("l1", "l2", "l3", "l4")),
                 self._sample(distractors=("l1",), unknown=True)]
        # the flat [question, opt, opt, ...] layout `batch_embeddings` emits
        texts, spans = [], []
        for sample in batch:
            start = len(texts)
            texts.append(td.canonical_question(sample.question))
            texts.extend(o["text"] for o in sample.options)
            spans.append((start, len(texts)))
        embs = torch.arange(len(texts) * 5,
                            dtype=torch.float32).reshape(len(texts), 5)
        q, opts, omask = td.pack_options(embs, spans, torch.device("cpu"))
        kmax = opts.shape[1]
        self.assertEqual(kmax, 5)   # 4 distractors + the gold
        for i, (start, end) in enumerate(spans):
            k = end - start - 1
            self.assertTrue(torch.equal(q[i], embs[start]))
            self.assertTrue(torch.equal(opts[i, :k], embs[start + 1:end]))
            self.assertTrue(bool(omask[i, :k].all()))
            self.assertFalse(bool(omask[i, k:].any()))
        gold = td.batch_gold(batch, kmax, torch.device("cpu"))
        # a real gold keeps its column; `unknown` moves to the shared K_max
        self.assertEqual(gold.tolist(), [2, 4, kmax])
        counts = {("d", "l1"): 100, ("d", "l2"): 10}
        pen = td.batch_prior_penalty(batch, counts, kmax,
                                     torch.device("cpu"), torch.float32)
        for i, sample in enumerate(batch):
            k = len(sample.options)
            row = td.apply_prior_penalty(torch.zeros(k + 1), sample,
                                         counts, 1.0)
            self.assertTrue(torch.allclose(pen[i, :k], -row[:k]))
            self.assertEqual(float(pen[i, k:].abs().sum()), 0.0)

    def test_label_dropout_zero_is_identity(self):
        batch = [self._sample(), self._sample(gold="l4")]
        out, stats = td.apply_label_dropout(
            batch, 0.0, random.Random("zero"))
        self.assertEqual(stats["options_replaced"], 0)
        for before, after in zip(batch, out):
            self.assertEqual([o["text"] for o in after.options],
                             [o["text"] for o in before.options])
            self.assertEqual(after.gold_index, before.gold_index)

    def test_label_dropout_substitutes_never_appear_in_train(self):
        batch = [self._sample(), self._sample(gold="l5")]
        train_texts = {o["text"] for s in batch for o in s.options}
        out, stats = td.apply_label_dropout(
            batch, 1.0, random.Random("full"))
        self.assertGreater(stats["options_replaced"], 0)
        for sample in out:
            for opt in sample.options:
                self.assertIn(td.NOVEL_OPTION_MARK, opt["text"])
                self.assertNotIn(opt["text"], train_texts)
        for before, after in zip(batch, out):
            self.assertEqual(after.gold_index, before.gold_index)
            self.assertEqual(after.answer, before.answer)
            # the input batch is not mutated: the copy carries the swap
            self.assertTrue(all(td.NOVEL_OPTION_MARK not in o["text"]
                                for o in before.options))

    def test_label_dropout_rejects_rates_outside_the_unit(self):
        with self.assertRaises(ValueError):
            td.apply_label_dropout([self._sample()], 1.5,
                                   random.Random(0))
        with self.assertRaises(ValueError):
            td.apply_label_dropout([self._sample()], -0.1,
                                   random.Random(0))

    def test_episodic_resample_keeps_gold_and_varies_distractors(self):
        sampler, sample = self._sampler(), self._sample()
        pool_ids = {o["id"] for o in self.POOL}
        seen_sets = set()
        for episode in range(8):
            out = td.resample_distractors(
                sample, sampler, random.Random(f"ep{episode}"))
            self.assertEqual(len(out.options), len(sample.options))
            self.assertEqual(out.options[out.gold_index]["id"], "l0")
            distract = tuple(sorted(o["id"] for i, o in
                                    enumerate(out.options)
                                    if i != out.gold_index))
            self.assertNotIn("l0", distract)
            self.assertTrue(set(distract) <= pool_ids)
            seen_sets.add(distract)
        self.assertGreater(len(seen_sets), 1)

    def test_episodic_resample_keeps_unknown_rows_unknown(self):
        out = td.resample_distractors(self._sample(unknown=True),
                                      self._sampler(), random.Random(7))
        self.assertEqual(out.answer, UNKNOWN_ID)
        self.assertEqual(out.gold_index, len(out.options))

    def test_fit_label_prior_counts_golds_and_skips_unknown(self):
        sampler = self._sampler()
        sampler.rows = {"d": [
            {"questions": [{"answer": "l0"}, {"answer": "l0"},
                           {"answer": "l1"}]},
            {"questions": [{"answer": UNKNOWN_ID}, {"answer": None},
                           {"answer": "l1"}]}]}
        counts = td.fit_label_prior({"d": sampler})
        self.assertEqual(counts, {("d", "l0"): 2, ("d", "l1"): 2})
        self.assertAlmostEqual(td.prior_penalty_for("d", "l0", counts),
                               math.log(2))
        self.assertEqual(td.prior_penalty_for("d", "l9", counts), 0.0)

    @unittest.skipUnless(td.HAVE_TORCH, "needs torch (never stubbed)")
    def test_contrastive_loss_carries_a_gradient(self):
        torch = td.torch
        q = torch.randn(2, 8, requires_grad=True)
        opts = [torch.randn(3, 8, requires_grad=True),
                torch.randn(4, 8, requires_grad=True)]
        loss = td.contrastive_q_option_loss(q, opts, [0, 2])
        self.assertTrue(math.isfinite(loss.item()))
        loss.backward()
        self.assertIsNotNone(q.grad)
        for o in opts:
            self.assertIsNotNone(o.grad)

    @unittest.skipUnless(td.HAVE_TORCH, "needs torch (never stubbed)")
    def test_contrastive_loss_is_bounded_by_log_k(self):
        """Cosine / tau, not raw dot / tau: the term cannot run away.

        The first contrastive arm of #T-gen-objective diverged (loss 39.2
        at step 1, 53.0 by step 50, against 1.45 on every other arm)
        because the raw projections of a d_model=256 head give dot
        products in the tens. Scaling the inputs by 100 must not change
        the loss at all — that is what normalisation means.
        """
        torch = td.torch
        q = torch.randn(4, 16)
        opts = [torch.randn(5, 16) for _ in range(4)]
        golds = [0, 1, 2, 3]
        loss = td.contrastive_q_option_loss(q, opts, golds, tau=0.07)
        # bounded by the worst case of a 5-way softmax at tau=0.07:
        # sims span at most 2/tau, so the loss cannot exceed that + log K
        self.assertLess(loss.item(), 2.0 / 0.07 + math.log(5))
        scaled = td.contrastive_q_option_loss(
            [x * 100 for x in [q]][0], [o * 100 for o in opts], golds,
            tau=0.07)
        self.assertAlmostEqual(loss.item(), scaled.item(), places=4)

    @unittest.skipUnless(td.HAVE_TORCH, "needs torch (never stubbed)")
    def test_contrastive_loss_rewards_the_aligned_gold(self):
        """A gold option pointing the same way as the question wins."""
        torch = td.torch
        q = torch.tensor([[1.0, 0.0]])
        aligned = td.contrastive_q_option_loss(
            q, [torch.tensor([[1.0, 0.0], [-1.0, 0.0]])], [0])
        opposed = td.contrastive_q_option_loss(
            q, [torch.tensor([[1.0, 0.0], [-1.0, 0.0]])], [1])
        self.assertLess(aligned.item(), opposed.item())

    @unittest.skipUnless(td.HAVE_TORCH, "needs torch (never stubbed)")
    def test_contrastive_loss_skips_unknown_rows(self):
        torch = td.torch
        q = torch.randn(2, 8, requires_grad=True)
        opts = [torch.randn(3, 8), torch.randn(2, 8)]
        loss = td.contrastive_q_option_loss(q, opts, [1, None])
        self.assertTrue(math.isfinite(loss.item()))
        loss.backward()
        self.assertIsNotNone(q.grad)

    @unittest.skipUnless(td.HAVE_TORCH, "needs torch (never stubbed)")
    def test_prior_penalty_discounts_the_frequent_label(self):
        torch = td.torch
        sample = self._sample(gold="l1", distractors=("l0", "l2", "l3"))
        counts = {("d", "l0"): 1000, ("d", "l1"): 1,
                  ("d", "l2"): 1, ("d", "l3"): 1}
        logits = torch.tensor([2.0, 0.0, 0.0, 1.0, -1.0],
                              requires_grad=True)
        self.assertEqual(int(torch.argmax(logits).item()), 0)
        adjusted = td.apply_prior_penalty(logits, sample, counts, 1.0)
        # the frequent distractor loses log(1000); the tail gold wins now
        self.assertEqual(int(torch.argmax(adjusted).item()),
                         sample.gold_index)
        # `unknown` (last) is untouched by the penalty
        self.assertAlmostEqual(float(adjusted[-1]), -1.0)
        adjusted.sum().backward()
        self.assertIsNotNone(logits.grad)

    @unittest.skipUnless(td.HAVE_TORCH, "needs torch (never stubbed)")
    def test_all_four_compose_on_one_synthetic_batch(self):
        torch = td.torch
        sampler = self._sampler()
        rng = random.Random("compose")
        batch = [self._sample(), self._sample(gold="l2")]
        batch = [td.resample_distractors(s, sampler, rng) for s in batch]
        batch, _ = td.apply_label_dropout(batch, 0.5, rng)
        counts = {("d", "l0"): 10, ("d", "l2"): 2}
        total = torch.zeros(())
        for sample in batch:
            logits = torch.randn(len(sample.options) + 1,
                                 requires_grad=True)
            logits = td.apply_prior_penalty(logits, sample, counts, 0.5)
            total = total + torch.nn.functional.cross_entropy(
                logits.unsqueeze(0),
                torch.tensor([sample.gold_index]))
        self.assertTrue(math.isfinite(total.item()))


@unittest.skipUnless(HAVE_STACK, SKIP)
class TestUnfreezeRegimes(unittest.TestCase):
    """#T-unfreeze-backbone: the regime is counted, never just declared."""

    @classmethod
    def setUpClass(cls):
        from model.encoder import load_backbone
        cls.backbone = load_backbone("ettin-68m", "cpu")

    def tearDown(self):
        td.set_backbone_trainable(self.backbone, "none")

    def trainable(self):
        return sum(p.numel() for p in self.backbone.model.parameters()
                   if p.requires_grad)

    def test_none_leaves_the_encoder_shut(self):
        regime = td.set_backbone_trainable(self.backbone, "none")
        self.assertTrue(regime["frozen"])
        self.assertEqual(regime["trainable_params"], 0)
        self.assertEqual(self.trainable(), 0)
        self.assertFalse(self.backbone.model.training)

    def test_full_opens_every_parameter(self):
        regime = td.set_backbone_trainable(self.backbone, "full")
        self.assertFalse(regime["frozen"])
        self.assertEqual(regime["trainable_params"], self.backbone.params)
        self.assertEqual(self.trainable(), self.backbone.params)
        self.assertTrue(self.backbone.model.training)

    def test_last_n_opens_exactly_the_top_blocks(self):
        blocks = td.backbone_layers(self.backbone.model)
        regime = td.set_backbone_trainable(self.backbone, "last-n", 2)
        want = sum(p.numel() for i in (len(blocks) - 2, len(blocks) - 1)
                   for p in blocks[i].parameters())
        want += sum(p.numel()
                    for p in self.backbone.model.final_norm.parameters())
        self.assertEqual(regime["trainable_params"], want)
        self.assertEqual(self.trainable(), want)
        self.assertLess(regime["trainable_params"], self.backbone.params)
        self.assertEqual(regime["opened"],
                         [f"layers.{len(blocks) - 2}",
                          f"layers.{len(blocks) - 1}", "final_norm"])

    def test_an_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            td.set_backbone_trainable(self.backbone, "top")


@unittest.skipUnless(HAVE_STACK, SKIP)
class TestUnfrozenRun(unittest.TestCase):
    """A tiny `last-n` run: the encoder must move, and say so on disk."""

    RUN = "test-train-unfrozen"

    @classmethod
    def setUpClass(cls):
        cls.summary = td.train(max_samples=64, batch_size=32,
                               rows_per_dataset=60, eval_samples=32,
                               log_every=1, run_id=cls.RUN, device="cpu",
                               backbone_id="ettin-68m", write_gate=False,
                               unfreeze="last-n", unfreeze_layers=1,
                               backbone_lr=1e-4)
        cls.ckpt = os.path.join(td.ROOT, cls.summary["checkpoint"])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(os.path.join(td.RUNS_DIR, cls.RUN), ignore_errors=True)
        shutil.rmtree(os.path.join(td.CKPT_DIR, cls.RUN), ignore_errors=True)

    def manifest(self):
        with open(os.path.join(self.ckpt, "manifest.json")) as fh:
            return json.load(fh)

    def test_the_manifest_declares_the_regime_it_really_trained(self):
        bb = self.manifest()["backbone"]
        self.assertFalse(bb["frozen"])
        self.assertEqual(bb["unfreeze"]["mode"], "last-n")
        self.assertEqual(bb["trainable_params"],
                         bb["state_dict_trainable_params"])
        self.assertGreater(bb["state_dict_trainable_params"], 0)
        self.assertLess(bb["state_dict_trainable_params"],
                        bb["state_dict_params"])

    def test_the_trained_encoder_ships_with_the_checkpoint(self):
        bb = self.manifest()["backbone"]
        path = os.path.join(self.ckpt, bb["weights_file"])
        self.assertTrue(os.path.exists(path))
        self.assertEqual(td.sha256_file(path), bb["weights_sha256"])

    def test_the_cold_load_restores_the_trained_encoder(self):
        from model.encoder import load_backbone
        engine, manifest = td.load_checkpoint(self.ckpt, "cpu")
        pristine = load_backbone(manifest["backbone"]["id"], "cpu")
        moved = {k for k, v in engine.backbone.model.state_dict().items()
                 if not td.torch.equal(
                     v, pristine.model.state_dict()[k])}
        self.assertTrue(moved, "the backbone_lr step left no trace")
        self.assertTrue(all(k.startswith("layers.18.")
                            or k.startswith("final_norm") for k in moved),
                        f"last-n touched blocks it did not open: {moved}")

    def test_the_cost_of_the_regime_is_published(self):
        cost = self.summary["cost"]
        self.assertFalse(cost["backbone"]["frozen"])
        self.assertGreater(cost["minutes_per_1m"], 0)


class TestInBatchNegativesArm(unittest.TestCase):
    """El brazo muestreado de #T-fullspace-objective, extremo a extremo.

    Lo que un test de unidad de `training.python.inbatch` no puede fijar:
    que la bandera llegue a `train()`, que el batch extendido sobreviva a
    `pack_options`/`batch_gold`/la penalización de prior, que la pérdida
    salga finita y que el régimen quede DECLARADO en el manifiesto — que es
    la regla R1 escrita en un fichero y no en una intención.
    """

    RUN = "test-train-inbatch"

    @classmethod
    def setUpClass(cls):
        cls.summary = td.train(max_samples=64, batch_size=16,
                               rows_per_dataset=60, eval_samples=32,
                               log_every=1, run_id=cls.RUN, device="cpu",
                               write_gate=False, prior_penalty=1.0,
                               in_batch_negatives=True,
                               inbatch_calib_batches=4)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(os.path.join(td.RUNS_DIR, cls.RUN), ignore_errors=True)
        shutil.rmtree(os.path.join(td.CKPT_DIR, cls.RUN), ignore_errors=True)

    def _regime(self):
        with open(os.path.join(td.RUNS_DIR, self.RUN, "run.json")) as fh:
            return json.load(fh)["cardinality_regime"]

    def test_the_run_finished_with_a_finite_objective(self):
        """El NLL del re-score de entreno: si la pérdida hubiera dado NaN
        (`-inf - inf` en una columna de relleno) esto no sería finito."""
        train = self.summary["train"]
        self.assertTrue(math.isfinite(train["nll"]))
        self.assertGreater(train["n"], 0)

    def test_the_denominator_actually_grew(self):
        """`mean_k` por encima del K del sampler: las columnas ajenas están."""
        self.assertGreater(self.summary["train"]["mean_k"], 8)

    def test_the_regime_is_declared_in_run_json(self):
        reg = self._regime()
        self.assertIn("in_batch_negatives", reg)
        ib = reg["in_batch_negatives"]
        self.assertTrue(ib["correction"]["pi"].startswith("pi(c) = 1 -"))
        self.assertIn("mixed batches", ib["loader"])
        self.assertIn("set_attention=True", ib["architecture"])
        self.assertEqual(reg["mode"], ib["mode"])

    def test_the_calibration_is_recorded(self):
        corr = self._regime()["in_batch_negatives"]["correction"]
        self.assertGreater(corr["calibration_rows"], 0)
        self.assertGreater(corr["calibrated_labels"], 0)

    def test_the_default_path_declares_no_in_batch_block(self):
        """Sin la bandera, el manifiesto es el de antes."""
        reg = td.cardinality_regime(SamplerConfig(seed=1))
        self.assertNotIn("in_batch_negatives", reg)
        self.assertEqual(reg["mode"], "sampled options")


if __name__ == "__main__":
    unittest.main()
