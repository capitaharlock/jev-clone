"""Unit tests for eval.scoreboard — the single table (#T-eval-cardinality).

No checkpoint is loaded: every section is composed from rows and scored
entries handed in directly, because what this module has to get right is the
COMPOSITION — which number is primary, which is labelled diagnostic, what
travels beside each one, and whether the three collapse diagnostics say what
they claim to say.
"""
import io
import json
import unittest
from contextlib import redirect_stderr

from data.optset import Sample

from . import cuts as K
from . import scoreboard as SB

SPACE = [{"id": f"intent_{i}", "text": f"intent {i}"} for i in range(5)]


def rows(golds: list) -> list:
    return [Sample(dataset="banking77", row_id=f"banking77-{i}",
                   question_id="q", state=f"row {i}", question="which intent",
                   options=list(SPACE), answer=SPACE[g]["id"], gold_index=g)
            for i, g in enumerate(golds)]


def scored(preds: list, k: int = 5) -> list:
    out = []
    for i, pred in enumerate(preds):
        probs = [0.02] * (k + 1)
        probs[pred] = 0.9
        out.append({"id": f"banking77:banking77-{i}:q", "probs": probs,
                    "pred": pred, "cardinality": k})
    return out


class TestPredictionFrequency(unittest.TestCase):
    def test_a_collapsed_head_is_called_collapsed(self):
        freq = SB.prediction_frequency(rows([0, 1, 2, 3, 4] * 4),
                                       scored([1] * 20))
        self.assertEqual(freq["distinct_predicted"], 1)
        self.assertEqual(freq["top1_label"], "intent_1")
        self.assertEqual(freq["top1_share"], 1.0)
        self.assertIn("collapsed", freq["reading"])
        self.assertEqual(freq["n_never_predicted"], 4)

    def test_a_spread_head_is_not_called_collapsed(self):
        freq = SB.prediction_frequency(rows([0, 1, 2, 3, 4] * 4),
                                       scored([i % 5 for i in range(20)]))
        self.assertEqual(freq["distinct_predicted"], 5)
        self.assertIn("spread", freq["reading"])
        self.assertNotIn("collapsed", freq["reading"])

    def test_a_head_that_reaches_part_of_the_space_is_called_skewed(self):
        """Neither a constant nor an even spread: the case the dev cut
        actually shows, 26 of 77 labels with one taking a third of them."""
        space = [{"id": f"intent_{i}", "text": f"intent {i}"}
                 for i in range(20)]
        golds = [i % 20 for i in range(20)]
        preds = [0] * 6 + list(range(1, 8)) + [1] * 7
        rows20 = [Sample(dataset="banking77", row_id=f"banking77-{i}",
                         question_id="q", state=f"row {i}", question="q",
                         options=list(space), answer=space[g]["id"],
                         gold_index=g) for i, g in enumerate(golds)]
        freq = SB.prediction_frequency(rows20, scored(preds, k=20))
        self.assertIn("skewed", freq["reading"])
        self.assertNotIn("collapsed", freq["reading"])
        self.assertLess(freq["top1_share"], 0.5)
        self.assertLessEqual(freq["distinct_predicted"], 10)

    def test_abstentions_are_a_label_of_their_own(self):
        freq = SB.prediction_frequency(rows([0] * 4), scored([5, 5, 5, 0]))
        self.assertEqual(freq["top1_label"], "⟨unknown⟩")
        self.assertEqual(freq["predicted"][0]["count"], 3)

    def test_the_gold_count_travels_with_the_prediction_count(self):
        """"Predicted 400 times" means nothing without "it is the gold 40
        times": that difference IS the bias."""
        freq = SB.prediction_frequency(rows([0] * 10), scored([1] * 10))
        row = freq["predicted"][0]
        self.assertEqual((row["label"], row["count"], row["gold_count"]),
                         ("intent_1", 10, 0))


class TestConfusion(unittest.TestCase):
    def test_the_matrix_counts_gold_against_prediction(self):
        conf = SB.confusion(rows([0, 0, 1]), scored([1, 1, 1]))
        self.assertEqual(conf["cells"]["intent_0"], {"intent_1": 2})
        self.assertEqual(conf["cells"]["intent_1"], {"intent_1": 1})
        self.assertEqual(conf["n_off_diagonal"], 2)

    def test_the_dominant_pair_is_named(self):
        conf = SB.confusion(rows([0, 0, 0, 2]), scored([1, 1, 1, 3]))
        top = conf["top_confusions"][0]
        self.assertEqual((top["gold"], top["predicted"], top["count"]),
                         ("intent_0", "intent_1", 3))

    def test_a_perfect_run_has_no_off_diagonal_mass(self):
        conf = SB.confusion(rows([0, 1, 2]), scored([0, 1, 2]))
        self.assertEqual(conf["top_confusions"], [])
        self.assertEqual(conf["n_off_diagonal"], 0)


class TestErrorSample(unittest.TestCase):
    def test_only_errors_are_sampled_and_they_carry_the_gold_rank(self):
        sample = SB.error_sample(rows([0, 1, 2]), scored([0, 2, 2]))
        self.assertEqual(len(sample), 1)
        self.assertEqual(sample[0]["gold"], "intent_1")
        self.assertEqual(sample[0]["predicted"], "intent_2")
        self.assertGreater(sample[0]["gold_rank"], 1)

    def test_a_clean_cut_samples_nothing(self):
        self.assertEqual(SB.error_sample(rows([0, 1]), scored([0, 1])), [])

    def test_the_sample_is_spread_over_the_cut_not_its_first_rows(self):
        board = SB.error_sample(rows([0] * 60), scored([1] * 60), n=3)
        self.assertEqual(len(board), 3)
        self.assertNotEqual([e["id"] for e in board],
                            [f"banking77:banking77-{i}:q" for i in range(3)])


class TestProtocolAndRegime(unittest.TestCase):
    def test_the_examples_protocol_says_the_model_got_identifiers(self):
        proto = SB.examples_protocol(rows([0]))
        self.assertEqual(proto["labelled_examples_in_state"], 0)
        self.assertIn("identifier", proto["option_text"])

    def test_an_undeclared_regime_is_not_guessed(self):
        regime = SB.training_regime({"loss": "listwise CE"}, {})
        self.assertFalse(regime["declared"])
        self.assertIn("not inferred", regime["why"])

    def test_a_declared_regime_is_published_as_declared(self):
        declared = {"k_min": 3, "k_max": 8, "mode": "sampled options"}
        regime = SB.training_regime({"cardinality_regime": declared}, {})
        self.assertTrue(regime["declared"])
        self.assertEqual(regime["k_max"], 8)

    def test_a_fenced_dataset_is_called_unseen_with_its_evidence(self):
        run = {"fence": {"clean_1m": True, "fenced_datasets": ["banking77"]},
               "datasets": ["huffpost", "massive"]}
        exposure = SB.label_exposure({}, run, "banking77")
        self.assertEqual(exposure["claim"], "unseen")
        self.assertIn("fenced_datasets", exposure["evidence"])
        self.assertTrue(exposure["not_verified"])

    def test_a_trained_dataset_is_called_seen(self):
        run = {"fence": {"fenced_datasets": []},
               "datasets": ["banking77", "huffpost"]}
        self.assertEqual(SB.label_exposure({}, run, "banking77")["claim"],
                         "seen")


class TestComposition(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "weights_sha256": "f" * 64, "model_version": "jev-dec-test",
            "tokenizer_hash": "e" * 64,
            "metrics": {"seen": {"n": 3008, "mean_k": 4.37,
                                 "accuracy": 0.62, "chance": 0.2067,
                                 "accuracy_ci95": [0.60, 0.64]},
                        "unseen": {"n": 3008, "mean_k": 5.1,
                                   "accuracy": 0.09, "chance": 0.19,
                                   "accuracy_ci95": [0.08, 0.10]}},
        }
        self.run = {"fence": {"clean_1m": True,
                              "fenced_datasets": ["banking77"]},
                    "datasets": ["huffpost"],
                    "data_manifest": {"banking77": {"sha256": "d" * 64}}}
        self.seal = {"split_sha256": "a" * 64, "manifest_sha256": "b" * 64,
                     "manifest": "artifacts/splits/x/manifest.json"}

    def primary(self):
        return SB.primary_section(
            K.DEV, rows([0, 1, 2, 3, 4]), scored([0, 1, 4, 4, 4]), self.seal,
            self.manifest, self.run, "ckpt/stage-1",
            SB.cost_section(1.0, 5, 5, "cpu"))

    def test_the_primary_carries_every_column_the_task_asks_for(self):
        p = self.primary()
        for field in ("checkpoint_sha256", "split_sha256", "data_sha256",
                      "cardinality", "n", "hits", "chance", "accuracy_ci95",
                      "abstain_rate", "examples_protocol", "seen_or_unseen",
                      "cost", "training_regime"):
            with self.subTest(field=field):
                self.assertIn(field, p)
        self.assertEqual((p["n"], p["hits"], p["cardinality"]), (5, 3, 5))

    def test_the_primary_says_it_is_the_primary(self):
        self.assertTrue(self.primary()["role"].startswith("PRIMARY"))

    def test_every_diagnostic_is_labelled_a_diagnostic_with_its_n_and_k(self):
        d = SB.diagnostics_section(self.manifest)
        self.assertIn("DIAGNOSTIC", d["role"])
        self.assertEqual(d["eval_unseen_gate"]["role"], "diagnostic")
        self.assertEqual(d["trainer_stage_eval"]["role"], "diagnostic")
        self.assertIn("diagnostic", d["cardinality_sweep"]["role"])
        for side in ("seen", "unseen"):
            block = d["trainer_stage_eval"][side]
            self.assertEqual(block["n"], 3008)
            self.assertIsNotNone(block["mean_k"])
            self.assertIsNotNone(block["chance"])

    def test_the_unseen_gate_is_read_off_the_published_artifact(self):
        d = SB.diagnostics_section(self.manifest)["eval_unseen_gate"]
        self.assertTrue(d["artifact"].endswith("T-unseen-labels/gate.json"))
        self.assertIsNotNone(d["unseen"]["mean_k"])
        self.assertIsNotNone(d["unseen"]["chance"])

    def test_the_sweep_is_published_with_counts_and_no_progression(self):
        d = SB.diagnostics_section(self.manifest)["cardinality_sweep"]
        self.assertTrue(d["points"])
        for k, point in d["points"].items():
            with self.subTest(k=k):
                self.assertIsNotNone(point["chance"])
                self.assertIsNotNone(point["accuracy_ci95"])
        self.assertIn("no narrative", d["no_progression"])

    def test_parity_is_a_citation_and_says_what_differs(self):
        parity = SB.parity_section("banking77")
        self.assertFalse(parity["same_rows"])
        self.assertTrue(all(r["citation"] for r in parity["references"]))
        self.assertTrue(parity["differences_declared"])
        self.assertIn("#T-jev-parity", parity["measured_by"])

    def test_same_rows_false_is_not_read_as_different_rows(self):
        """#T-jev-parity: the section used to assert a fact nobody has."""
        reading = SB.parity_section("banking77")["same_rows_reading"]
        self.assertFalse(reading["verified"])
        self.assertIn("NOT ESTABLISHED", reading["value_means"])
        self.assertIn("nobody has compared them", reading["value_means"])
        self.assertTrue(reading["what_is_NOT_verified"])

    def test_the_parity_gate_is_attached_and_never_borrowed(self):
        """done-when 6: the number enters the report by itself, and a
        report for a checkpoint with no parity read says so instead of
        quoting somebody else's."""
        attached = SB.parity_section("banking77", "no/such/checkpoint")
        self.assertFalse(attached["jev_parity"]["measured"])
        self.assertIn("eval.parity gate", attached["jev_parity"]["how"])
        self.assertEqual(attached["jev_parity"]["baseline"]["accuracy"],
                         0.012338)

    def test_cost_warns_that_it_does_not_scale_linearly_in_k(self):
        cost = SB.cost_section(12.5, 1000, 77, "mps")
        self.assertEqual(cost["ms_per_row"], 12.5)
        self.assertIn("K²", cost["note"])


class TestTheReservedCutIsGuarded(unittest.TestCase):
    def test_reading_the_test_cut_without_a_reason_is_refused(self):
        with self.assertRaises(SystemExit) as cm, \
                redirect_stderr(io.StringIO()):
            SB.main(["show", "--checkpoint", "x", "--cut", "test"])
        self.assertEqual(cm.exception.code, 2)

    def test_the_dev_cut_is_the_default(self):
        self.assertEqual(K.CUTS["dev"].name, "banking77-dev")
        self.assertFalse(K.CUTS["dev"].reserved)


class TestRender(unittest.TestCase):
    def board(self) -> dict:
        p = SB.primary_section(
            K.DEV, rows([0, 1, 2, 3, 4]), scored([0, 1, 4, 4, 4]),
            {"split_sha256": "a" * 64}, {"weights_sha256": "f" * 64},
            {"fence": {"fenced_datasets": ["banking77"]}}, "ckpt/stage-1",
            SB.cost_section(1.0, 5, 5, "cpu"))
        return {
            "checkpoint": "ckpt/stage-1", "model_version": "jev-dec-test",
            "generated_utc": "2026-09-24T00:00:00Z", "primary": p,
            "diagnostics": SB.diagnostics_section({}),
            "parity": SB.parity_section("banking77"),
            "cost": SB.cost_section(1.0, 5, 5, "cpu"),
            "training_regime": p["training_regime"],
            "prediction_frequency": SB.prediction_frequency(
                rows([0, 1, 2, 3, 4]), scored([0, 1, 4, 4, 4])),
            "confusion": SB.confusion(rows([0, 1, 2, 3, 4]),
                                      scored([0, 1, 4, 4, 4])),
            "errors": SB.error_sample(rows([0, 1, 2, 3, 4]),
                                      scored([0, 1, 4, 4, 4])),
            "cuts": K.report(),
            "reconciliation": ".meshkore/docs/cortes-de-evaluacion.md",
        }

    def test_the_table_leads_with_the_primary_and_labels_the_rest(self):
        text = SB.render(self.board())
        self.assertLess(text.index("PRIMARY"), text.index("DIAGNOSTICS"))
        self.assertLess(text.index("DIAGNOSTICS"), text.index("PARITY"))
        for section in ("COST", "COLLAPSE", "CONFUSION", "ERRORS"):
            self.assertIn(section, text)

    def test_the_primary_row_prints_its_chance_and_its_interval(self):
        text = SB.render(self.board())
        self.assertIn("chance", text)
        self.assertIn("0.2000", text)
        self.assertIn("[0.", text)

    def test_an_undeclared_regime_is_printed_as_undeclared(self):
        self.assertIn("NOT DECLARED", SB.render(self.board()))

    def test_the_reserved_cut_is_named_as_reserved(self):
        self.assertIn("RESERVED", SB.render(self.board()))


class TestPublishedScoreboard(unittest.TestCase):
    """What the command actually wrote to disk."""

    def test_the_artifact_exists_and_leads_with_the_primary(self):
        self.assertTrue(SB.GATE_PATH.exists(),
                        "run `python -m eval.scoreboard show --checkpoint …`")
        board = json.loads(SB.GATE_PATH.read_text())
        self.assertEqual(board["hierarchy"][0], "primary")
        p = board["primary"]
        self.assertEqual(p["cardinality"], 77)
        for field in ("chance", "accuracy_ci95", "hits", "n",
                      "abstain_rate", "split_sha256"):
            with self.subTest(field=field):
                self.assertIsNotNone(p[field])

    def test_the_published_artifact_passes_the_coherence_rules(self):
        """C4 applies to this artifact like to any other."""
        from . import gate_rules as G
        r = G.check_gate_dir(SB.GATE_DIR, G.FALLBACK_COHERENCE)
        self.assertTrue(r["pass"], r["errors"])
        self.assertGreater(r["accuracy_claims"]["published"], 0)


if __name__ == "__main__":
    unittest.main()
