"""Tests del contrato del scorer compartido (#T-ce-scorer, unittest).

Dos capas, la misma disciplina que `model/test_torch_stack.py`:

* el CONTRATO — permutación, seguimiento del texto y contexto comparativo
  en las dos direcciones — contra `HashPairScorer`, sin pesos y sin red,
  para que un checkout limpio ejecute la suite entera;
* los pesos REALES, saltados cuando los checkpoints no están
  materializados, nunca falseados.

El contrato es lo que este módulo promete, y por eso se prueba con un
scorer de pares que es función hash del par completo: si `z` dependiera de
algo que el contrato no declara (la posición, el id opaco), el hash no lo
escondería.
"""
from __future__ import annotations

import os
import unittest

from . import ce_scorer as CE
from . import weights

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:  # pragma: no cover - env sin el venv de entreno
    HAVE_TORCH = False

REAL_SCORERS = [s for s in weights.SCORERS
                if os.path.exists(weights.manifest_path(s))]


def _scorer() -> CE.CrossEncoderScorer:
    return CE.CrossEncoderScorer(CE.HashPairScorer())


CASE = CE.CONTRACT_CASE


class TestContractRegistry(unittest.TestCase):
    def test_five_families_and_a_context_rule_for_each(self):
        self.assertEqual(len(CE.FAMILIES), 5)
        self.assertEqual(set(CE.COMPARATIVE_CONTEXT), set(CE.FAMILIES))

    def test_hypothesis_format_is_fixed_for_both_languages(self):
        for lang in CE.LANGS:
            self.assertIn("{option}", CE.HYPOTHESIS[lang])
            self.assertIn("{state}", CE.PREMISE[lang])
            self.assertIn("{question}", CE.PREMISE[lang])

    def test_scorer_registry_pins_commit_shas(self):
        for sid, spec in weights.SCORERS.items():
            self.assertRegex(spec["revision"], r"^[0-9a-f]{40}$",
                             f"{sid} must pin a commit sha, not a branch")
            self.assertIn("config.json", spec["files"])
            self.assertTrue(any(f.endswith((".bin", ".safetensors"))
                                for f in spec["files"]))

    def test_registries_do_not_overlap(self):
        # `tools/smoke_torch` y `model.bakeoff` recorren BACKBONES: un id
        # en las dos listas les metería un cabezal NLI por la puerta de
        # atrás.
        self.assertEqual(set(weights.BACKBONES) & set(weights.SCORERS), set())
        for sid in weights.SCORERS:
            self.assertIs(weights.spec_for(sid), weights.SCORERS[sid])

    def test_unknown_family_is_refused(self):
        with self.assertRaises(CE.ScorerContractError):
            CE.Decision(state="s", question="q", family="made_up",
                        candidates=(CE.Candidate("a", "x"),
                                    CE.Candidate("b", "y")))


class TestRendering(unittest.TestCase):
    def test_premise_carries_state_question_and_the_option_is_the_hypothesis(
            self):
        premise, hypothesis = CE.render_pair(CASE, CASE.candidates[0])
        self.assertIn(CASE.state, premise)
        self.assertIn(CASE.question, premise)
        self.assertEqual(
            hypothesis,
            CE.HYPOTHESIS["en"].format(option=CASE.candidates[0].text))

    def test_comparative_context_block_is_identical_across_the_k_passes(self):
        premises = {p for p, _ in CE.render_pairs(CASE)}
        self.assertEqual(len(premises), 1, "the K premises must be one string")

    def test_context_block_holds_every_candidate_and_no_id(self):
        premise, _ = CE.render_pair(CASE, CASE.candidates[0])
        for cand in CASE.candidates:
            self.assertIn(cand.text, premise)
            self.assertNotIn(f"- {cand.id}", premise)

    def test_context_off_never_mentions_the_others(self):
        off = CE.Decision(state=CASE.state, question=CASE.question,
                          candidates=CASE.candidates, family=CE.EXTRACTION)
        premise, _ = CE.render_pair(off, off.candidates[0])
        self.assertNotIn(off.candidates[1].text, premise)
        self.assertNotIn(CE.CONTEXT_HEADER["en"], premise)

    def test_family_decides_the_context_unless_overridden(self):
        self.assertTrue(CASE.wants_context())          # attribute_comparison
        self.assertFalse(CE.Decision(
            state="s", question="q", family=CE.EXTRACTION,
            candidates=CASE.candidates).wants_context())
        self.assertFalse(CE.Decision(
            state="s", question="q", family=CE.COMPARISON,
            candidates=CASE.candidates,
            comparative_context=False).wants_context())


class TestGateChecks(unittest.TestCase):
    """Las tres comprobaciones que el `Verification gate` nombra."""

    def test_permutation_does_not_move_the_probabilities(self):
        got = CE.check_permutation_invariance(_scorer(), CASE)
        self.assertTrue(got["pass"], got)
        self.assertLessEqual(got["max_abs_prob_delta"], CE.PERMUTATION_TOL)
        self.assertEqual(got["n_perms"], 6)

    def test_swapping_texts_keeping_ids_swaps_the_scores(self):
        got = CE.check_text_follows_id(_scorer(), CASE)
        self.assertTrue(got["pass"], got)
        self.assertGreater(got["base_separation"], CE.PERMUTATION_TOL,
                           "the two candidates must start apart or the "
                           "check cannot tell 'follows text' from 'nothing "
                           "happened'")

    def test_comparative_context_both_directions(self):
        got = CE.check_comparative_context(
            _scorer(), CASE, CE.CONTRACT_FOREIGN_ID, CE.CONTRACT_FOREIGN_TEXT)
        self.assertTrue(got["pass"], got)
        self.assertGreater(got["on"]["abs_delta"], CE.PERMUTATION_TOL)
        self.assertEqual(got["off"]["abs_delta"], 0.0)

    def test_a_positional_scorer_would_fail_the_text_check(self):
        """El test que hace que el anterior signifique algo.

        Un scorer que puntuara por POSICIÓN pasa la permutación (realinear
        lo tapa) pero no puede pasar el seguimiento del texto. Si
        `check_text_follows_id` lo aprobara, no estaría midiendo nada.
        """

        class Positional(CE.PairScorer):
            id = "positional"

            def score_pairs(self, pairs):
                return [float(i % 3) for i in range(len(pairs))]

        got = CE.check_text_follows_id(
            CE.CrossEncoderScorer(Positional()), CASE)
        self.assertFalse(got["pass"])

    def test_contract_report_passes_and_is_writable(self):
        report = CE.run_contract(write=False)
        self.assertTrue(report["pass"], report["checks"])
        self.assertTrue(report["scalar_output"]["label_free"])


class TestScoring(unittest.TestCase):
    def test_probabilities_are_a_softmax_over_the_k_candidates(self):
        got = _scorer().score(CASE)
        self.assertEqual(got["k"], 3)
        self.assertAlmostEqual(sum(got["probs"]), 1.0, places=9)
        self.assertEqual(got["argmax_id"], got["ranking"][0])
        self.assertEqual(len(got["z"]), 3)

    def test_no_unknown_column_is_invented(self):
        # el softmax es sobre las opciones OFRECIDAS: la abstención es otra
        # task (#T-battery-calib), no un apaño dentro del scorer
        got = _scorer().score(CASE)
        self.assertEqual(len(got["probs"]), CASE.k)
        self.assertNotIn("unknown", got["candidate_ids"])

    def test_score_many_batches_every_pair_in_one_call(self):
        ps = CE.HashPairScorer()
        scorer = CE.CrossEncoderScorer(ps)
        decs = [CASE, CASE, CASE]
        out = scorer.score_many(decs)
        self.assertEqual(len(out), 3)
        self.assertEqual(ps.calls, 1, "the K passes of every question must "
                                      "share one batched call")
        self.assertEqual(out[0]["z"], out[1]["z"])

    def test_gold_gives_correct_and_nll(self):
        got = _scorer().score(CASE)
        self.assertIsInstance(got["correct"], bool)
        self.assertGreater(got["nll"], 0.0)

    def test_k_below_two_is_refused(self):
        with self.assertRaises(ValueError):
            CE.Decision(state="s", question="q",
                        candidates=(CE.Candidate("a", "x"),))

    def test_duplicate_ids_are_refused(self):
        with self.assertRaises(ValueError):
            CE.Decision(state="s", question="q",
                        candidates=(CE.Candidate("a", "x"),
                                    CE.Candidate("a", "y")))

    def test_variable_k_needs_no_padding(self):
        scorer = _scorer()
        for k in (2, 3, 5):
            dec = CE.Decision(
                state="the berth ledger is balanced",
                question="what does the ledger say?",
                candidates=tuple(CE.Candidate(f"c{i}", f"note number {i}")
                                 for i in range(k)),
                family=CE.EXTRACTION)
            self.assertEqual(len(scorer.score(dec)["z"]), k)


@unittest.skipUnless(HAVE_TORCH and REAL_SCORERS,
                     "NLI checkpoints not materialised under "
                     "artifacts/weights/ (model.weights fetch)")
class TestRealCheckpoints(unittest.TestCase):
    """Los pesos de verdad: cargan, puntúan y respetan el contrato."""

    def test_checkpoints_load_and_score_without_training(self):
        for sid in REAL_SCORERS:
            with self.subTest(scorer=sid):
                ps = CE.NliPairScorer(sid, device="cpu", batch_size=8)
                got = CE.CrossEncoderScorer(ps).score(CASE)
                self.assertEqual(len(got["z"]), CASE.k)
                self.assertAlmostEqual(sum(got["probs"]), 1.0, places=6)
                info = ps.describe()
                self.assertGreater(info["params"], 0)
                self.assertIn(info["n_classes"], (2, 3))

    def test_the_head_has_no_neuron_per_label(self):
        for sid in REAL_SCORERS:
            with self.subTest(scorer=sid):
                ps = CE.NliPairScorer(sid, device="cpu", batch_size=8)
                report = CE.scalar_output_report(ps)
                self.assertTrue(report["label_free"], report["offenders"])
                self.assertGreater(report["audited_params"], 0)

    def test_gate_checks_hold_on_real_weights(self):
        for sid in REAL_SCORERS:
            with self.subTest(scorer=sid):
                scorer = CE.CrossEncoderScorer(
                    CE.NliPairScorer(sid, device="cpu", batch_size=8))
                self.assertTrue(CE.check_permutation_invariance(
                    scorer, CASE, max_perms=6)["pass"])
                self.assertTrue(CE.check_text_follows_id(scorer, CASE)["pass"])
                self.assertTrue(CE.check_comparative_context(
                    scorer, CASE, CE.CONTRACT_FOREIGN_ID,
                    CE.CONTRACT_FOREIGN_TEXT)["pass"])


if __name__ == "__main__":
    unittest.main()
