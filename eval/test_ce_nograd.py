"""Tests del conjunto de sondas sin entrenar (#T-ce-scorer, unittest).

No miden calidad de ningún checkpoint: comprueban que el conjunto de casos
es válido como evidencia antes de que nadie lo ejecute. Un caso con un
gold que no está entre los candidatos, un grupo contrafactual cuyas dos
variantes tienen la misma respuesta, o una familia sin cobertura en los
dos idiomas convierten la cifra publicada en un número sin sujeto.

La parte que sí toca pesos vive en `model/test_ce_scorer.py` y se salta
cuando los checkpoints no están descargados.
"""
from __future__ import annotations

import unittest

from eval import ce_nograd as NG
from model import ce_scorer as CE


class TestCaseSet(unittest.TestCase):
    def test_every_case_builds_a_decision(self):
        decs = NG.decisions()
        self.assertEqual(len(decs), len(NG.CASES))
        for dec in decs:
            self.assertGreaterEqual(dec.k, 2)
            self.assertIsNotNone(dec.gold)

    def test_case_ids_are_unique(self):
        ids = [c["id"] for c in NG.CASES]
        self.assertEqual(len(set(ids)), len(ids))

    def test_the_five_families_are_all_covered_in_both_languages(self):
        seen = {(c["family"], c["lang"]) for c in NG.CASES}
        for family in CE.FAMILIES:
            for lang in CE.LANGS:
                self.assertIn((family, lang), seen,
                              f"{family} has no {lang} case")

    def test_counterfactual_groups_disagree_on_the_answer(self):
        """Un grupo cuyas variantes comparten respuesta no es contrafactual.

        Es el control que impide que `pair_joint` se pueda aprobar con una
        preferencia fija por un texto: dentro de un grupo, el gold tiene
        que cambiar.
        """
        groups: dict[str, list[dict]] = {}
        for case in NG.CASES:
            groups.setdefault(case["group"], []).append(case)
        multi = {g: cs for g, cs in groups.items() if len(cs) > 1}
        self.assertGreaterEqual(len(multi), 5)
        for group, cases in multi.items():
            golds = {c["gold"] for c in cases}
            self.assertGreater(len(golds), 1,
                               f"group {group} has one answer for every "
                               f"variant: it is not a counterfactual")

    def test_a_group_keeps_its_family_and_language(self):
        groups: dict[str, set] = {}
        for case in NG.CASES:
            groups.setdefault(case["group"], set()).add(
                (case["family"], case["lang"]))
        for group, kinds in groups.items():
            self.assertEqual(len(kinds), 1, f"group {group} straddles "
                                            f"families or languages: {kinds}")

    def test_gold_is_one_of_the_candidates(self):
        for case in NG.CASES:
            ids = [cid for cid, _ in case["candidates"]]
            self.assertIn(case["gold"], ids, case["id"])

    def test_candidate_ids_are_opaque(self):
        """El id nunca puede llevar la respuesta dentro."""
        for case in NG.CASES:
            for cid, text in case["candidates"]:
                self.assertRegex(cid, r"^c\d+$", case["id"])
                self.assertNotIn(cid.lower(), text.lower())

    def test_comparison_and_priority_carry_the_comparative_context(self):
        for dec in NG.decisions():
            want = dec.family in (CE.COMPARISON, CE.PRIORITY)
            self.assertEqual(dec.wants_context(), want,
                             f"{dec.meta['id']} ({dec.family})")


class TestAggregation(unittest.TestCase):
    ROWS = [
        {"group": "g1", "family": "f", "lang": "en", "k": 3, "correct": True},
        {"group": "g1", "family": "f", "lang": "en", "k": 3, "correct": False},
        {"group": "g2", "family": "f", "lang": "es", "k": 2, "correct": True},
        {"group": "g2", "family": "f", "lang": "es", "k": 2, "correct": True},
    ]

    def test_pair_joint_needs_every_variant(self):
        got = NG._pairs(self.ROWS)
        self.assertEqual(got["n_groups"], 2)
        self.assertEqual(got["joint_correct"], 1)
        self.assertFalse(got["per_group"]["g1"])
        self.assertTrue(got["per_group"]["g2"])

    def test_bucket_reports_chance_next_to_accuracy(self):
        got = NG._bucket(self.ROWS, "lang")
        self.assertAlmostEqual(got["en"]["accuracy"], 0.5)
        self.assertAlmostEqual(got["en"]["chance"], 1 / 3)
        self.assertAlmostEqual(got["es"]["chance"], 0.5)


if __name__ == "__main__":
    unittest.main()
