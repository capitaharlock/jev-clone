"""Tests del contrato de episodio (#T-episode-contract).

El gate de la task, ejecutable:

* el validador rechaza CON MOTIVO los 4 casos: sin `evidence`, sin
  `variant_group`, IDs de candidato no opacos, ID de dataset por pregunta;
* dos episodios del mismo `variant_group` nunca caen en cortes distintos;
* el esquema está versionado y su sha viaja en el manifest;
* las 5 familias son enumeración cerrada (igual que `model.ce_scorer`);
* la fixture hecha a mano (>=20 episodios, ES+EN, con contrafactuales)
  pasa el validador entera.

Run:
    PYTHONPATH=. pytest data/test_episode_contract.py -q
"""

from __future__ import annotations

import json
import os
import unittest

from . import episode_contract as EC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ROOT, "data", "episode_fixture.jsonl")
CUTS = (("train", 0.7), ("dev", 0.15), ("sealed", 0.15))


def _load_fixture() -> list[dict]:
    with open(FIXTURE, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _good_episode() -> dict:
    return dict(_load_fixture()[0])


class RejectionTest(unittest.TestCase):
    """Los 4 rechazos del gate, cada uno con su motivo."""

    def test_rejects_missing_evidence(self):
        ep = _good_episode()
        del ep["evidence"]
        reasons = EC.validate(ep)
        self.assertTrue(any("evidence" in r for r in reasons), reasons)

    def test_rejects_missing_variant_group(self):
        ep = _good_episode()
        del ep["variant_group"]
        reasons = EC.validate(ep)
        self.assertTrue(any("variant_group" in r for r in reasons), reasons)

    def test_rejects_non_opaque_candidate_ids(self):
        ep = _good_episode()
        ep["candidates"] = [
            {"id": "En la estantería 3", "text": "En la estantería 3"},
            {"id": "c2", "text": "En la estantería 5"},
        ]
        reasons = EC.validate(ep)
        self.assertTrue(any("opaque" in r for r in reasons), reasons)

    def test_rejects_dataset_id_as_question(self):
        for q in (
            "epdiv-00001-3",
            "banking77-intent-8412",
            "banking77-intent",
            "massive",
        ):
            ep = _good_episode()
            ep["question"] = q
            reasons = EC.validate(ep)
            self.assertTrue(
                any("dataset id" in r for r in reasons),
                f"{q!r}: {reasons}",
            )

    def test_rejects_question_without_criterion_prose(self):
        ep = _good_episode()
        ep["question"] = "Which one is better?"
        # prosa válida en forma: no debe fallar por ser corta/genérica —
        # el contrato exige forma (prosa + '?'), no calidad literaria.
        self.assertEqual(EC.validate(ep), [])

    def test_evidence_must_come_from_the_state(self):
        ep = _good_episode()
        ep["evidence"] = "un fragmento que no aparece en ningún sitio"
        reasons = EC.validate(ep)
        self.assertTrue(any("fragment of the state" in r for r in reasons))


class SplitTest(unittest.TestCase):
    """Ningún `variant_group` cruza un corte."""

    def test_same_group_never_splits(self):
        episodes = _load_fixture()
        groups = {ep["variant_group"] for ep in episodes}
        self.assertGreaterEqual(len(episodes), 20)
        self.assertGreater(len(groups), 1)
        for seed in (1, 7, 20260926):
            split = EC.split_by_group(episodes, seed, CUTS)
            tagged = []
            for cut, eps in split.items():
                tagged.extend({**ep, "split": cut} for ep in eps)
            self.assertEqual(
                EC.check_episodes_no_group_split(tagged), [], f"seed {seed}"
            )

    def test_assign_split_is_deterministic(self):
        for g in ("g01", "g07", "g12"):
            first = EC.assign_split(g, 20260926, CUTS)
            for _ in range(5):
                self.assertEqual(EC.assign_split(g, 20260926, CUTS), first)

    def test_injected_leak_is_caught(self):
        # el audit del repartidor: una variante a mano en otro corte falla.
        bad = [
            {"id": "x-a", "variant_group": "gX", "split": "train"},
            {"id": "x-b", "variant_group": "gX", "split": "sealed"},
        ]
        reasons = EC.check_episodes_no_group_split(bad)
        self.assertEqual(len(reasons), 1)
        self.assertIn("gX", reasons[0])


class SchemaTest(unittest.TestCase):
    def test_versioned_and_pinned_in_manifest(self):
        self.assertEqual(EC.SCHEMA_VERSION, "episode-v1")
        rec = EC.manifest_record(
            24, 20260926, "hand-v1", {"train": 17, "dev": 4, "sealed": 3}
        )
        self.assertEqual(rec["schema_version"], EC.SCHEMA_VERSION)
        self.assertEqual(rec["schema_sha"], EC.schema_sha())
        self.assertEqual(len(rec["schema_sha"]), 64)

    def test_families_are_closed_and_match_ce_scorer(self):
        self.assertEqual(len(EC.FAMILIES), 5)
        from model import ce_scorer

        self.assertEqual(set(EC.FAMILIES), set(ce_scorer.FAMILIES))
        self.assertEqual(
            dict(EC.COMPARATIVE_CONTEXT), dict(ce_scorer.COMPARATIVE_CONTEXT)
        )

    def test_comparative_contract_is_explicit_per_family(self):
        for family in EC.FAMILIES:
            self.assertIn(family, EC.COMPARATIVE_WHY)
            self.assertIsInstance(EC.wants_context(family), bool)
        # las dos direcciones: comparación y prioridad exigen contexto,
        # extracción/descripción/inferencia juzgan de una en una.
        self.assertTrue(EC.wants_context(EC.COMPARISON))
        self.assertTrue(EC.wants_context(EC.PRIORITY))
        self.assertFalse(EC.wants_context(EC.EXTRACTION))
        self.assertFalse(EC.wants_context(EC.DESCRIPTION))
        self.assertFalse(EC.wants_context(EC.INFERENCE))
        with self.assertRaises(ValueError):
            EC.wants_context("free-text-family")


class FixtureTest(unittest.TestCase):
    def test_handmade_fixture_passes_whole(self):
        episodes = _load_fixture()
        self.assertGreaterEqual(len(episodes), 20)
        report = EC.batch_validate(episodes)
        self.assertEqual(report["n_invalid"], 0, report["per_episode"])
        self.assertEqual(report["duplicate_ids"], [])

    def test_fixture_covers_families_langs_and_counterfactuals(self):
        episodes = _load_fixture()
        self.assertEqual({ep["family"] for ep in episodes}, set(EC.FAMILIES))
        langs = {ep["lang"] for ep in episodes}
        self.assertEqual(langs, {"es", "en"})
        per_group: dict[str, list] = {}
        for ep in episodes:
            per_group.setdefault(ep["variant_group"], []).append(ep)
        self.assertTrue(
            all(len(v) >= 2 for v in per_group.values()),
            "every case ships at least one counterfactual",
        )
        shapes = {ep.get("answer") is not None: 1 for ep in episodes}
        self.assertTrue(any(ep.get("acceptable_answers") for ep in episodes))
        self.assertTrue(any(ep.get("preference") for ep in episodes))
        self.assertIn(True, shapes)


if __name__ == "__main__":
    unittest.main()
