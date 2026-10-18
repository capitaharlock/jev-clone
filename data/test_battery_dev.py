"""Tests de la batería de desarrollo (#T-battery-dev).

El gate de la task, ejecutable:

* los 400 casos pasan el validador de `#T-episode-contract`;
* el reparto real familia x idioma x K coincide con la mezcla
  declarada (fechada antes de generar);
* el gold cumple la regla determinista con semilla en los 400;
* cada grupo es contrafactual de verdad (control conserva,
  decisiva mueve; sin grupos compartidos a medias);
* namespace `dev-` estricto y disyunción con el sellado
  (prefijos hoy; intersección en cuanto el sellado exista).

Run:
    PYTHONPATH=. pytest data/test_battery_dev.py -q
"""

from __future__ import annotations

import unittest
from collections import Counter

from . import battery_dev as BD
from . import episode_contract as EC

MANIFEST_CELLS = {
    (c["family"], c["lang"], 2): c["k2"]
    for c in BD.load_manifest()["declared_mix"]["cells"]
}
MANIFEST_CELLS.update(
    {
        (c["family"], c["lang"], 3): c["k3"]
        for c in BD.load_manifest()["declared_mix"]["cells"]
    }
)
MANIFEST_CELLS.update(
    {
        (c["family"], c["lang"], 8): c["k8"]
        for c in BD.load_manifest()["declared_mix"]["cells"]
    }
)


def _episodes() -> list[dict]:
    return BD.load_battery()


class ContractTest(unittest.TestCase):
    """Los 400 casos pasan el validador del contrato, sin duplicados."""

    def test_all_400_pass_validator(self):
        episodes = _episodes()
        self.assertEqual(len(episodes), 400)
        report = EC.batch_validate(episodes)
        bad = [
            (ep["id"], reasons)
            for ep, reasons in zip(episodes, report["per_episode"])
            if reasons
        ]
        self.assertEqual(bad, [])
        self.assertEqual(report["duplicate_ids"], [])


class MixTest(unittest.TestCase):
    """El reparto real coincide con la mezcla declarada (exacto)."""

    def test_declared_mix_is_dated_and_versioned(self):
        manifest = BD.load_manifest()
        self.assertEqual(manifest["version"], "battery-dev-v1")
        self.assertRegex(manifest["date_declared"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertEqual(manifest["schema_version"], EC.SCHEMA_VERSION)

    def test_family_lang_k_matches_declared(self):
        episodes = _episodes()
        real = Counter(
            (ep["family"], ep["lang"], len(ep["candidates"])) for ep in episodes
        )
        self.assertEqual(dict(real), MANIFEST_CELLS)

    def test_marginals(self):
        episodes = _episodes()
        self.assertEqual(
            Counter(ep["family"] for ep in episodes),
            {fam: 80 for fam in EC.FAMILIES},
        )
        self.assertEqual(
            Counter(ep["lang"] for ep in episodes), {"es": 200, "en": 200}
        )
        self.assertEqual(
            Counter(len(ep["candidates"]) for ep in episodes),
            {2: 160, 3: 120, 8: 120},
        )


class GoldTest(unittest.TestCase):
    """El gold lo fijó la regla con semilla, no el juicio del autor."""

    def test_gold_matches_seeded_rule(self):
        self.assertEqual(BD.check_gold(_episodes()), [])

    def test_control_keeps_gold_and_decisive_moves_it(self):
        self.assertEqual(BD.check_counterfactual_shape(_episodes()), [])

    def test_every_group_is_a_full_pair_or_triple(self):
        per_group: dict[str, list] = {}
        for ep in _episodes():
            per_group.setdefault(ep["variant_group"], []).append(ep)
        self.assertEqual(len(per_group), 140)
        sizes = Counter(len(v) for v in per_group.values())
        self.assertEqual(sizes[3], 120)
        self.assertEqual(sizes[2], 20)


class NamespaceTest(unittest.TestCase):
    """`dev-` aquí, `sealed-` nunca aquí, y cero grupos compartidos."""

    def test_dev_prefix_strict(self):
        self.assertEqual(BD.check_namespace(_episodes()), [])

    def test_no_group_shared_with_sealed(self):
        state, problems = BD.check_sealed_disjoint(_episodes())
        self.assertIn(state, ("absent", "clean"), problems)
        self.assertEqual(problems, [])
        if state == "absent":
            # el sellado no existe aún: la garantía es el prefijo.
            self.assertEqual(BD.find_sealed_files(), [])


if __name__ == "__main__":
    unittest.main()
