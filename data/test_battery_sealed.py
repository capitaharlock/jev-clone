"""Tests de la batería sellada (#T-battery-sealed).

El gate de la task, ejecutable:

* los 600 casos pasan el validador de `#T-episode-contract`;
* el reparto real familia x idioma x K coincide con la mezcla
  declarada (fechada antes de redactar);
* el gold cumple la regla determinista con la sal sellada en los 600;
* cada grupo es contrafactual de verdad y CERO variant_group se
  comparten con desarrollo (assert, no comentario);
* el solape entidad/espacio con desarrollo está bajo el umbral
  que el manifest escribió antes de medir;
* el diagnóstico K=20/77 vive en su bloque: cualquier mezcla con
  la meta del 70 % falla;
* apertura única: abrir una vez ok, abrir dos veces raise/exit!=0,
  y solo #T-ce-confirm puede abrir (los tests usan copias
  temporales; el ledger real sigue ausente: consumed=false).

Run:
    PYTHONPATH=. pytest data/test_battery_sealed.py -q
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import unittest
from collections import Counter

from . import battery_sealed as BS
from . import episode_contract as EC

MANIFEST_CELLS = {
    (c["family"], c["lang"], 2): c["k2"]
    for c in BS.load_manifest()["declared_mix"]["cells"]
}
MANIFEST_CELLS.update(
    {
        (c["family"], c["lang"], 3): c["k3"]
        for c in BS.load_manifest()["declared_mix"]["cells"]
    }
)
MANIFEST_CELLS.update(
    {
        (c["family"], c["lang"], 8): c["k8"]
        for c in BS.load_manifest()["declared_mix"]["cells"]
    }
)


def _episodes() -> list[dict]:
    return BS.load_sealed()


def _diag() -> list[dict]:
    return BS.load_diag()


class ContractTest(unittest.TestCase):
    """Los 600 sellados pasan el MISMO validador, sin duplicados."""

    def test_all_600_pass_validator(self):
        episodes = _episodes()
        self.assertEqual(len(episodes), 600)
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
        manifest = BS.load_manifest()
        self.assertEqual(manifest["version"], "battery-sealed-v1")
        self.assertRegex(manifest["date_declared"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertEqual(manifest["schema_version"], EC.SCHEMA_VERSION)
        self.assertEqual(manifest["gold_rule"]["salt"], BS.GOLD_SALT)

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
            {fam: 120 for fam in EC.FAMILIES},
        )
        self.assertEqual(
            Counter(ep["lang"] for ep in episodes), {"es": 300, "en": 300}
        )
        self.assertEqual(
            Counter(len(ep["candidates"]) for ep in episodes),
            {2: 240, 3: 180, 8: 180},
        )


class GoldTest(unittest.TestCase):
    """El gold lo fijó la regla sellada con semilla, no el autor."""

    def test_gold_matches_sealed_rule(self):
        self.assertEqual(BS.check_gold(_episodes()), [])

    def test_control_keeps_gold_and_decisive_moves_it(self):
        from .battery_dev import check_counterfactual_shape

        self.assertEqual(check_counterfactual_shape(_episodes()), [])

    def test_every_group_is_a_full_pair_or_triple(self):
        per_group: dict[str, list] = {}
        for ep in _episodes():
            per_group.setdefault(ep["variant_group"], []).append(ep)
        self.assertEqual(len(per_group), 210)
        sizes = Counter(len(v) for v in per_group.values())
        self.assertEqual(sizes[3], 180)
        self.assertEqual(sizes[2], 30)


class NamespaceTest(unittest.TestCase):
    """`sealed-` aquí; cero grupos compartidos con desarrollo."""

    def test_sealed_prefix_strict(self):
        self.assertEqual(BS.check_namespace(_episodes()), [])

    def test_no_group_shared_with_dev(self):
        # assert, no comentario: la intersección se MIDE aquí.
        sealed_groups = {ep["variant_group"] for ep in _episodes()}
        with open(BS.DEV_BATTERY, encoding="utf-8") as fh:
            dev_groups = {
                json.loads(line)["variant_group"]
                for line in fh
                if line.strip()
            }
        self.assertEqual(sealed_groups & dev_groups, set())
        self.assertEqual(BS.check_dev_disjoint(_episodes()), [])


class OverlapTest(unittest.TestCase):
    """Solape entidad/espacio bajo el umbral escrito ANTES de medir."""

    def test_entity_overlap_under_written_threshold(self):
        overlap = BS.measure_overlap(_episodes())
        manifest_thr = float(
            BS.load_manifest()["overlap_rule"]["entity"].split("<=")[1].strip()
        )
        self.assertEqual(overlap["entity_threshold"], manifest_thr)
        self.assertLessEqual(overlap["entity_overlap"], manifest_thr)
        self.assertTrue(overlap["entity_pass"])

    def test_candidate_text_overlap_is_zero(self):
        overlap = BS.measure_overlap(_episodes())
        self.assertEqual(overlap["candidate_n_shared"], 0)
        self.assertTrue(overlap["candidate_pass"])


class DiagSeparationTest(unittest.TestCase):
    """El diagnóstico K=20/77 nunca se promedia con la meta."""

    def test_diag_is_valid_and_ks_are_20_77(self):
        diag = _diag()
        self.assertEqual(len(diag), 9)
        report = EC.batch_validate(diag)
        self.assertEqual(
            [(ep["id"], r) for ep, r in zip(diag, report["per_episode"]) if r],
            [],
        )
        self.assertEqual(sorted({len(ep["candidates"]) for ep in diag}), [20, 77])
        self.assertTrue(all(ep["variant_group"].startswith("diag-") for ep in diag))
        self.assertEqual(BS.check_diag_gold(diag), [])

    def test_target_report_reads_sealed_only(self):
        report = BS.build_target_report()
        self.assertEqual(report["n"], 600)
        self.assertFalse(report["diag_included"])
        self.assertEqual(
            report["k_counts"], {"2": 240, "3": 180, "8": 180}
        )

    def test_target_report_refuses_diag_cases(self):
        with self.assertRaises(ValueError):
            BS.build_target_report(_diag())
        mixed = _episodes()[:10] + _diag()[:1]
        with self.assertRaises(ValueError):
            BS.build_target_report(mixed)

    def test_diag_and_sealed_share_no_groups_or_ks(self):
        sealed_groups = {ep["variant_group"] for ep in _episodes()}
        diag_groups = {ep["variant_group"] for ep in _diag()}
        self.assertEqual(sealed_groups & diag_groups, set())
        sealed_ks = {len(ep["candidates"]) for ep in _episodes()}
        diag_ks = {len(ep["candidates"]) for ep in _diag()}
        self.assertEqual(sealed_ks & diag_ks, set())


class SingleOpenTest(unittest.TestCase):
    """Apertura única, solo #T-ce-confirm (copias temporales)."""

    def _tmp_copy(self, tmp_dir: str) -> tuple[str, str]:
        sealed_tmp = os.path.join(tmp_dir, "battery_sealed.jsonl")
        shutil.copy(BS.SEALED, sealed_tmp)
        return sealed_tmp, os.path.join(tmp_dir, "opened.json")

    def test_open_once_ok_open_twice_raises(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            sealed_tmp, ledger_tmp = self._tmp_copy(tmp)
            opened = BS.open_sealed(
                "T-ce-confirm",
                sealed_path=sealed_tmp,
                ledger_path=ledger_tmp,
            )
            self.assertEqual(len(opened), 600)
            self.assertTrue(os.path.exists(ledger_tmp))
            with open(ledger_tmp, encoding="utf-8") as fh:
                record = json.load(fh)
            self.assertTrue(record["consumed"])
            self.assertEqual(record["opened_by"], "T-ce-confirm")
            with self.assertRaises(BS.SealedAlreadyConsumed):
                BS.open_sealed(
                    "T-ce-confirm",
                    sealed_path=sealed_tmp,
                    ledger_path=ledger_tmp,
                )

    def test_only_ce_confirm_may_open(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            sealed_tmp, ledger_tmp = self._tmp_copy(tmp)
            with self.assertRaises(BS.SealedAccessDenied):
                BS.open_sealed(
                    "T-battery-sealed",
                    sealed_path=sealed_tmp,
                    ledger_path=ledger_tmp,
                )
            self.assertFalse(os.path.exists(ledger_tmp))

    def test_cli_refuses_with_nonzero_exit(self):
        proc = subprocess.run(
            [sys.executable, "-m", "data.battery_sealed", "--open", "T-nobody"],
            capture_output=True,
            text=True,
            cwd=BS.ROOT,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("REFUSED", proc.stdout + proc.stderr)

    def test_real_ledger_still_absent(self):
        # Nada de esta task abre el sellado real: consumed=false.
        self.assertFalse(BS.is_consumed())


if __name__ == "__main__":
    unittest.main()
