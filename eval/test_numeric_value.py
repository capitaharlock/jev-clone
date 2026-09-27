"""Tests de la prueba de valor del generador por regla (#T-numeric-gen).

Sin GPU y sin pesos: lo que se comprueba aquí es la fontanería que decide
el veredicto, no el entreno. El entreno va por el job `trainer` y su
cifra la firma `measure` sobre el checkpoint real.

* la predicción está escrita antes y `predict` no la reescribe;
* la mezcla toma GRUPOS enteros y equilibrados entre las cuatro celdas —
  cortar los 5 000 primeros del lote daría una sola celda;
* el holdout se recomputa desde el directorio y la semilla, sin reentrenar;
* el veredicto sale de la familia que la task manda mirar, y un NO-GO
  escribe el cuello de botella en vez de callarlo.

Run:
    PYTHONPATH=. pytest eval/test_numeric_value.py -q
"""

from __future__ import annotations

import json
import shutil
import unittest
import unittest.mock
from pathlib import Path

from data import episode_contract as EC
from data import episode_gen as EG
from data import rule_variety as RV
from eval import numeric_value as NV

TMP = Path("/tmp/numeric-value-test")


def _rule_batch(n: int = 800) -> Path:
    out = TMP / "rule"
    shutil.rmtree(out, ignore_errors=True)
    EG.run(n=n, seed=1, prose="local", teacher="stub", out_dir=str(out),
           families=list(RV.FAMILIES), variety=16, group_prefix="rule0001-")
    return out


def _pilot(n: int = 40) -> Path:
    """Un «piloto» de mentira: los episodios de las cinco familias que
    escribe `data.episode_gen` sin variedad — otro generator_version."""
    out = TMP / "pilot"
    shutil.rmtree(out, ignore_errors=True)
    EG.run(n=n, seed=99, prose="local", teacher="stub", out_dir=str(out))
    return out / "episodes.jsonl"


class PredictionTest(unittest.TestCase):
    def test_the_prediction_names_its_rule_and_its_date(self):
        self.assertEqual(NV.PREDICTION["written_utc"], "2026-09-27")
        self.assertIn("CI95", NV.PREDICTION["go_if"])
        self.assertIn("backbone", NV.PREDICTION["no_go_if"])
        self.assertIn(NV.TARGET_FAMILY, NV.PREDICTION["expect"])
        self.assertEqual(NV.PREDICTION["bootstrap"],
                         {"B": 2000, "seed": 20260927, "unit": "paired row"})

    def test_predict_writes_it_once_and_never_rewrites_it(self):
        path = TMP / "gate-predict.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        with unittest.mock.patch.object(RV, "GATE_PATH", str(path)):
            NV.predict()
            first = json.loads(path.read_text())
            tampered = dict(first)
            tampered["value_proof_prediction"] = dict(
                first["value_proof_prediction"], go_if="whatever moves")
            path.write_text(json.dumps(tampered))
            NV.predict()
            after = json.loads(path.read_text())
        self.assertEqual(after["value_proof_prediction"]["go_if"],
                         "whatever moves")  # no se toca: no se reescribe
        self.assertEqual(first["value_proof_prediction"]["go_if"],
                         NV.PREDICTION["go_if"])


class MixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rule = _rule_batch()
        cls.pilot = _pilot()

    def test_the_sample_is_balanced_over_the_four_cells(self):
        episodes = NV._read_jsonl(self.rule / "episodes.jsonl")
        taken = NV.sample_by_group(episodes, 400, 20260927)
        cells: dict = {}
        for ep in taken:
            key = f"{ep['family']}/{ep['lang']}"
            cells[key] = cells.get(key, 0) + 1
        self.assertEqual(len(cells), 4, cells)
        self.assertLessEqual(max(cells.values()) - min(cells.values()), 4,
                             cells)

    def test_cutting_the_first_n_would_give_one_cell_the_sample_does_not(self):
        episodes = NV._read_jsonl(self.rule / "episodes.jsonl")
        naive = {f"{ep['family']}/{ep['lang']}" for ep in episodes[:200]}
        self.assertEqual(len(naive), 1)  # el fallo que este muestreo evita
        taken = NV.sample_by_group(episodes, 200, 20260927)
        self.assertEqual(
            len({f"{ep['family']}/{ep['lang']}" for ep in taken}), 4)

    def test_no_variant_group_is_cut_in_half(self):
        episodes = NV._read_jsonl(self.rule / "episodes.jsonl")
        whole = {}
        for ep in episodes:
            whole[ep["variant_group"]] = whole.get(ep["variant_group"], 0) + 1
        taken = NV.sample_by_group(episodes, 400, 20260927)
        got: dict = {}
        for ep in taken:
            got[ep["variant_group"]] = got.get(ep["variant_group"], 0) + 1
        for group, count in got.items():
            self.assertEqual(count, whole[group], group)

    def test_the_mix_manifest_names_both_sources_with_their_shas(self):
        out = TMP / "mix"
        shutil.rmtree(out, ignore_errors=True)
        manifest = NV.mix(self.rule, self.pilot, 400, out)
        self.assertEqual(manifest["sources"]["rule"]["n_taken"], 400)
        self.assertEqual(manifest["sources"]["pilot"]["n_taken"], 40)
        self.assertEqual(manifest["n_episodes"], 440)
        for name in ("rule", "pilot"):
            self.assertEqual(len(manifest["sources"][name]["sha256"]), 64)
        self.assertEqual(len(manifest["episodes_sha256"]), 64)
        self.assertIs(manifest["episodes_versioned"], False)
        episodes = NV._read_jsonl(out / "episodes.jsonl")
        self.assertEqual(len(episodes), 440)
        self.assertEqual(EC.batch_validate(episodes)["n_invalid"], 0)

    def test_the_two_sources_stay_tellable_apart(self):
        out = TMP / "mix2"
        shutil.rmtree(out, ignore_errors=True)
        NV.mix(self.rule, self.pilot, 200, out)
        episodes = NV._read_jsonl(out / "episodes.jsonl")
        self.assertEqual(sum(1 for ep in episodes if NV.is_rule(ep)), 200)
        self.assertEqual(sum(1 for ep in episodes if not NV.is_rule(ep)), 40)

    def test_a_duplicate_id_refuses_the_mix_it_does_not_train_on_it(self):
        out = TMP / "mix-dupe"
        shutil.rmtree(out, ignore_errors=True)
        with self.assertRaises(ValueError):
            NV.mix(self.rule, self.rule / "episodes.jsonl", 200, out)


class HoldoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rule = _rule_batch()
        cls.pilot = _pilot()
        cls.mix = TMP / "mix-hold"
        shutil.rmtree(cls.mix, ignore_errors=True)
        NV.mix(cls.rule, cls.pilot, 400, cls.mix)

    def test_the_holdout_recomputes_without_retraining(self):
        a = NV.holdout_of(self.mix, 20260927, 0.2)
        b = NV.holdout_of(self.mix, 20260927, 0.2)
        self.assertEqual([e["id"] for e in a["all"]],
                         [e["id"] for e in b["all"]])
        self.assertGreater(len(a["pilot"]), 0)
        self.assertGreater(len(a["rule"]), 0)
        self.assertEqual(len(a["all"]), len(a["pilot"]) + len(a["rule"]))
        self.assertEqual(a["split"]["n_train"] + a["split"]["n_holdout"], 440)

    def test_no_group_straddles_train_and_holdout(self):
        hold = NV.holdout_of(self.mix, 20260927, 0.2)
        held = {ep["variant_group"] for ep in hold["all"]}
        episodes = NV._read_jsonl(self.mix / "episodes.jsonl")
        held_ids = {ep["id"] for ep in hold["all"]}
        for ep in episodes:
            if ep["variant_group"] in held:
                self.assertIn(ep["id"], held_ids, ep["id"])

    def test_another_split_seed_is_another_holdout(self):
        a = NV.holdout_of(self.mix, 20260927, 0.2)
        b = NV.holdout_of(self.mix, 20260101, 0.2)
        self.assertNotEqual([e["id"] for e in a["all"]],
                            [e["id"] for e in b["all"]])


class TrackingControlTest(unittest.TestCase):
    def _rows(self, picks: list, episodes: list) -> list:
        rows = []
        for ep, pick in zip(episodes, picks):
            k = len(ep["candidates"])
            probs = [0.1] * k
            probs[pick] = 0.9
            rows.append({"row_id": ep["id"], "family": ep["family"],
                         "variant_group": ep["variant_group"], "k": k,
                         "gold_index": RV.gold_position(ep),
                         "pred": pick, "probs": probs})
        return rows

    def test_a_pick_that_follows_the_gold_counts_as_followed(self):
        eps = EG.load_episodes(str(_rule_batch(80)))
        group = eps[0]["variant_group"]
        members = [ep for ep in eps if ep["variant_group"] == group]
        picks = [RV.gold_position(ep) for ep in members]
        got = NV.tracking_of(members, self._rows(picks, members), "test")
        self.assertGreater(got["n_pairs"], 0)
        self.assertEqual(got["same_slot"], 0)
        self.assertEqual(got["rate"], 1.0)

    def test_a_pick_stuck_in_one_slot_counts_as_not_followed(self):
        eps = EG.load_episodes(str(_rule_batch(80)))
        group = eps[0]["variant_group"]
        members = [ep for ep in eps if ep["variant_group"] == group]
        picks = [0] * len(members)
        got = NV.tracking_of(members, self._rows(picks, members), "test")
        self.assertGreater(got["n_pairs"], 0)
        self.assertEqual(got["followed"], 0)
        self.assertEqual(got["rate"], 0.0)


class GateVerdictTest(unittest.TestCase):
    """El veredicto: lo decide la familia que la task manda mirar."""

    def _files(self, control: float, tuned: float, lo: float) -> tuple:
        base = TMP / "gate"
        base.mkdir(parents=True, exist_ok=True)
        smoke = base / "smoke.json"
        smoke.write_text(json.dumps({"comparison": {
            "verdict": "GO", "forced": {"point": 0.05, "ci95": [0.01, 0.09]},
            "counterfactual_joint": {"point": 0.1, "ci95": [0.04, 0.2]},
            "by_family": {}, "by_lang": {}}}))
        hold = base / "holdout.json"
        hold.write_text(json.dumps({
            "run": "artifacts/checkpoints/ce/numeric-5k",
            "mix": "artifacts/episodes-rule/mix-smoke",
            "budget_decisions": 5000, "decisions_consumed": 5000,
            "whole_holdout": {"by_source": {}, "by_family": {
                NV.TARGET_FAMILY: {"control": control, "tuned": tuned,
                                   "point": round(tuned - control, 6),
                                   "ci95": [lo, lo + 0.2], "n": 510,
                                   "moves": lo > 0}}},
            "pilot_holdout": {"n": 120, "by_lang": {}, "by_family": {
                NV.TARGET_FAMILY: {
                    "control": control, "tuned": tuned, "point":
                    round(tuned - control, 6), "ci95": [lo, lo + 0.2],
                    "n": 60, "moves": lo > 0},
                EC.PRIORITY: {"control": 0.62, "tuned": 0.78, "point": 0.16,
                              "ci95": [0.05, 0.27], "n": 60, "moves": True},
            }}}))
        return base / "gate.json", smoke, hold

    def test_a_family_that_moves_is_a_go(self):
        gate_path, smoke, hold = self._files(0.574, 0.71, 0.06)
        gate_path.unlink(missing_ok=True)
        with unittest.mock.patch.object(RV, "GATE_PATH", str(gate_path)):
            got = NV.gate("run", smoke=smoke, holdout=hold)["value_proof"]
        self.assertEqual(got["verdict"], "GO")
        self.assertIs(got["pass"], True)
        self.assertNotIn("bottleneck", got)

    def test_a_family_that_does_not_move_writes_the_bottleneck(self):
        gate_path, smoke, hold = self._files(0.574, 0.574, -0.08)
        gate_path.unlink(missing_ok=True)
        with unittest.mock.patch.object(RV, "GATE_PATH", str(gate_path)):
            got = NV.gate("run", smoke=smoke, holdout=hold)["value_proof"]
        self.assertEqual(got["verdict"], "NO-GO")
        self.assertIs(got["pass"], False)
        self.assertEqual(got["bottleneck"]["is"], "backbone")
        self.assertEqual(got["bottleneck"]["hand_to"], "#T-backbone-ladder")
        self.assertIn("0.574", got["bottleneck"]["why"])

    def test_a_holdout_without_the_target_family_is_an_error(self):
        gate_path, smoke, hold = self._files(0.5, 0.6, 0.01)
        doc = json.loads(hold.read_text())
        doc["pilot_holdout"]["by_family"].pop(NV.TARGET_FAMILY)
        hold.write_text(json.dumps(doc))
        with unittest.mock.patch.object(RV, "GATE_PATH", str(gate_path)):
            with self.assertRaises(ValueError):
                NV.gate("run", smoke=smoke, holdout=hold)

    def test_an_average_that_moves_does_not_carry_the_family(self):
        # dev dice GO y la familia del holdout no se mueve: el veredicto de
        # ESTA task es el del holdout, no el promedio de dev.
        gate_path, smoke, hold = self._files(0.574, 0.60, -0.02)
        gate_path.unlink(missing_ok=True)
        with unittest.mock.patch.object(RV, "GATE_PATH", str(gate_path)):
            got = NV.gate("run", smoke=smoke, holdout=hold)["value_proof"]
        self.assertEqual(got["dev"]["verdict"], "GO")
        self.assertEqual(got["verdict"], "NO-GO")
        self.assertEqual(got["power"]["n_target_rows"], 60)
        self.assertEqual(got["power"]["higher_powered_reading"]["n"], 510)


class PairedDeltaTest(unittest.TestCase):
    def test_the_bucket_delta_is_paired_on_the_same_rows(self):
        def rows(right: list) -> list:
            return [{"row_id": f"r{i}", "family": "attribute_comparison",
                     "variant_group": f"g{i}", "k": 2,
                     "gold_index": 0, "pred": 0 if ok else 1,
                     "probs": [0.9, 0.1] if ok else [0.1, 0.9]}
                    for i, ok in enumerate(right)]

        control = rows([True] * 20 + [False] * 20)
        tuned = rows([True] * 32 + [False] * 8)
        got = NV._by(control, tuned, "family", 20260927, 400)
        node = got["attribute_comparison"]
        self.assertEqual(node["control"], 0.5)
        self.assertEqual(node["tuned"], 0.8)
        self.assertAlmostEqual(node["point"], 0.3, places=6)
        self.assertEqual(node["n"], 40)
        self.assertIs(node["moves"], True)

    def test_no_movement_is_not_a_movement(self):
        rows = [{"row_id": f"r{i}", "family": "priority_decision",
                 "variant_group": f"g{i}", "k": 3, "gold_index": 0,
                 "pred": 1, "probs": [0.2, 0.7, 0.1]} for i in range(30)]
        got = NV._by(rows, [dict(r) for r in rows], "family", 1, 200)
        node = got["priority_decision"]
        self.assertEqual(node["point"], 0.0)
        self.assertIs(node["moves"], False)


if __name__ == "__main__":
    unittest.main()
