"""The ablation of #T-antiscale-diag reproduces, number for number.

Two layers, because the inputs live on two sides of `.gitignore`:

* the **arithmetic** layer reads the committed
  `artifacts/gates/T-antiscale-diag/gate.json` and re-derives every claim
  it makes — the decomposition is exact, the shares sum to one, the
  dominant axis is the one the stated rule picks, an unmeasured axis
  carries a reason and no number. It runs anywhere, CI included.
* the **re-run** layer re-executes the ablation at the same seed and
  asserts it reproduces the committed gate field for field. It needs
  `artifacts/runs/`, which `.gitignore` keeps out of the repo, so it skips
  itself when those runs are not on the machine — loudly, by name.
"""
from __future__ import annotations

import json
import os
import unittest

from tools.diagnose import antiscale as A

GATE = A.GATE_PATH
#: `generated_utc` is a clock reading, not a measurement
VOLATILE = ("generated_utc",)


def committed_gate() -> dict:
    with open(GATE) as fh:
        return json.load(fh)


def runs_present(seed: int = A.ABLATION_SEED) -> bool:
    return all(os.path.exists(os.path.join(
        A.ROOT, "artifacts", "runs",
        A.CURVE_RUN.format(backbone=b, seed=seed), "summary.json"))
        for b in A.ARMS)


def strip(gate: dict) -> dict:
    return {k: v for k, v in gate.items() if k not in VOLATILE}


class GateArithmetic(unittest.TestCase):
    """What the committed gate claims, re-derived from the gate itself."""

    @classmethod
    def setUpClass(cls):
        cls.gate = committed_gate()

    def test_gate_is_valid_json_of_this_task(self):
        self.assertEqual(self.gate["task"], A.TASK)
        self.assertEqual(self.gate["format"], 1)
        self.assertEqual(self.gate["seed"], A.ABLATION_SEED)
        self.assertEqual(len(self.gate["axes"]), 5)

    def test_decomposition_is_exact(self):
        """fall = fall_ranking + fall_abstention, to the last digit."""
        for arm, f in self.gate["fall"].items():
            with self.subTest(arm=arm):
                self.assertAlmostEqual(
                    f["fall"], f["fall_ranking"] + f["fall_abstention"],
                    places=6)
                self.assertAlmostEqual(
                    f["share_ranking"] + f["share_abstention"], 1.0,
                    places=6)
                ref, fin = f["unseen_accuracy"]
                self.assertAlmostEqual(f["fall"], ref - fin, places=6)
                self.assertGreater(f["fall"], 0.0,
                                   "the segment under ablation must fall")

    def test_shares_are_the_mean_of_the_arms(self):
        falls = self.gate["fall"]
        for key, field in (("2_runaway_abstention", "share_abstention"),
                           ("4_calibration_vs_ranking", "share_ranking")):
            axis = self.gate["axes"][key]
            want = round(sum(f[field] for f in falls.values())
                         / len(falls), 6)
            self.assertAlmostEqual(axis["number"]["share_of_fall"], want,
                                   places=6, msg=key)

    def test_dominant_axis_follows_the_stated_rule(self):
        dom = self.gate["dominant"]
        part = [a for a in self.gate["axes"].values()
                if a.get("role") == "partition" and a["measured"]]
        self.assertEqual(sorted(a["axis"] for a in part),
                         sorted(self.gate["decomposition"]["partition"]))
        best = max(a["number"]["share_of_fall"] for a in part)
        self.assertEqual(dom["share_of_fall"], best)
        self.assertGreaterEqual(dom["share_of_fall"], 0.5,
                                "a dominant axis carries most of the fall")

    def test_the_dominant_axis_publishes_its_robustness(self):
        """A share measured on one protocol says which cells it survives."""
        rob = self.gate["dominant"]["robustness"]
        cells = [v for arm in rob["cells"].values() for v in arm.values()]
        self.assertEqual(rob["pooled_share"],
                         round(sum(cells) / len(cells), 6))
        self.assertEqual(rob["range"],
                         [round(min(cells), 6), round(max(cells), 6)])
        won, total = rob["dominant_in_cells"].split(" of ")
        self.assertEqual(int(total), len(cells))
        self.assertEqual(int(won), sum(1 for v in cells if v > 0.5))
        self.assertIn(self.gate["dominant"]["measured_on"], rob["cells"])
        self.assertIn(rob["dominant_in_cells"], self.gate["verdict"])

    def test_the_second_protocol_decomposition_is_exact_too(self):
        second = self.gate["fall_second_protocol"]
        if not second.get("measured"):
            self.skipTest(second.get("reason", "not measured"))
        for arm, f in second["per_arm"].items():
            with self.subTest(arm=arm):
                self.assertAlmostEqual(
                    f["fall"], f["fall_ranking"] + f["fall_abstention"],
                    places=6)
                self.assertAlmostEqual(
                    f["share_ranking"] + f["share_abstention"], 1.0,
                    places=6)
                self.assertGreater(f["fall"], 0.0)

    def test_only_the_partition_axes_carry_a_share(self):
        for key, axis in self.gate["axes"].items():
            with self.subTest(axis=key):
                has_share = "share_of_fall" in (axis.get("number") or {})
                self.assertEqual(has_share, axis.get("role") == "partition")

    def test_an_unmeasured_axis_states_why_and_claims_nothing(self):
        for key, axis in self.gate["axes"].items():
            if axis["measured"]:
                continue
            with self.subTest(axis=key):
                self.assertTrue(axis.get("reason_not_measured"),
                                "measured:false needs a reason")
                self.assertNotIn("number", axis,
                                 "an unmeasured axis publishes no number")
                self.assertIn(key, self.gate["failed_criteria"])
        self.assertEqual(self.gate["pass"],
                         not self.gate["failed_criteria"])

    def test_the_priority_names_both_fix_tasks_and_what_flips_it(self):
        pri = self.gate["priority"]
        self.assertEqual({pri["first"], pri["second"]},
                         {"T-gen-objective", "T-unfreeze-backbone"})
        self.assertIn(str(self.gate["dominant"]["axis"]), pri["decided_by"])
        self.assertTrue(pri["flips_when"])

    def test_the_report_is_written_next_to_the_gate(self):
        path = os.path.join(A.ROOT, self.gate["report"])
        self.assertTrue(os.path.exists(path), path)
        with open(path) as fh:
            text = fh.read()
        self.assertIn(self.gate["dominant"]["name"], text)
        self.assertIn(self.gate["priority"]["first"], text)


class CrossChecksUpstream(unittest.TestCase):
    """The fall this gate ablates is the one #T-mix-5m published."""

    def test_matches_the_scaling_evidence_of_t_mix_5m(self):
        path = os.path.join(A.ROOT, "artifacts", "gates", "T-mix-5m",
                            "gate.json")
        with open(path) as fh:
            upstream = json.load(fh)["evidence"]["scaling_250k_to_1m"]
        gate = committed_gate()
        for run_id, ev in upstream.items():
            arm = ev["backbone"]
            with self.subTest(arm=arm):
                self.assertIn(arm, gate["fall"], run_id)
                f = gate["fall"][arm]
                self.assertEqual(f["unseen_accuracy"],
                                 [ev["at_250k"], ev["at_1m"]])
                self.assertEqual(f["unseen_abstain_rate"],
                                 [ev["abstain_at_250k"], ev["abstain_at_1m"]])
                self.assertAlmostEqual(f["fall"], -ev["delta"], places=6)


@unittest.skipUnless(runs_present(),
                     "artifacts/runs/ is gitignored: the curve runs "
                     f"{', '.join(A.CURVE_RUN.format(backbone=b, seed=A.ABLATION_SEED) for b in A.ARMS)} "
                     "are not on this machine")
class Rerun(unittest.TestCase):
    """Re-execute the ablation at the same seed; expect the same numbers."""

    def test_same_seed_reproduces_the_committed_gate(self):
        again = A.run(A.ABLATION_SEED, write=False)
        self.assertEqual(strip(again), strip(committed_gate()))

    def test_the_gate_survives_its_own_json_round_trip(self):
        """An int dict key or a tuple would compare equal to nothing
        after a write: the gate has to be its own fixture."""
        again = A.run(A.ABLATION_SEED, write=False)
        self.assertEqual(json.loads(json.dumps(again)), again)

    def test_twice_in_a_row_is_identical(self):
        a = A.run(A.ABLATION_SEED, write=False)
        b = A.run(A.ABLATION_SEED, write=False)
        self.assertEqual(strip(a), strip(b))

    def test_recorded_input_digests_still_match_the_files(self):
        for path, digest in committed_gate()["provenance"]["inputs"].items():
            full = os.path.join(A.ROOT, path)
            if not os.path.exists(full):
                continue
            with self.subTest(path=path):
                self.assertEqual(A.sha256_file(full), digest,
                                 f"{path} changed under the gate")

    def test_a_run_trained_at_another_seed_is_refused(self):
        with self.assertRaises(ValueError):
            A.load_run(A.CURVE_RUN.format(backbone=A.ARMS[0],
                                          seed=A.ABLATION_SEED),
                       A.ABLATION_SEED + 1, {})


class CrossCheckTUnseen(unittest.TestCase):
    """The release-protocol cross-check, re-derived from committed inputs."""

    @classmethod
    def setUpClass(cls):
        cls.gate = committed_gate()
        cls.cc = cls.gate["crosscheck_t_unseen"]

    def test_pooled_all_falls_under_both_protocols(self):
        for arm, lv in self.cc["all_level"].items():
            with self.subTest(arm=arm):
                self.assertGreater(lv["fall"], 0.0)
                self.assertAlmostEqual(
                    lv["share_ranking"] + lv["share_abstention"], 1.0,
                    places=6)

    def test_extreme_cell_is_fully_silent_yet_above_chance(self):
        ex = self.cc["extreme_cell"]
        self.assertEqual(ex["abstain_1m"], 1.0)
        self.assertTrue(ex["forced_above_chance_while_fully_silent"])

    def test_per_cut_falls_match_the_table(self):
        for arm, rows in self.cc["per_cut"].items():
            for cut, row in rows.items():
                with self.subTest(arm=arm, cut=cut):
                    self.assertAlmostEqual(
                        row["fall"], row["unseen_accuracy"][0]
                        - row["unseen_accuracy"][1], places=6)

    def test_inputs_are_recorded_in_provenance(self):
        inputs = self.gate["provenance"]["inputs"]
        for arm in A.ARMS:
            self.assertIn(
                os.path.join("artifacts", "gates", "T-antiscale-diag",
                             "inputs", A.T_UNSEEN_250K[arm]), inputs)


if __name__ == "__main__":
    unittest.main()
