"""Tests for decision-mix-clean-1m (#T-mix-1m gate, stdlib unittest).

Two properties carry most of the weight here:

* the mixer has NO mixture authority of its own — every cap, every keep
  decision and every count comes from `data/mix.py`, and a test proves
  the mixer's plan is the trainer's plan;
* the §§18/77 benchmark fence is an ASSERTION. A fenced row reaching the
  selection raises, it does not get counted and reported as a share.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from data import mix
from data.test_mix import write_corpus
from tools.mix_1m import backbones, diversity, run_mix, sampler
from tools.mix_1m.fence import (
    FENCED_DATASETS,
    Fence,
    FenceBreach,
    assert_spec_clean,
    load_blocklist,
    load_jevals_ids,
    state_hash,
)
from tools.mix_1m.strata import (
    LAYER_TARGETS,
    TYPE_TARGETS,
    is_hard,
    layer_report,
    layer_supply,
    layer_weights,
    question_type,
    tag,
    type_report,
)


def bool_q(qid="q1", ans="yes", conf=None):
    q = {"id": qid, "kind": "boolean", "answer": ans,
         "options": [{"id": "yes", "text": "Sí"}, {"id": "no", "text": "No"}]}
    if conf is not None:
        q["teacher_conf"] = conf
    return q


class TestTypes(unittest.TestCase):
    def test_boolean_is_noul(self):
        self.assertEqual(question_type("boolean", ["Sí", "No"]), "noul")

    def test_k2_choice_is_noul(self):
        self.assertEqual(question_type("choice", ["a", "b"]), "noul")

    def test_k4_choice_is_choice(self):
        self.assertEqual(question_type("choice", ["a", "b", "c", "d"]), "choice")

    def test_score_scale_is_score(self):
        self.assertEqual(
            question_type("choice", [f"score {i}" for i in range(5)]), "score")

    def test_tag_massive_lang(self):
        t = tag("massive", "[de-DE] wecke mich", bool_q())
        self.assertEqual(t["lang"], "de-DE")
        self.assertEqual(t["family"], "intent")
        self.assertEqual(t["origin"], "human")
        self.assertEqual(t["layer"], "intent-multi")

    def test_tag_layer_comes_from_the_one_registry(self):
        for dataset in mix.SOURCES:
            self.assertEqual(tag(dataset, "s", bool_q())["layer"],
                             mix.layer_of(dataset))


class TestHard(unittest.TestCase):
    def test_hard_distractor(self):
        q = bool_q()
        q["options"].append({"id": "d", "text": "Todo lo anterior es correcto."})
        self.assertTrue(is_hard(q, "grounded-synth"))

    def test_hard_low_conf(self):
        self.assertTrue(is_hard(bool_q(conf=0.55), "grounded-synth"))
        self.assertFalse(is_hard(bool_q(conf=0.9), "grounded-synth"))

    def test_adversarial_layer_is_hard_by_construction(self):
        self.assertTrue(is_hard(bool_q(), "adversarial"))

    def test_wide_k_is_hard(self):
        q = {"id": "q", "kind": "choice",
             "options": [{"id": str(i), "text": f"l{i}"} for i in range(9)]}
        self.assertTrue(is_hard(q, "intent-multi"))

    def test_prog_gold_difficulty_is_honoured(self):
        q = bool_q()
        q["quality"] = {"difficulty": "hard", "decider": "graph"}
        self.assertTrue(is_hard(q, "programmatic"))
        q["quality"]["difficulty"] = "easy"
        self.assertFalse(is_hard(q, "programmatic"))


class TestFence(unittest.TestCase):
    def test_fenced_datasets_rejected(self):
        f = Fence()
        for ds in ("banking77", "helpsteer2", "pubmedqa"):
            ok, reason = f.check(ds, "some state")
            self.assertFalse(ok, ds)
            self.assertIn("fenced", reason)

    def test_fenced_set_covers_gate(self):
        self.assertTrue({"banking77", "helpsteer2", "pubmedqa"}
                        <= FENCED_DATASETS)

    def test_the_clean_registry_has_no_fenced_source(self):
        self.assertEqual(assert_spec_clean(mix.clean_datasets()),
                         sorted(mix.clean_datasets()))
        for fenced in ("banking77", "helpsteer2"):
            self.assertNotIn(fenced, mix.clean_datasets())

    def test_a_fenced_source_in_the_mixture_raises(self):
        with self.assertRaises(FenceBreach) as ctx:
            assert_spec_clean(["massive", "banking77"])
        self.assertIn("banking77", str(ctx.exception))

    def test_jevals_id_blocked(self):
        f = Fence(jevals_ids={"jev-000000001"})
        ok, _ = f.check("massive", "state", "jev-000000001")
        self.assertFalse(ok)
        self.assertEqual(f.blocked_id, 1)

    def test_jevals_hash_blocked(self):
        f = Fence({state_hash("jevals banking probe")})
        self.assertFalse(f.check("massive", "jevals banking probe")[0])
        self.assertTrue(f.check("massive", "ordinary intent text")[0])

    def test_a_blocked_row_makes_assert_clean_raise(self):
        f = Fence({state_hash("probe")})
        f.check("massive", "probe")
        with self.assertRaises(FenceBreach):
            f.assert_clean()

    def test_a_clean_run_reports_and_does_not_raise(self):
        f = Fence()
        f.check("massive", "ordinary text", "q1")
        report = f.assert_clean()
        self.assertTrue(report["clean"])
        self.assertEqual(report["questions_checked"], 1)

    def test_missing_blocklist_is_empty_never_satisfied(self):
        self.assertEqual(load_blocklist("/nonexistent/jevals.json"), set())
        self.assertEqual(load_blocklist(None), set())
        self.assertEqual(load_jevals_ids("/nonexistent/ids.json"), set())

    def test_the_published_jevals_registry_is_read(self):
        ids = load_jevals_ids()
        if not ids:
            self.skipTest("#T-data-eval has not published jevals_ids.json")
        self.assertTrue(all(i.startswith("jev-") for i in list(ids)[:20]))


class TestGuardrails(unittest.TestCase):
    """The §§65-66 numbers are `data.mix`'s, and a breach is an error."""

    def good_counts(self):
        return {"dataset": {f"d{i}": 100 for i in range(10)},
                "family": {"a": 300, "b": 300, "c": 200, "d": 200},
                "origin": {"human": 600, "synthetic": 300, "programmatic": 100},
                "hard": 150}

    def test_pass(self):
        rep = sampler.verify_guardrails(self.good_counts(), 1000)
        self.assertTrue(all(v["ok"] for v in rep["checks"].values()))

    def test_dataset_over_15_fails_as_error(self):
        c = self.good_counts()
        c["dataset"] = {"civil-comments": 890, "rest": 110}
        with self.assertRaises(sampler.CapViolation):
            sampler.verify_guardrails(c, 1000)

    def test_civil_dominance_is_dataset_breach(self):
        c = self.good_counts()
        c["dataset"] = {"civil-comments": 500, "a": 250, "b": 250}
        with self.assertRaises(sampler.CapViolation):
            sampler.verify_guardrails(c, 1000)

    def test_family_over_30_fails(self):
        c = self.good_counts()
        c["family"] = {"toxicity": 400, "b": 600}
        with self.assertRaises(sampler.CapViolation):
            sampler.verify_guardrails(c, 1000)

    def test_synthetic_over_50_fails(self):
        c = self.good_counts()
        c["origin"] = {"human": 300, "synthetic": 600, "programmatic": 100}
        with self.assertRaises(sampler.CapViolation):
            sampler.verify_guardrails(c, 1000)

    def test_human_under_20_fails(self):
        c = self.good_counts()
        c["origin"] = {"human": 100, "synthetic": 500, "programmatic": 400}
        with self.assertRaises(sampler.CapViolation):
            sampler.verify_guardrails(c, 1000)

    def test_hard_under_10_fails(self):
        c = self.good_counts()
        c["hard"] = 50
        with self.assertRaises(sampler.CapViolation):
            sampler.verify_guardrails(c, 1000)

    def test_the_cap_violation_is_the_loaders_error(self):
        self.assertIs(sampler.CapViolation, mix.MixGuardrailError)

    def test_the_report_form_does_not_raise_but_names_the_failure(self):
        counts = {"dataset": {"a": 1000}, "family": {"f": 1000},
                  "origin": {"human": 1000}, "hard": 0, "rows": 1000}
        rep = sampler.guardrail_report(counts)
        self.assertFalse(rep["pass"])
        self.assertIn("hard>=10%", rep["failed"])
        self.assertIn("dataset:a<=15%", rep["failed"])


class TestFeasibility(unittest.TestCase):
    """The ceiling is arithmetic; the gate has to publish it, not guess it."""

    def test_the_fence_is_what_makes_1m_impossible(self):
        _scan, supply = sampler.clean_supply()
        feas = run_mix.corpus_feasibility(supply)
        self.assertLess(feas["achievable_rows"], run_mix.CORPUS_TARGET)
        self.assertFalse(feas["pass"])
        self.assertGreater(feas["shortfall_rows"], 0)
        self.assertIn("independent source", feas["why"])

    def test_a_target_over_the_ceiling_is_refused_with_the_arithmetic(self):
        supply = {"massive": 50_000, "huffpost": 50_000, "boolq": 9_000,
                  "civil-comments": 900_000, "synth-v1": 30_000,
                  "prog-gold": 50_000, "email-triage": 4_000}
        with self.assertRaises(sampler.SupplyShortfall) as ctx:
            sampler.assert_feasible(5_000_000, supply, cap_margin=0.0)
        self.assertIn("nowhere to come from", str(ctx.exception))

    def test_dropping_the_synthetic_source_breaks_cap_coverage(self):
        """Why the §128 baseline arm has to state a different cap."""
        clean = mix.clean_datasets()
        without = [d for d in clean if d != "synth-v1"]
        self.assertGreaterEqual(mix.cap_units(clean), 1.0)
        self.assertLess(mix.cap_units(without), 1.0)
        self.assertGreaterEqual(mix.cap_units(without, 0.18, 0.30), 1.0)


class TestOneAuthority(unittest.TestCase):
    """The mixer plans what the trainer trains, or the manifest is fiction."""

    def test_the_layer_plan_is_a_pull_not_a_cap(self):
        weights = layer_weights(mix.clean_datasets())
        self.assertAlmostEqual(
            weights["prog-gold"], LAYER_TARGETS["programmatic"])
        # four intent-multi sources in the registry, three of them clean
        self.assertAlmostEqual(weights["massive"] * 3,
                               LAYER_TARGETS["intent-multi"], places=6)

    def test_the_mixer_plan_is_the_trainer_plan(self):
        from training.python.train_decision import build_mix

        spec, _scan, _holdout = sampler.plan_clean(4000, seed=7)
        datasets = mix.clean_datasets()
        other, _s, _p, _h = build_mix(7, target=4000, datasets=datasets,
                                      cap_margin=0.0,
                                      weights=mix.layer_weights(datasets))
        self.assertEqual(spec.quotas, other.quotas)
        self.assertEqual(spec.keep_fractions, other.keep_fractions)

    def test_selection_uses_data_mix_and_nothing_else(self):
        """No second keep function may exist in the package."""
        import tools.mix_1m.sampler as mod
        self.assertFalse(hasattr(mod, "keep_prob"))
        self.assertIs(mod.verify_guardrails, mix.verify_guardrails)

    def test_layer_supply_sums_to_the_supply(self):
        supply = {"massive": 10, "prog-gold": 5, "boolq": 3}
        self.assertEqual(sum(layer_supply(supply).values()), 18)


class TestAssembleOnAFixture(unittest.TestCase):
    """The whole path on a tiny corpus: plan, fence, count, verify."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        for dataset in mix.clean_datasets():
            labels = (["toxic", "not_toxic"]
                      if not mix.SOURCES[dataset].dynamic_options
                      else [f"{dataset}_l{i}" for i in range(5)])
            write_corpus(self.root, dataset, 400, labels)
        scan = mix.scan_supply(mix.clean_datasets(), self.root,
                               cache_path=os.path.join(self.root, "c.json"))
        self.spec = mix.plan_mix(mix.effective_supply(scan), None, 11,
                                 scan=scan, cap_margin=0.0,
                                 weights=mix.layer_weights(
                                     mix.clean_datasets()))

    def tearDown(self):
        self.tmp.cleanup()

    def test_it_assembles_and_the_fence_is_clean(self):
        counts, fence = sampler.assemble(self.spec, self.root)
        self.assertGreater(counts["rows"], 0)
        self.assertTrue(counts["fence"]["clean"])
        self.assertEqual(counts["fence"]["blocked"]["dataset"], 0)
        self.assertEqual(fence.checked, counts["rows"])

    def test_the_fence_sees_every_row_the_caps_counted(self):
        counts, fence = sampler.assemble(self.spec, self.root)
        self.assertEqual(sum(counts["strata"]["dataset"].values()),
                         counts["rows"])
        self.assertEqual(counts["strata"]["dataset"], counts["dataset"])

    def selected_qids(self) -> list:
        """The ids the mixture actually selected, from the same pass."""
        seen = []
        mix.realise(self.spec, self.root,
                    lambda d, i, row, q: seen.append(f"{d}:train:{q['id']}"))
        return seen

    def test_a_jevals_id_in_the_selection_aborts_the_build(self):
        picked = self.selected_qids()
        self.assertTrue(picked)
        with self.assertRaises(FenceBreach) as ctx:
            sampler.assemble(self.spec, self.root,
                             jevals_ids={picked[len(picked) // 2]})
        self.assertIn("jevals id registry", str(ctx.exception))

    def test_a_jevals_id_that_is_NOT_in_the_selection_does_not_fire(self):
        counts, _f = sampler.assemble(
            self.spec, self.root, jevals_ids={"jev-000000001"})
        self.assertTrue(counts["fence"]["clean"])
        self.assertEqual(counts["fence"]["jevals_ids_known"], 1)

    def test_two_assemblies_agree_bit_for_bit(self):
        first, _a = sampler.assemble(self.spec, self.root)
        second, _b = sampler.assemble(self.spec, self.root)
        self.assertEqual(first["members_sha256"], second["members_sha256"])
        self.assertEqual(first["strata"], second["strata"])

    def test_the_token_ledger_is_per_shard(self):
        counts, _f = sampler.assemble(self.spec, self.root)
        ledger = diversity.token_ledger(counts)
        self.assertEqual(ledger["total"]["examples"], counts["rows"])
        self.assertGreater(ledger["total"]["total_tokens"], 0)
        self.assertTrue(ledger["by_shard"])
        for shard in ledger["by_shard"].values():
            self.assertIn("mean_k", shard)
            self.assertGreater(shard["examples"], 0)

    def test_the_dashboard_publishes_every_axis(self):
        counts, _f = sampler.assemble(self.spec, self.root)
        dash = diversity.full_dashboard(counts)
        for axis in ("dataset", "family", "layer", "qtype", "lang",
                     "k_bucket", "origin"):
            self.assertIn(axis, dash["axes"])
            self.assertAlmostEqual(sum(dash["axes"][axis].values()), 1.0,
                                   places=6)
        self.assertIn("simpson", dash["axes"])
        self.assertEqual(set(dash["layer_plan"]["rows"]) >= set(LAYER_TARGETS),
                         True)
        self.assertEqual(set(dash["type_mix"]["rows"]) >= set(TYPE_TARGETS),
                         True)


class TestLayerAndTypeReports(unittest.TestCase):
    def test_an_unsupplied_layer_is_published_not_dropped(self):
        rep = layer_report({"layer": {"programmatic": 100}, "rows": 100})
        self.assertIn("nli", rep["rows"])
        self.assertEqual(rep["rows"]["nli"]["n"], 0)
        self.assertIn("nli", rep["unsupplied"])
        self.assertIn("preference", rep["why_unsupplied"])

    def test_the_type_gap_is_signed(self):
        rep = type_report({"qtype": {"choice": 55, "noul": 45}, "rows": 100})
        self.assertAlmostEqual(rep["rows"]["choice"]["gap"], 0.0, places=6)
        self.assertAlmostEqual(rep["rows"]["noul"]["gap"], 0.15, places=6)
        self.assertAlmostEqual(rep["rows"]["score"]["gap"], -0.15, places=6)


class TestBackbones(unittest.TestCase):
    """Top-2 comes from the measured Pareto; a proxy row is refused."""

    def report(self, **over):
        base = {"top2": ["modernbert-base", "ettin-68m"],
                "pareto": {"rows": [
                    {"id": "modernbert-base", "status": "trained",
                     "lineage": {"run_id": "r1"}, "quality": {}},
                    {"id": "ettin-68m", "status": "trained",
                     "lineage": {"run_id": "r2"}, "quality": {}}]}}
        base.update(over)
        return base

    def test_top2_is_read_from_the_bakeoff(self):
        self.assertEqual(backbones.top2(self.report()),
                         ["modernbert-base", "ettin-68m"])

    def test_a_proxy_row_is_refused(self):
        rep = self.report()
        rep["pareto"]["rows"][0]["pending_weights"] = ["x"]
        with self.assertRaises(backbones.NoBakeoff):
            backbones.top2(rep)

    def test_a_row_without_lineage_is_refused(self):
        rep = self.report()
        rep["pareto"]["rows"] = rep["pareto"]["rows"][1:]
        with self.assertRaises(backbones.NoBakeoff):
            backbones.top2(rep)

    def test_the_published_report_names_two_trained_backbones(self):
        if not os.path.exists(backbones.BAKEOFF):
            self.skipTest("#T-bakeoff-real has not published a report")
        self.assertEqual(len(backbones.top2()), 2)

    def test_the_train_command_is_the_real_trainer(self):
        cmd = backbones.train_command("ettin-68m", "run-x", 250_000)
        self.assertIn("training.python.train_decision", cmd)
        self.assertIn("--fence-clean", cmd)
        self.assertIn("train", cmd)

    def test_a_missing_stage_is_pending_never_imputed(self):
        row = backbones.backbone_report("ettin-68m", "no-such-run")
        for stage in row["stages"].values():
            self.assertEqual(stage["status"], "pending")
            self.assertNotIn("unseen", stage)
        self.assertFalse(row["complete"])

    def test_a_pending_synthetic_verdict_has_no_verdict(self):
        out = backbones.synthetic_verdict("none-a", "none-b", 250_000)
        self.assertEqual(out["status"], "pending")
        self.assertIsNone(out["verdict"])


class TestSyntheticVerdict(unittest.TestCase):
    """§128: the rule is stated, and NO-GO is a valid outcome."""

    def _run(self, tmp, run_id, nll, acc):
        d = os.path.join(tmp, run_id)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "metrics.jsonl"), "w") as fh:
            fh.write(json.dumps({
                "t": "stage", "stage": 250000, "samples": 250000,
                "unseen": {"nll": nll, "accuracy": acc, "brier": 0.5,
                           "ece": 0.1},
                "seen": {"nll": nll, "accuracy": acc, "brier": 0.5,
                         "ece": 0.1}}) + "\n")

    def test_no_gain_is_a_no_go_with_a_fallback(self):
        tmp = tempfile.mkdtemp()
        old = backbones.RUNS
        try:
            backbones.RUNS = tmp
            self._run(tmp, "base", 1.0, 0.30)
            self._run(tmp, "plus", 1.2, 0.28)
            out = backbones.synthetic_verdict("base", "plus", 250000)
            self.assertEqual(out["verdict"], "NO-GO-fix-prompts")
            self.assertIn("do not scale", out["fallback"])
            self.assertLess(out["gain"]["unseen_nll"], 0)
        finally:
            backbones.RUNS = old

    def test_a_real_gain_is_a_go(self):
        tmp = tempfile.mkdtemp()
        old = backbones.RUNS
        try:
            backbones.RUNS = tmp
            self._run(tmp, "base", 1.2, 0.28)
            self._run(tmp, "plus", 1.0, 0.31)
            out = backbones.synthetic_verdict("base", "plus", 250000)
            self.assertEqual(out["verdict"], "GO-scale")
            self.assertIsNone(out["fallback"])
        finally:
            backbones.RUNS = old


class TestPublishedGate(unittest.TestCase):
    """What the gate on disk claims, checked against the task's "Done when"."""

    def setUp(self):
        if not os.path.exists(run_mix.GATE):
            self.skipTest("gate not built: python -m tools.mix_1m.run_mix build")
        with open(run_mix.GATE) as fh:
            self.gate = json.load(fh)

    def test_the_gate_states_pass_or_fail_and_why(self):
        self.assertIn("pass", self.gate)
        if not self.gate["pass"]:
            self.assertTrue(self.gate["failed_checks"])
            for name in self.gate["failed_checks"]:
                self.assertIn(name, self.gate["checks"])

    def test_the_fence_is_asserted_clean(self):
        self.assertTrue(self.gate["fence"]["clean"])
        self.assertEqual(self.gate["fence"]["blocked"],
                         {"dataset": 0, "jevals_id": 0, "jevals_hash": 0})
        self.assertTrue({"banking77", "helpsteer2", "pubmedqa"}
                        <= set(self.gate["fence"]["fenced_datasets"]))

    def test_no_fenced_source_is_in_the_composition(self):
        for fenced in ("banking77", "helpsteer2", "pubmedqa"):
            self.assertNotIn(fenced, self.gate["composition"]["by_dataset"])

    def test_no_dataset_over_fifteen_percent(self):
        rows = self.gate["composition"]["rows"]
        for dataset, n in self.gate["composition"]["by_dataset"].items():
            self.assertLessEqual(n / rows, mix.MAX_DATASET_FRACTION + 1e-9,
                                 dataset)

    def test_the_token_ledger_is_published_per_shard(self):
        self.assertGreater(self.gate["total_tokens"], 0)
        self.assertTrue(self.gate["tokens"]["by_shard"])

    def test_the_corpus_is_reproducible_from_seed_and_manifest(self):
        self.assertTrue(self.gate["mix"]["members_sha256"])
        self.assertTrue(os.path.exists(
            os.path.join(mix.ROOT, self.gate["mix"]["manifest"])))

    def test_every_measured_metric_is_a_number_or_pending(self):
        for row in self.gate["training"].get("backbones", {}).values():
            for stage in row["stages"].values():
                if stage["status"] == "pending":
                    continue
                for metric in backbones.METRICS:
                    self.assertIsInstance(stage["unseen"][metric], (int, float))


if __name__ == "__main__":
    unittest.main()
