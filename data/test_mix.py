"""Tests for the corpus mix descriptor (#T-corpus-rebalance, stdlib only).

The task's verification gate is three claims, and each one is a test here
rather than a number in a report:

1. a mixture with `civil-comments` at 74 % fails with an explicit error
   that names the dataset and its share — and it fails in the LOADER, not
   only in a checker nobody calls;
2. the manifest reproduces the mixture bit for bit from seed + shard shas,
   with no member list in between;
3. the composition the gate publishes is the one the caps allow: no
   dataset over 15 %, no family over 30 %, dynamic-option rows a majority.

Run:
    python3 -m unittest data.test_mix -v
"""
from __future__ import annotations

import ast
import json
import os
import tempfile
import unittest

from . import mix
from .mix import (MixGuardrailError, MixInfeasible, MixSpec, SOURCES,
                  allocate, build_manifest, cap_units, check_composition,
                  composition_report, effective_supply, keep_row,
                  max_feasible_target, plan_mix, realise, spec_from_manifest,
                  verify_mix)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: the real supply of the converted corpus on 2026-09-21 (train split,
#: questions), so the arithmetic below is tested against real numbers
REAL_SUPPLY = {"banking77": 10_003, "boolq": 9_427, "civil-comments": 1_804_874,
               "email-triage": 4_000, "helpsteer2": 101_620,
               "huffpost": 162_935, "massive": 69_084, "synth-v1": 34_847}


def write_corpus(root: str, dataset: str, n_rows: int, labels: list,
                 split: str = "train", question_id: str | None = None,
                 kind: str = "choice") -> None:
    """A tiny V1 corpus. `question_id` fixed = the civil-comments shape."""
    path = os.path.join(root, f"{dataset}.jsonl")
    with open(path, "a") as fh:
        for i in range(n_rows):
            gold = labels[i % len(labels)]
            qid = question_id or f"{dataset}-q-{i}"
            fh.write(json.dumps({
                "state": f"{dataset} state number {i}",
                "split": split,
                "questions": [{
                    "id": qid, "kind": kind, "answer": gold,
                    "options": [{"id": lab, "text": lab} for lab in labels]}],
            }) + "\n")


class Guardrails(unittest.TestCase):
    """The caps are an error, and the error says which and how much."""

    def test_civil_comments_at_74_percent_aborts(self):
        """Finding E's own numbers: 1 999 514 of 2 681 195 examples."""
        with self.assertRaises(MixGuardrailError) as ctx:
            verify_mix({"civil-comments": 1_999_514}, 2_681_195)
        message = str(ctx.exception)
        self.assertIn("civil-comments", message)
        self.assertIn("74.", message)          # the share, not just a flag
        self.assertIn("1,999,514", message)    # and the count behind it
        self.assertIn("15.00 %", message)      # and the cap it broke

    def test_a_family_over_thirty_percent_aborts(self):
        # every dataset under 15 %, but three intent sources add up to 45 %
        counts = {"banking77": 150, "massive": 150, "email-triage": 150,
                  "huffpost": 140, "boolq": 140, "civil-comments": 140,
                  "helpsteer2": 130}
        with self.assertRaises(MixGuardrailError) as ctx:
            verify_mix(counts)
        message = str(ctx.exception)
        self.assertIn("family 'intent'", message)
        self.assertIn("30.00 %", message)
        self.assertNotIn("dataset 'banking77'", message)

    def test_a_balanced_mixture_passes(self):
        counts = {d: 100 for d in SOURCES if d != "email-triage"}
        report = verify_mix(counts)
        self.assertEqual(report["total"], 100 * len(counts))
        self.assertTrue(all(c["ok"] for c in report["checks"].values()))

    def test_empty_mixture_is_an_error_not_a_pass(self):
        with self.assertRaises(MixGuardrailError):
            verify_mix({})


class Allocator(unittest.TestCase):
    def test_every_quota_is_inside_both_caps(self):
        target = max_feasible_target(REAL_SUPPLY)
        quota = allocate(target, REAL_SUPPLY)
        verify_mix(quota, sum(quota.values()))  # raises if it is not

    def test_the_ceiling_is_a_ceiling(self):
        target = max_feasible_target(REAL_SUPPLY)
        self.assertGreater(target, 0)
        with self.assertRaises(MixInfeasible):
            allocate(target + 1, REAL_SUPPLY)

    def test_supply_never_over_drawn(self):
        quota = allocate(max_feasible_target(REAL_SUPPLY), REAL_SUPPLY)
        for dataset, n in quota.items():
            self.assertLessEqual(n, REAL_SUPPLY[dataset], dataset)

    def test_dropping_a_source_makes_the_caps_unsatisfiable(self):
        """Seven cap units are needed; these six cover 90 % of any target."""
        supply = {d: n for d, n in REAL_SUPPLY.items() if d != "helpsteer2"}
        self.assertLess(cap_units(list(supply)), 1.0)
        with self.assertRaises(MixInfeasible) as ctx:
            allocate(10_000, supply)
        self.assertIn("no target size fixes this", str(ctx.exception))

    def test_an_unregistered_source_is_refused(self):
        with self.assertRaises(MixInfeasible):
            allocate(100, {"banking77": 500, "logiqa": 500})

    def test_weights_favour_dynamic_options_when_caps_do_not_bind(self):
        supply = {d: 1_000_000 for d in SOURCES}
        quota = allocate(20_000, supply)
        self.assertGreater(quota["banking77"], quota["civil-comments"])
        self.assertGreater(quota["huffpost"], quota["boolq"])


class Selection(unittest.TestCase):
    def test_keep_is_deterministic_and_not_salted(self):
        a = keep_row(7, "civil-comments", 42, "civil-toxicity-all-0", 0.5)
        b = keep_row(7, "civil-comments", 42, "civil-toxicity-all-0", 0.5)
        self.assertEqual(a, b)
        self.assertTrue(keep_row(7, "x", 1, "q", 1.0))
        self.assertFalse(keep_row(7, "x", 1, "q", 0.0))

    def test_a_shared_question_id_still_splits_by_row(self):
        """The civil-comments shape: one question id on every row."""
        kept = sum(keep_row(7, "civil-comments", i, "civil-toxicity-all-0",
                            0.1) for i in range(5000))
        self.assertGreater(kept, 350)   # ~500 expected, never 0
        self.assertLess(kept, 650)      # and never "all of it"

    def test_the_rate_is_the_rate(self):
        for fraction in (0.05, 0.25, 0.6):
            kept = sum(keep_row(11, "d", i, "q", fraction)
                       for i in range(20_000))
            self.assertAlmostEqual(kept / 20_000, fraction, delta=0.02)


class ReproducibleManifest(unittest.TestCase):
    """Seed + shard shas rebuild the mixture; no member list is shipped."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        write_corpus(self.root, "banking77", 400,
                     ["card_arrival", "card_lost", "top_up", "fee", "pin"])
        write_corpus(self.root, "civil-comments", 2000, ["toxic", "not_toxic"],
                     question_id="civil-toxicity-all-0", kind="boolean")
        self.spec = MixSpec(seed=99, target=500,
                            quotas={"banking77": 300, "civil-comments": 200},
                            supply={"banking77": 400, "civil-comments": 2000},
                            keep_fractions={"banking77": 0.75,
                                            "civil-comments": 0.1})

    def tearDown(self):
        self.tmp.cleanup()

    def test_two_realisations_agree_bit_for_bit(self):
        first = realise(self.spec, self.root)
        second = realise(self.spec, self.root)
        self.assertEqual(first["members_sha256"], second["members_sha256"])
        self.assertEqual(first["dataset"], second["dataset"])

    def test_a_manifest_round_trip_rebuilds_the_same_members(self):
        counts = realise(self.spec, self.root)
        manifest = build_manifest(self.spec, counts)
        rebuilt = spec_from_manifest(json.loads(json.dumps(manifest)))
        self.assertEqual(realise(rebuilt, self.root)["members_sha256"],
                         manifest["members_sha256"])

    def test_another_seed_is_another_mixture(self):
        other = MixSpec(**{**self.spec.to_dict(), "seed": 100})
        self.assertNotEqual(realise(self.spec, self.root)["members_sha256"],
                            realise(other, self.root)["members_sha256"])

    def test_the_kept_counts_land_on_the_quota(self):
        counts = realise(self.spec, self.root)
        self.assertAlmostEqual(counts["dataset"]["banking77"], 300, delta=25)
        self.assertAlmostEqual(counts["dataset"]["civil-comments"], 200,
                               delta=30)

    def test_a_manifest_from_another_registry_version_is_refused(self):
        manifest = build_manifest(self.spec, realise(self.spec, self.root))
        manifest["version"] = "decision-mix-v1"
        with self.assertRaises(ValueError):
            spec_from_manifest(manifest)


class LoaderAborts(unittest.TestCase):
    """The hard error has to fire where the batches are built."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        write_corpus(self.root, "banking77", 40,
                     ["card_arrival", "card_lost", "top_up", "fee", "pin"])
        write_corpus(self.root, "civil-comments", 400, ["toxic", "not_toxic"],
                     question_id="civil-toxicity-all-0", kind="boolean")

    def tearDown(self):
        self.tmp.cleanup()

    def _samplers(self):
        from .optset import OptionSetSampler, SamplerConfig
        return {d: OptionSetSampler(datasets=(d,), config=SamplerConfig(),
                                    root=self.root)
                for d in ("banking77", "civil-comments")}

    def test_mixture_stream_refuses_a_dominated_corpus(self):
        from training.python.train_decision import MixtureStream

        with self.assertRaises(MixGuardrailError) as ctx:
            MixtureStream(self._samplers(), batch_size=8)
        message = str(ctx.exception)
        self.assertIn("civil-comments", message)
        self.assertIn("90.91 %", message)  # 400 of 440

    def test_the_same_stream_is_allowed_when_asked_not_to_verify(self):
        from training.python.train_decision import MixtureStream

        stream = MixtureStream(self._samplers(), batch_size=8, verify=False)
        self.assertEqual(stream.total(), 440)
        self.assertIsNone(stream.guardrails)

    def test_a_capped_selection_passes_the_same_check(self):
        """The same loader, fed the descriptor, builds the stream fine."""
        from training.python.train_decision import MixtureStream
        from .optset import OptionSetSampler, SamplerConfig

        root = tempfile.mkdtemp()
        for dataset in SOURCES:
            labels = (["toxic", "not_toxic"]
                      if not SOURCES[dataset].dynamic_options
                      else [f"{dataset}_l{i}" for i in range(5)])
            write_corpus(root, dataset, 600, labels)
        scan = mix.scan_supply(root=root,
                               cache_path=os.path.join(root, "cache.json"))
        spec = plan_mix(effective_supply(scan), seed=3, scan=scan)
        # the quota is half the contract: the selector draws with
        # `DRAW_HEADROOM` and the loader trims, which is what keeps the
        # realised share ON the planned one instead of binomially above it
        samplers = {
            d: OptionSetSampler(datasets=(d,), config=SamplerConfig(),
                                root=root, keep={d: spec.keep(d)},
                                quotas={d: spec.quotas[d]})
            for d in spec.datasets}
        stream = MixtureStream(samplers, batch_size=8)
        self.assertIsNotNone(stream.guardrails)
        self.assertFalse(stream.guardrails["breaches"])
        self.assertEqual(sorted(stream.sizes),
                         sorted(mix.default_datasets()))


class PublishedComposition(unittest.TestCase):
    """What the gate publishes, checked against the task's "Done when"."""

    def setUp(self):
        path = mix.manifest_path()
        if not os.path.exists(path):
            self.skipTest("mixture not built yet: python3 -m data.mix build")
        self.manifest = mix.load_manifest(path)

    def test_no_dataset_over_fifteen_percent(self):
        composition = self.manifest["composition"]
        for dataset, n in composition["by_dataset"].items():
            self.assertLessEqual(n / composition["rows"],
                                 mix.MAX_DATASET_FRACTION + 1e-9, dataset)

    def test_no_family_over_thirty_percent(self):
        composition = self.manifest["composition"]
        for family, n in composition["by_family"].items():
            self.assertLessEqual(n / composition["rows"],
                                 mix.MAX_FAMILY_FRACTION + 1e-9, family)

    def test_dynamic_options_are_the_published_majority(self):
        self.assertGreater(self.manifest["composition"]["dynamic_option_share"],
                           0.5)

    def test_every_axis_of_the_diversity_index_is_published(self):
        simpson = self.manifest["diversity"]["simpson"]
        self.assertEqual(set(simpson), {"dataset", "family", "origin",
                                        "qtype", "k_bucket", "lang"})
        self.assertGreater(simpson["dataset"], 0.8)  # 0 would be a monopoly

    def test_every_source_pins_its_shards(self):
        for dataset, source in self.manifest["sources"].items():
            self.assertTrue(source["shards"], dataset)
            for shard in source["shards"]:
                self.assertEqual(len(shard["sha256"]), 64, dataset)


class JobsRegistry(unittest.TestCase):
    """`JOBS` samples every source, at the descriptor's quota."""

    def jobs(self) -> list:
        # AST, so the test never imports sklearn (same trick as test_firewall)
        with open(os.path.join(ROOT, "data", "train_baseline.py")) as fh:
            tree = ast.parse(fh.read())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "JOBS"
                    for t in node.targets):
                return [ast.literal_eval(elt) for elt in node.value.elts]
        raise AssertionError("JOBS not found in data/train_baseline.py")

    def test_sampling_is_the_rule_not_the_exception(self):
        for job in self.jobs():
            self.assertIsNotNone(job["sample"], job["name"])

    def test_every_job_is_a_registered_source(self):
        for job in self.jobs():
            self.assertIn(job["name"], SOURCES, job["name"])

    def test_the_quotas_match_the_descriptor(self):
        planned = plan_mix(REAL_SUPPLY).quotas
        for job in self.jobs():
            self.assertEqual(job["sample"], planned[job["name"]], job["name"])

    def test_civil_comments_has_no_privileged_quota(self):
        """Finding E in one assertion: it gets what the others get."""
        by_name = {job["name"]: job["sample"] for job in self.jobs()}
        self.assertEqual(by_name["civil-comments"], by_name["huffpost"])
        self.assertEqual(by_name["civil-comments"], by_name["massive"])

    def test_no_job_draws_more_than_the_descriptor_grants(self):
        """The baseline reads one file per job, so it has no `synth-v1`.

        Seven sources cannot satisfy the 15 % cap on their own (six cap
        units, 90 % coverage) — that is the product path's mixture, not
        this superseded trainer's. What IS enforced here: no job takes
        more than `data.mix` grants it, so none of them can grow back.
        """
        planned = plan_mix(REAL_SUPPLY).quotas
        for job in self.jobs():
            self.assertLessEqual(job["sample"], planned[job["name"]],
                                 job["name"])


class SupplyAndComposition(unittest.TestCase):
    def test_holdout_shrinks_the_supply_it_plans_against(self):
        scan = {"banking77": {"train_questions": 100,
                              "gold_counts": {"a": 60, "b": 40}}}
        self.assertEqual(effective_supply(scan), {"banking77": 100})
        self.assertEqual(effective_supply(scan, {"banking77": ["a"]}),
                         {"banking77": 60})

    def test_composition_report_names_the_biggest_source(self):
        counts = {"rows": 100, "dataset": {"a": 60, "b": 40}, "dynamic": 40}
        report = composition_report(counts)
        self.assertEqual(report["top_dataset_share"], 0.6)
        self.assertEqual(report["dynamic_option_share"], 0.4)

    def test_check_composition_fails_a_fixed_label_majority(self):
        counts = {"rows": 800, "dynamic": 300,
                  "dataset": {d: 100 for d in SOURCES},
                  "family": {}, "origin": {"human": 800}}
        report = check_composition(counts)
        self.assertFalse(report["pass"])
        self.assertFalse(
            report["checks"]["dynamic_options_are_the_majority"]["pass"])
        # the family axis is rebuilt from the registry: intent's three
        # sources are 37.5 % here, so the cap check fails with it
        self.assertFalse(report["checks"]["guardrails_hold"]["pass"])


if __name__ == "__main__":
    unittest.main()
