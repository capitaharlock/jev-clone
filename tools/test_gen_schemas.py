"""Tests for the decision-schema generator (#T-gen-schemas)."""
import json
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.firewall import ContaminationScanner, canary_texts  # noqa: E402
from data.schema import validate  # noqa: E402
from tools.gen_schemas import (Budget, DOMAIN_IDS, LANGS, MAX_K,  # noqa: E402
                               MIN_K, SchemaDeduper, build, diversity,
                               domain, generate, normalized_text)
from tools.gen_schemas import gate as gate_mod  # noqa: E402
from tools.gen_schemas import llm  # noqa: E402
from tools.gen_schemas.vocab import t  # noqa: E402

QUIET = lambda *a, **k: None  # noqa: E731


def small_budget(per_cell=2, ks=(3, 5, 8)):
    return Budget.plan(per_cell=per_cell, ks=ks)


class SchemaShapeTest(unittest.TestCase):
    def test_options_are_k_and_texts_not_positions(self):
        rng = random.Random(1)
        for k in range(MIN_K, MAX_K + 1):
            schema, why = build(domain("it-incident"), "en", k, rng,
                                unknown_rate=0.0)
            self.assertIsNotNone(schema, why)
            self.assertEqual(len(schema.options), k)
            ids = [o["id"] for o in schema.options]
            self.assertEqual(len(set(ids)), k)
            # ids are value keys, never opt0..optK-1 (audit finding B)
            self.assertTrue(all("." in i for i in ids), ids)
            self.assertTrue(all(o["text"].strip() for o in schema.options))

    def test_universal_schema_row_validates(self):
        res = generate(small_budget(1), seed=7, scanner=None, log=QUIET)
        self.assertTrue(res.schemas)
        for s in res.schemas[:50]:
            self.assertEqual(validate(s.to_example()), [])
            row = s.to_row()
            json.dumps(row)  # must be serializable as-is
            self.assertEqual(row["questions"][0]["answer"], s.answer)
            self.assertIn(s.question, row["state"])

    def test_hard_negatives_are_same_group_values(self):
        rng = random.Random(5)
        dom = domain("support-routing")
        fld = dom.field("priority")
        found = False
        for _ in range(200):
            schema, _ = build(dom, "es", 6, rng, unknown_rate=0.0)
            if schema is None or schema.queried_field != "priority":
                continue
            found = True
            group = fld.group_of(schema.answer)
            for key in schema.hard_negatives:
                self.assertEqual(fld.group_of(key), group)
                self.assertNotEqual(key, schema.answer)
        self.assertTrue(found, "no priority schema produced")


class UnknownTest(unittest.TestCase):
    def test_answer_absent_from_state_is_labelled_unknown(self):
        rng = random.Random(11)
        seen = 0
        for _ in range(300):
            schema, _ = build(domain("finance-approval"), "fr", 4, rng,
                              unknown_rate=1.0)
            if schema is None:
                continue
            seen += 1
            self.assertTrue(schema.unknown)
            self.assertEqual(schema.answer, "unknown")
            # every option text, including the one that WOULD be right, is
            # missing from the state: the state simply does not say.
            for opt in schema.options:
                self.assertNotIn(opt["text"].lower(), schema.state.lower())
        self.assertGreater(seen, 100)

    def test_answerable_schema_keeps_its_answer_in_the_state(self):
        rng = random.Random(12)
        for _ in range(200):
            schema, _ = build(domain("logistics"), "de", 5, rng,
                              unknown_rate=0.0)
            if schema is None:
                continue
            self.assertFalse(schema.unknown)
            self.assertNotEqual(schema.answer, "unknown")
            self.assertIn(t(schema.answer, "de").lower(), schema.state.lower())

    def test_unknown_fraction_is_controlled(self):
        res = generate(small_budget(4), seed=3, unknown_rate=0.25,
                       scanner=None, log=QUIET)
        rate = sum(1 for s in res.schemas if s.unknown) / len(res.schemas)
        self.assertGreater(rate, 0.15)
        self.assertLess(rate, 0.35)


class DedupTest(unittest.TestCase):
    def test_near_duplicate_is_rejected_and_counted(self):
        d = SchemaDeduper(threshold=0.7)
        base = ("prioridad: alta cola de destino: facturacion canal de entrada: "
                "correo electronico estado: abierto")
        self.assertEqual(d.add(base)[0], True)
        # one word changed out of sixteen: still the same schema
        near = base.replace("estado: abierto", "estado: abierto.")
        accepted, sim = d.add(near)
        self.assertFalse(accepted)
        self.assertGreaterEqual(sim, 0.7)
        self.assertEqual(d.n_rejected, 1)
        self.assertGreater(d.rejection_rate(), 0.0)

    def test_churned_proper_nouns_do_not_hide_a_duplicate(self):
        """The exact illusion of finding C: same schema, new name and figures."""
        rng = random.Random(21)
        schema, why = build(domain("support-routing"), "es", 4, rng,
                            unknown_rate=0.0)
        self.assertIsNotNone(schema, why)
        ref, who = schema.entities
        churned_state = schema.state.replace(ref, "ZZZ-98765").replace(
            who, "Persona Distinta")
        self.assertNotEqual(churned_state, schema.state)
        churned = normalized_text(churned_state, schema.question,
                                  ("ZZZ-98765", "Persona Distinta"))

        d = SchemaDeduper(threshold=0.7)
        self.assertTrue(d.add(schema.norm_text)[0])
        accepted, sim = d.add(churned)
        self.assertFalse(accepted, "a renamed copy passed as a new schema")
        self.assertEqual(sim, 1.0)

    def test_generator_counts_duplicate_rejections(self):
        # threshold 0.01 makes almost everything a "duplicate": the loop must
        # then stop on its own instead of spinning, and say so.
        res = generate(small_budget(2), seed=9, dedup_threshold=0.01,
                       scanner=None, log=QUIET)
        self.assertGreater(res.rejections["duplicate"], 0)
        self.assertIn("no candidate accepted", res.stopped)


class BudgetTest(unittest.TestCase):
    def test_quota_is_never_exceeded_and_run_stops_when_full(self):
        budget = small_budget(per_cell=3)
        res = generate(budget, seed=4, scanner=None, log=QUIET)
        self.assertEqual(budget.overflow(), {})
        self.assertTrue(budget.report()["full"])
        self.assertEqual(len(res.schemas), budget.total())
        self.assertEqual(res.stopped, "quota full")
        for cell in budget.cells():
            self.assertEqual(budget.filled[cell], budget.quota[cell])
            self.assertFalse(budget.take(cell))  # full means full

    def test_subset_budget_only_generates_its_cells(self):
        budget = Budget.plan(per_cell=2, domains=("logistics",),
                             languages=("en",), ks=(4,))
        res = generate(budget, seed=6, scanner=None, log=QUIET)
        self.assertEqual(len(res.schemas), 2)
        self.assertEqual({s.domain for s in res.schemas}, {"logistics"})
        self.assertEqual({s.language for s in res.schemas}, {"en"})
        self.assertEqual({s.k for s in res.schemas}, {4})

    def test_k_outside_family_size_is_refused(self):
        with self.assertRaises(ValueError):
            Budget.plan(per_cell=1, ks=(MAX_K + 1,))


class DiversityTest(unittest.TestCase):
    def test_tiny_quota_run_covers_every_domain_and_language(self):
        res = generate(small_budget(1), seed=2, scanner=None, log=QUIET)
        report = diversity(res.schemas, res.rejections, res.deduper)
        self.assertGreaterEqual(len(report["domains"]), 5)
        self.assertEqual(set(report["domains"]), set(DOMAIN_IDS))
        self.assertGreaterEqual(len(report["languages"]), 2)
        self.assertEqual(set(report["languages"]), set(LANGS))
        self.assertGreaterEqual(len(report["k_distribution"]), 3)
        self.assertGreater(len(report["state_formats"]), 1)

    def test_skeletons_track_content_not_row_count(self):
        res = generate(small_budget(3), seed=8, scanner=None, log=QUIET)
        report = diversity(res.schemas, res.rejections, res.deduper)
        # the old corpus scored 190 / 258700 = 0.0007 here
        self.assertGreater(report["skeletons_per_schema"], 0.9)
        self.assertEqual(report["skeleton_curve"][-1]["schemas"],
                         len(res.schemas))


class FirewallTest(unittest.TestCase):
    def test_generated_corpus_is_clean_against_the_canary_corpus(self):
        res = generate(small_budget(2), seed=13,
                       scanner=ContaminationScanner(), log=QUIET)
        self.assertTrue(res.schemas)
        self.assertEqual(res.rejections["firewall"], 0)

    def test_benchmark_content_is_rejected_not_written(self):
        class Trap:
            """Stands in for a schema that leaked benchmark content."""
            def scan(self, text):
                return True, "canary hit (test double)"

        res = generate(small_budget(1), seed=14, scanner=Trap(), log=QUIET)
        self.assertEqual(res.schemas, [])
        self.assertGreater(res.rejections["firewall"], 0)

    def test_the_wired_scanner_really_catches_a_canary(self):
        hit, reason = ContaminationScanner().scan(canary_texts()[0])
        self.assertTrue(hit, reason)


class ProposerTest(unittest.TestCase):
    def test_unconfigured_endpoint_stubs_and_falls_back(self):
        lines = []
        p = llm.RecordProposer(log=lines.append)
        self.assertFalse(llm.configured())
        self.assertEqual(p.propose(domain("it-incident"), "es", 8), [])
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("STUB: would call "))
        self.assertIn("it-incident/es", lines[0])
        p.propose(domain("it-incident"), "es", 8)  # announced once per cell
        self.assertEqual(len(lines), 1)
        self.assertEqual(p.report()["configured"], False)
        self.assertEqual(p.n_stubbed, 2)

    def test_configured_endpoint_parses_and_drops_off_vocabulary(self):
        dom = domain("logistics")
        good = {f.name: f.values()[0][0] for f in dom.fields}
        payload = json.dumps([good, dict(good, status="status.invented"), 42])
        p = llm.RecordProposer(log=QUIET)
        p._post = lambda prompt: f"here you go: {payload}"
        try:
            llm.os.environ[llm.ENDPOINT_ENV] = "https://example.invalid/v1/chat"
            recs = p.propose(dom, "en", 3)
        finally:
            llm.os.environ.pop(llm.ENDPOINT_ENV, None)
        self.assertEqual(recs, [good])
        self.assertEqual(p.n_dropped, 2)

    def test_proposed_records_reach_the_schemas(self):
        dom_ids = ("logistics",)
        fixed = {f.name: f.values()[1][0] for f in domain("logistics").fields}

        class Fixed:
            n_calls = n_stubbed = n_records = n_dropped = 0

            def propose(self, dom, lang, n):
                return [dict(fixed) for _ in range(n)]

            def report(self):
                return {"configured": "test-double"}

        budget = Budget.plan(per_cell=2, domains=dom_ids, languages=("en",),
                             ks=(3,))
        res = generate(budget, seed=17, scanner=None, proposer=Fixed(),
                       log=QUIET)
        self.assertTrue(res.schemas)
        self.assertEqual({s.source for s in res.schemas}, {"llm"})


class GateTest(unittest.TestCase):
    def test_gate_runs_and_reports_every_criterion(self):
        gate = gate_mod.run(per_cell=1, seed=5, log=QUIET)
        self.assertEqual(gate["task"], "T-gen-schemas")
        self.assertTrue(gate["checks"]["dedup_rejects_duplicate"])
        self.assertTrue(gate["checks"]["firewall_clean"])
        self.assertTrue(gate["checks"]["no_quota_overflow"])
        self.assertTrue(gate["checks"]["budget_full"])
        # per_cell=1 is under MIN_SCHEMAS on purpose: the gate must say so
        # rather than pass a run that is too small to mean anything.
        self.assertFalse(gate["checks"]["schemas"])
        self.assertFalse(gate["pass"])

    def test_committed_gate_artifact_passes(self):
        path = gate_mod.GATE_DIR / "gate.json"
        self.assertTrue(path.exists(), f"missing {path}")
        gate = json.loads(path.read_text())
        self.assertTrue(gate["pass"], gate["checks"])
        self.assertGreaterEqual(gate["diversity"]["unique_skeletons"],
                                gate_mod.MIN_SKELETONS)
        self.assertGreaterEqual(len(gate["diversity"]["domains"]),
                                gate_mod.MIN_DOMAINS)


if __name__ == "__main__":
    unittest.main()
