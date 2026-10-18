"""Tests for #T-bigsrc (stdlib unittest)."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from .adapters import AdapterError
from .bigsrc import (
    adapt_docnli,
    adapt_p3,
    adapt_tasksource,
    balance_docnli,
    bigsrc_cards,
    bigsrc_registry,
    build_audit,
    commercial_clean_ok,
    leak_check,
    require_manifest,
    sample_p3,
    write_shard,
)
from .firewall import BenchmarkRegistry
from .registry import DatasetCard, Registry
from .schema import is_valid

FIX_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "artifacts", "fixtures", "bigsrc",
)


def load(name: str) -> list[dict]:
    with open(os.path.join(FIX_DIR, f"{name}.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


class TestBigsrcAdapters(unittest.TestCase):
    def test_all_three_convert_and_validate(self):
        ts = adapt_tasksource(load("tasksource"))
        ex, metas = adapt_p3(load("p3"))
        dn = adapt_docnli(load("docnli"))
        self.assertTrue(ts and ex and dn)
        for e in ts + ex + dn:
            self.assertTrue(is_valid(e), e)

    def test_tasksource_gold_shapes(self):
        ts = adapt_tasksource(load("tasksource"))
        self.assertEqual(ts[0].questions[0].answer, "yes")  # entailment NLI
        self.assertEqual(ts[1].questions[0].answer, "no")  # contradiction NLI
        self.assertEqual(ts[2].questions[0].answer, "water")
        self.assertEqual(len(ts[2].questions[0].options), 4)

    def test_docnli_is_noul_boolean(self):
        dn = adapt_docnli(load("docnli"))
        for e in dn:
            q = e.questions[0]
            self.assertEqual(q.kind, "boolean")
            self.assertEqual([o.id for o in q.options], ["yes", "no"])
        self.assertEqual([e.questions[0].answer for e in dn[:2]], ["yes", "no"])

    def test_filter_rejects_free_gen_cot_dialogue(self):
        for row in load("reject"):
            with self.assertRaises(AdapterError, msg=row):
                adapt_tasksource([row])
        # P3 path rejects a free-form target (not in finite choices).
        with self.assertRaises(AdapterError):
            adapt_p3([{"dataset": "d", "template": "t", "input": "Write an essay",
                       "answer_choices": ["a", "b"], "target": "a very long free essay text here that exceeds the short target budget by far"}])
        # Tasksource rejects a target outside the candidate set.
        with self.assertRaises(AdapterError):
            adapt_tasksource([{"task": "t", "task_type": "multiple-choice",
                               "input": "Q?", "target": "zzz",
                               "choices": ["a", "b"]}])

    def test_deterministic_shard_bytes_and_hash(self):
        ts = adapt_tasksource(load("tasksource"))
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            _, m1 = write_shard(ts, source="tasksource", rows_before=len(ts),
                                out_dir=d1, license="Apache-2.0", revision="r")
            _, m2 = write_shard(ts, source="tasksource", rows_before=len(ts),
                                out_dir=d2, license="Apache-2.0", revision="r")
            with open(os.path.join(d1, "tasksource.jsonl"), "rb") as f:
                b1 = f.read()
            with open(os.path.join(d2, "tasksource.jsonl"), "rb") as f:
                b2 = f.read()
            self.assertEqual(b1, b2)
            self.assertEqual(m1["shard_sha256"], m2["shard_sha256"])
            require_manifest(m1)

    def test_incomplete_manifest_refused(self):
        with self.assertRaises(AdapterError):
            require_manifest({"source": "x"})  # missing fields, no `complete`

    def test_p3_sampler_no_template_duplication(self):
        ex, metas = adapt_p3(load("p3"))
        kept, rep = sample_p3(ex, metas, per_template_cap=2, seed=7)
        # 4 anli templates share 3 inputs each: input-dedupe + template cap
        # must keep far fewer than the 12 raw anli rows.
        self.assertLessEqual(rep["accepted"], 3 + 2)  # 3 unique inputs + 2 arc
        self.assertGreater(rep["rejected_dupe_input"] + rep["rejected_template_cap"], 0)
        # Same source x N prompts is not N knowledges: sampled states unique.
        states = [e.state for e in kept]
        self.assertEqual(len(set(states)), len(states))
        # Deterministic under same seed.
        kept2, rep2 = sample_p3(ex, metas, per_template_cap=2, seed=7)
        self.assertEqual([e.state for e in kept], [e.state for e in kept2])
        self.assertEqual(rep, rep2)

    def test_docnli_balanced_sampler(self):
        dn = adapt_docnli(load("docnli"))
        kept, counts = balance_docnli(dn, seed=3)
        self.assertEqual(counts.get("yes"), counts.get("no"))
        self.assertLessEqual(len(kept), len(dn))

    def test_cards_pinned_and_train_clear(self):
        for card in bigsrc_cards():
            self.assertTrue(card.revision and "TBD" not in card.revision)
            self.assertEqual(len(card.sha256), 64)
        reg = bigsrc_registry()
        ok, errors = commercial_clean_ok(
            reg, ["tasksource-instruct", "p3", "docnli"])
        self.assertTrue(ok, errors)

    def test_commercial_branch_rejects_non_commercial(self):
        reg = Registry()
        reg.register(DatasetCard(
            id="nc-src", source_original="s", mirror="m", license="CC-BY-NC-4.0",
            usage="train", revision="r1", sha256="a" * 64, transform="t"))
        ok, errors = commercial_clean_ok(reg, ["nc-src"])
        self.assertFalse(ok)
        self.assertTrue(errors)

    def test_leak_detector_green_on_fixtures_red_on_canary(self):
        ts = adapt_tasksource(load("tasksource"))
        rep = leak_check([e.state for e in ts], BenchmarkRegistry())
        self.assertTrue(rep["clean"], rep)
        self.assertEqual(rep["jevals_overlap"], 0)
        with open(os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "artifacts", "fixtures", "firewall",
                "canary-exact.jsonl")) as f:
            canary = [json.loads(line)["text"] for line in f if line.strip()]
        bad = leak_check(["ordinary text", canary[0]], BenchmarkRegistry())
        self.assertFalse(bad["clean"])
        self.assertGreater(bad["contamination_hits"], 0)
        # Benchmark ids can never enter train.
        self.assertTrue(BenchmarkRegistry().check_train_barrier(["mmlu-pro"], "train"))

    def test_audit_table(self):
        md = build_audit({"tasksource-instruct": {
            "accepted": 6, "rejected": 4, "tokens_approx": 120,
            "family": "mixed-discriminative", "qtype": "Choice/Noul",
            "lang": "en", "license": "Apache-2.0", "k_dist": "2:3,3:1,4:1"}})
        self.assertIn("tasksource-instruct", md)
        self.assertIn("| 6 | 4 |", md)


if __name__ == "__main__":
    unittest.main()
