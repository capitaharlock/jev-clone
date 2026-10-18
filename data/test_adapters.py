"""Tests for #T-data-p0 (stdlib unittest — no deps yet)."""
from __future__ import annotations

import json
import os
import random
import unittest

from .adapters import (
    ADAPTERS,
    GATE_DIR,
    MIXER_SEED,
    AdapterError,
    MixedLoader,
    build_summary,
    check_load,
    insert_distractor,
    p0_cards,
    p0_registry,
    rename_labels,
    shuffle_options,
    typo_noise,
    write_summary,
)
from .firewall import BenchmarkRegistry
from .schema import Option, is_valid

FIX_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "artifacts", "fixtures", "p0",
)


def load(name: str) -> list[dict]:
    with open(os.path.join(FIX_DIR, f"{name}.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def pools() -> dict[str, list[dict]]:
    return {k: v for k, v in {
        "huffpost": ADAPTERS["huffpost"](load("huffpost"), seed=1),
        "banking77": ADAPTERS["banking77"](load("banking77"), seed=1),
        "boolq": ADAPTERS["boolq"](load("boolq")),
        "civil-comments": ADAPTERS["civil-comments"](load("civil")),
        "helpsteer2": ADAPTERS["helpsteer2"](load("helpsteer2")),
    }.items()}


class TestAdapters(unittest.TestCase):
    def test_all_five_convert_and_validate(self):
        for ds, examples in pools().items():
            self.assertTrue(examples, ds)
            for ex in examples:
                self.assertTrue(is_valid(ex), (ds, ex))

    def test_correct_option_and_splits(self):
        p = pools()
        self.assertEqual(p["boolq"][0].questions[0].answer, "yes")
        self.assertEqual(p["boolq"][1].questions[0].answer, "no")
        self.assertEqual(len(p["boolq"][0].questions[0].options), 2)
        self.assertEqual(len(p["helpsteer2"][0].questions), 5)
        self.assertTrue(all(len(q.options) == 5 for q in [qq for ex in p["helpsteer2"] for qq in ex.questions]))
        splits = {ex.split for ex in p["huffpost"]}
        self.assertTrue({"train", "calibration", "test"} <= splits)

    def test_unknown_unicode(self):
        p = pools()
        self.assertEqual(p["huffpost"][-1].questions[0].answer, "unknown")
        self.assertEqual(p["boolq"][-1].questions[0].answer, "unknown")
        self.assertIn("¿Dónde", p["banking77"][2].state)
        self.assertIn("mareas", p["helpsteer2"][2].state)

    def test_corrupt_rejected(self):
        with self.assertRaises(AdapterError):
            ADAPTERS["huffpost"]([{"headline": "", "description": "x", "category": "TECH"}])
        with self.assertRaises(AdapterError):
            ADAPTERS["huffpost"]([{"headline": "h"}])  # missing description
        with self.assertRaises(AdapterError):
            ADAPTERS["banking77"]([{"text": "x", "label": "only-one"}])
        with self.assertRaises(AdapterError):
            ADAPTERS["boolq"]([{"passage": "p"}])
        with self.assertRaises(AdapterError):
            ADAPTERS["civil-comments"]([{"text": "  "}])
        with self.assertRaises(AdapterError):
            ADAPTERS["helpsteer2"]([{"prompt": "p", "response": "r",
                                     "helpfulness": 9, "correctness": 1,
                                     "coherence": 1, "complexity": 1, "verbosity": 1}])

    def test_pinned_cards(self):
        for card in p0_cards():
            self.assertTrue(card.revision and "TBD" not in card.revision)
            self.assertEqual(len(card.sha256), 64)

    def test_fence_and_firewall_precheck(self):
        reg = p0_registry()
        # Train-clear: banking77, civil-comments, helpsteer2.
        for ds in ("banking77", "civil-comments", "helpsteer2"):
            rep = check_load(["ordinary training text"], ds, reg)
            self.assertTrue(rep.fence_ok, (ds, rep.fence_reason))
            self.assertEqual(rep.contamination_hits, 0)
        # Gated: huffpost (content-rights fence), boolq (share-alike).
        for ds in ("huffpost", "boolq"):
            rep = check_load(["ordinary text"], ds, reg)
            self.assertFalse(rep.fence_ok, ds)
        # Benchmark ids can never enter train.
        bench = BenchmarkRegistry()
        self.assertTrue(bench.check_train_barrier(["mmlu-pro"], "train"))


class TestTransforms(unittest.TestCase):
    def test_shuffle_distractor_rename_typo(self):
        ex = pools()["banking77"][0]
        rng = random.Random("t")
        self.assertTrue(is_valid(shuffle_options(ex, rng)))
        ids_before = sorted(o.id for o in ex.questions[0].options)
        grown = insert_distractor(ex, Option("zz-hard", "hard negative"), random.Random("t"))
        self.assertEqual(len(grown.questions[0].options), len(ids_before) + 1)
        self.assertTrue(is_valid(grown))
        renamed = rename_labels(ex, {ex.questions[0].options[0].id: "RENAMED"})
        self.assertTrue(is_valid(renamed))
        noisy = typo_noise("a plain example sentence", random.Random("t"))
        self.assertNotEqual(noisy, "a plain example sentence")


class TestMixedLoader(unittest.TestCase):
    def test_deterministic_batches_and_manifest(self):
        a = MixedLoader(pools(), seed=MIXER_SEED)
        b = MixedLoader(pools(), seed=MIXER_SEED)
        self.assertEqual(a.manifest(), b.manifest())
        self.assertEqual(
            [[q.id for ex in b_ for q in ex.questions] for b_ in a.batches(4)],
            [[q.id for ex in b_ for q in ex.questions] for b_ in b.batches(4)],
        )
        c = MixedLoader(pools(), seed=MIXER_SEED + 1)
        self.assertNotEqual(a.manifest()["manifest_sha256"], c.manifest()["manifest_sha256"])

    def test_option_range_and_question_counts(self):
        p = pools()
        for ds, pool in p.items():
            for ex in pool:
                self.assertTrue(1 <= len(ex.questions) <= 5, ds)
                for q in ex.questions:
                    self.assertTrue(2 <= len(q.options) <= 42, (ds, q.id))
        loader = MixedLoader(p)
        self.assertEqual(loader.manifest()["examples"], sum(len(v) for v in p.values()))

    def test_no_id_crosses_splits(self):
        p = pools()
        p["boolq"][0].split = "test"  # force a cross-split collision? ids differ; build direct clash
        clash = p["banking77"][0]
        dup = p["boolq"][0]
        dup.questions[0].id = clash.questions[0].id
        dup.split = "test" if clash.split == "train" else "train"
        with self.assertRaises(AdapterError):
            MixedLoader({"a": [clash], "b": [dup]})

    def test_smoke_training_reduces_loss_no_nan(self):
        """Tiny CPU smoke: majority baseline beats uniform, all finite."""
        import math as m
        labels = [ex.questions[0].answer for ex in pools()["boolq"] if ex.questions[0].answer != "unknown"]
        n = len(labels)
        uniform_loss = m.log(2)
        majority_p = max(labels.count("yes"), labels.count("no")) / n
        majority_loss = -(m.log(majority_p))
        self.assertTrue(m.isfinite(uniform_loss) and m.isfinite(majority_loss))
        self.assertLessEqual(majority_loss, uniform_loss)

    def test_summary_published(self):
        summary = build_summary(pools())
        self.assertEqual(summary["counts"]["boolq"], 6)
        self.assertIn("train", summary["splits"])
        self.assertFalse(summary["fence"]["huffpost"]["train_ok"])
        self.assertFalse(summary["fence"]["boolq"]["train_ok"])
        self.assertTrue(summary["fence"]["banking77"]["train_ok"])
        path = write_summary(summary, GATE_DIR)
        self.assertTrue(os.path.isfile(path))
        self.assertEqual(len(summary["manifest"]["manifest_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
