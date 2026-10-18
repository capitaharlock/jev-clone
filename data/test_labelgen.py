"""Tests for the label-space factory (#T-labelspace-factory).

Covers the pure pipeline — prompt cache keys, space validation, dedupe,
template rows (schema shape, whole-space denominator, determinism),
registry cards, firewall screening, neighbourhood stats and the factory
mix quotas — without touching the model. The one live test needs
LABELGEN_LIVE=1 and a serving ollama.

Run:
    PYTHONPATH=. python3 -m pytest data/test_labelgen.py -q
"""
from __future__ import annotations

import os
import unittest

from . import hardneg
from . import labelgen
from . import mix as mixmod
from . import schema as schemamod
from .firewall import reject_train_rows
from .registry import Registry


def _space(k=5):
    labels = [
        {"name": f"Harbor Rule {w}",
         "def": f"Berth assignment rule number {i} for tidal ports."}
        for i, w in enumerate(
            ("Alpha", "Beta", "Gamma", "Delta", "Epsilon")[:k])]
    names = [lab["name"] for lab in labels]
    confs = [[names[0], names[1], "both govern arrival timing"]]
    if k >= 4:
        confs.append([names[2], names[3], "both govern crane use"])
    if k >= 5:
        confs.append([names[0], names[4], "both mention draft limits"])
    return {
        "domain": "port_berthing",
        "labels": labels,
        "confusions": confs,
    }


class TestCacheKeys(unittest.TestCase):
    def test_same_prompt_same_key(self):
        self.assertEqual(labelgen.cache_key("a b", 5),
                         labelgen.cache_key("a b", 5))

    def test_k_changes_the_key(self):
        self.assertNotEqual(labelgen.cache_key("a b", 4),
                            labelgen.cache_key("a b", 5))

    def test_hints_are_deterministic(self):
        self.assertEqual(labelgen.domain_hints(7, 40),
                         labelgen.domain_hints(7, 40))

    def test_catalog_covers_the_target(self):
        self.assertGreaterEqual(len(labelgen.FIELDS) * len(labelgen.CONTEXTS),
                                labelgen.N_TARGET)


class TestValidateSpace(unittest.TestCase):
    def test_good_space_passes(self):
        self.assertEqual(labelgen.validate_space(_space()), [])

    def test_k_bounds(self):
        bad = _space(k=3)
        self.assertTrue(labelgen.validate_space(bad))

    def test_duplicate_names_rejected(self):
        bad = _space()
        bad["labels"][1]["name"] = bad["labels"][0]["name"]
        self.assertTrue(labelgen.validate_space(bad))

    def test_confusion_must_name_two_distinct_labels(self):
        bad = _space()
        bad["confusions"][0] = ["No Such Label", bad["labels"][0]["name"],
                                "why"]
        self.assertTrue(labelgen.validate_space(bad))
        bad2 = _space()
        bad2["confusions"][0] = [bad2["labels"][0]["name"],
                                 bad2["labels"][0]["name"], "why"]
        self.assertTrue(labelgen.validate_space(bad2))


class TestDedupe(unittest.TestCase):
    def test_near_identical_label_rejected(self):
        prev = [_space()]
        alt = _space()
        alt["labels"][0] = {"name": "Harbor Rule Alpha",
                            "def": "something else entirely here"}
        dup, _ = labelgen.is_duplicate(alt, prev)
        self.assertTrue(dup)

    def test_distant_space_accepted(self):
        prev = [_space()]
        other = {"domain": "oven_scheduling",
                 "labels": [{"name": "Stone Deck Preheat",
                             "def": "Heat the stone deck before loading."},
                            {"name": "Steam Burst Timing",
                             "def": "Release steam early in the bake."},
                            {"name": "Crust Color Check",
                             "def": "Judge doneness by crust color."},
                            {"name": "Cooling Rack Rest",
                             "def": "Rest loaves before slicing."}],
                 "confusions": [["Stone Deck Preheat", "Steam Burst Timing",
                                 "both early in the bake"],
                                ["Crust Color Check", "Cooling Rack Rest",
                                 "both after the bake"]]}
        self.assertEqual(labelgen.validate_space(other), [])
        dup, _ = labelgen.is_duplicate(other, prev)
        self.assertFalse(dup)


class TestRows(unittest.TestCase):
    def test_rows_validate_against_schema(self):
        space = _space()
        for row in labelgen.rows_of(space, "lq-0000", 1):
            ex = schemamod.Example(
                state=row["state"],
                questions=[schemamod.Question(
                    id=q["id"], kind=q["kind"],
                    options=[schemamod.Option(id=o["id"], text=o["text"])
                             for o in q["options"]],
                    answer=q["answer"]) for q in row["questions"]],
                split=row["split"])
            self.assertEqual(schemamod.validate(ex), [])

    def test_whole_space_is_the_denominator(self):
        space = _space()
        rows = labelgen.rows_of(space, "lq-0000", 1)
        self.assertEqual(len(rows), labelgen.ROWS_PER_SPACE)
        names = {lab["name"] for lab in space["labels"]}
        for row in rows:
            q = row["questions"][0]
            self.assertEqual({o["text"] for o in q["options"]}, names)
            gold = next(o["id"] for o in q["options"]
                        if o["id"] == q["answer"])
            self.assertTrue(gold.startswith("lq-0000--"))

    def test_golds_cover_the_space(self):
        space = _space()
        rows = labelgen.rows_of(space, "lq-0000", 1)
        self.assertEqual(len({r["questions"][0]["answer"] for r in rows}),
                         len(space["labels"]))

    def test_deterministic(self):
        space = _space()
        self.assertEqual(labelgen.rows_of(space, "lq-0000", 1),
                         labelgen.rows_of(space, "lq-0000", 1))


class TestRegistryAndFirewall(unittest.TestCase):
    def test_card_registers_and_trains(self):
        reg = Registry()
        card = labelgen.card_of("lq-0000", _space())
        reg.register(card)  # raises on a bad card
        ok, _ = reg.train_ok(card.id)
        self.assertTrue(ok)
        self.assertEqual(len(card.sha256), 64)

    def test_generated_texts_pass_the_firewall(self):
        space = _space()
        rows = labelgen.rows_of(space, "lq-0000", 1)
        self.assertTrue(
            reject_train_rows(labelgen.space_texts(space, rows)))

    def test_firewall_still_catches_a_canary(self):
        from .firewall import canary_texts
        with self.assertRaises(ValueError):
            reject_train_rows([canary_texts()[0]])


class TestNeighbourhood(unittest.TestCase):
    def test_shape_and_hardest_pair(self):
        stat = labelgen.neighbourhood(_space())
        self.assertEqual(stat["k"], 5)
        self.assertEqual(len(stat["per_label"]), 5)
        a, b, cos = stat["hardest_pair"]
        self.assertNotEqual(a, b)
        self.assertGreaterEqual(cos, stat["space_min_nn"])
        self.assertLessEqual(cos, 1.0)
        self.assertGreaterEqual(stat["space_min_nn"], 0.0)

    def test_identical_labels_score_one(self):
        vec = hardneg._embed("same text here")
        cos = sum(x * y for x, y in zip(vec, vec))
        self.assertAlmostEqual(cos, 1.0, places=6)


class TestFactoryMix(unittest.TestCase):
    def test_new_source_registered_but_not_default(self):
        self.assertIn("labelspace-qwen", mixmod.SOURCES)
        self.assertTrue(mixmod.SOURCES["labelspace-qwen"].experimental)
        self.assertNotIn("labelspace-qwen",
                         mixmod.default_datasets())
        self.assertIn("labelspace-qwen",
                      mixmod.default_datasets(["labelspace-qwen"]))

    def test_quotas_hold_on_synthetic_supply(self):
        supply = {"labelspace-qwen": 24000, "episodic-div": 1000000,
                  "banking77": 10000, "massive": 11000, "huffpost": 40000,
                  "dbpedia14": 100000, "goemotions": 50000, "swag": 73000}
        spec = mixmod.plan_mix(supply, target=62000, seed=1)
        quotas = spec.quotas
        total = sum(quotas.values())
        self.assertEqual(total, 62000)
        syn = quotas.get("labelspace-qwen", 0) + quotas.get("episodic-div", 0)
        hum = sum(quotas.get(d, 0) for d in
                  ("banking77", "massive", "huffpost", "dbpedia14",
                   "goemotions"))
        self.assertLessEqual(syn / total, mixmod.MAX_SYNTHETIC_FRACTION)
        self.assertGreaterEqual(hum / total, mixmod.MIN_HUMAN_FRACTION)
        for dataset, n in quotas.items():
            self.assertLessEqual(n / total,
                                 mixmod.MAX_DATASET_FRACTION + 1e-9)

    def test_factory_selection_lists_the_new_spaces(self):
        self.assertIn("labelspace-qwen", labelgen.FACTORY_DATASETS)
        self.assertIn("episodic-div", labelgen.FACTORY_DATASETS)

    def test_tiny_factory_supply_falls_back_to_feasible(self):
        # the authentic failure: 4 spaces (48 rows) cannot fill a 62k mix
        # because nobody else may use the labelspace family's headroom.
        supply = {"labelspace-qwen": 48, "episodic-div": 1040000,
                  "banking77": 10003, "massive": 69084, "huffpost": 162935,
                  "dbpedia14": 560000, "goemotions": 87076, "swag": 73546}
        with self.assertRaises(mixmod.MixInfeasible):
            mixmod.plan_mix(supply, target=62000, seed=1)
        # under the SAME margin caps plan_mix plans against — the default
        # caps admit a bigger mix the loader would then refuse.
        margin_ds = mixmod.MAX_DATASET_FRACTION - mixmod.CAP_SAFETY_MARGIN
        margin_fam = mixmod.MAX_FAMILY_FRACTION - mixmod.CAP_SAFETY_MARGIN
        feasible = mixmod.max_feasible_target(supply, margin_ds, margin_fam)
        self.assertGreater(feasible, 0)
        self.assertLess(feasible, 62000)
        spec = mixmod.plan_mix(supply, target=feasible, seed=1)
        self.assertEqual(sum(spec.quotas.values()), feasible)


class TestLiveGeneration(unittest.TestCase):
    def test_one_space_end_to_end(self):
        if os.environ.get("LABELGEN_LIVE") != "1":
            self.skipTest("needs LABELGEN_LIVE=1 + serving ollama")
        hint, k = labelgen.domain_hints(20260924, 1)[0]
        reply = labelgen.qwen_space(hint, k)
        self.assertNotIn("error", reply)
        self.assertEqual(labelgen.validate_space(reply["space"]), [])


if __name__ == "__main__":
    unittest.main()
