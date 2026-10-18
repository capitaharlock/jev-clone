"""Tests for the label-space counter (#T-labelspace-div, stdlib only).

Run:
    python3 -m unittest data.test_labelspace -v
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from . import labelspace
from .labelspace import (REUSABLE_MIN_ROWS, SHARED_POOL_THRESHOLD,
                         classify_space, inventory_of, option_signature)
from . import mix as mixmod


def _row(qid, options, answer, state="s"):
    return {"state": state,
            "questions": [{"id": qid, "kind": "choice",
                           "options": [{"id": t, "text": t} for t in options],
                           "answer": answer}],
            "split": "train"}


class SignatureTest(unittest.TestCase):
    def test_order_invariant(self):
        a = {"options": [{"text": "x"}, {"text": "y"}]}
        b = {"options": [{"text": "y"}, {"text": "x"}]}
        self.assertEqual(option_signature(a), option_signature(b))

    def test_distinguishes_sets(self):
        a = {"options": [{"text": "x"}, {"text": "y"}]}
        c = {"options": [{"text": "x"}, {"text": "z"}]}
        self.assertNotEqual(option_signature(a), option_signature(c))

    def test_threshold_boundary(self):
        self.assertEqual(classify_space(SHARED_POOL_THRESHOLD), "shared")
        self.assertEqual(classify_space(SHARED_POOL_THRESHOLD + 1), "per-row")


class CounterFixtureTest(unittest.TestCase):
    """The counter over two fixture shards behind a tmp prefetch root.

    `banking77` stands in for a shared taxonomy (3 labels, every row
    reuses them); `swag` for per-row spaces (every row its own options).
    `Source.paths(root)` reads `<root>/<id>.jsonl` for a non-default root,
    so no fixture reproduces the real layout.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        with open(os.path.join(self.tmp.name, "banking77.jsonl"), "w") as fh:
            for i, gold in enumerate(["a", "b", "c", "a"]):
                fh.write(json.dumps(_row(f"tax-{i}", ["a", "b", "c"],
                                         gold)) + "\n")
        with open(os.path.join(self.tmp.name, "swag.jsonl"), "w") as fh:
            for i in range(9):
                opts = [f"cont-{i}-{k}" for k in range(4)]
                fh.write(json.dumps(_row(f"sw-{i}", opts, opts[0])) + "\n")
        self.spec = mixmod.MixSpec(
            version=mixmod.MIX_VERSION, seed=1, target=13,
            quotas={"banking77": 4, "swag": 9},
            supply={"banking77": 4, "swag": 9},
            keep_fractions={"banking77": 1.0, "swag": 1.0},
            seen_labels={}, shards={})

    def tearDown(self):
        self.tmp.cleanup()

    def test_shared_vs_per_row(self):
        per, counts = labelspace.collect_spaces(self.spec, self.tmp.name,
                                                  threshold=32)
        self.assertEqual(counts["members_sha256"],
                         counts["members_sha256"])  # deterministic below
        tax, sw = per["banking77"], per["swag"]
        from data.labelspace import classify_space as _cs
        tax_kind = _cs(len(tax["pool"]), 32)
        sw_kind = _cs(len(sw["pool"]), 32)
        self.assertEqual(tax_kind, "shared")
        self.assertEqual(len(tax["pool"]), 3)
        self.assertEqual(len(tax["signatures"]), 1)
        self.assertEqual(sw_kind, "per-row")
        self.assertEqual(len(sw["signatures"]), 9)

    def test_deterministic_digest(self):
        _, c1 = labelspace.collect_spaces(self.spec, self.tmp.name)
        _, c2 = labelspace.collect_spaces(self.spec, self.tmp.name)
        self.assertEqual(c1["members_sha256"], c2["members_sha256"])


class InventoryTest(unittest.TestCase):
    def test_poor_verdict_and_reusable_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "pref")
            os.makedirs(root)
            with open(os.path.join(root, "banking77.jsonl"), "w") as fh:
                for i in range(90):
                    gold = ["a", "b", "c"][i % 3]
                    fh.write(json.dumps(_row(f"t-{i}", ["a", "b", "c"],
                                             gold)) + "\n")
            with open(os.path.join(root, "swag.jsonl"), "w") as fh:
                for i in range(10):
                    opts = [f"u-{i}-{k}" for k in range(4)]
                    fh.write(json.dumps(_row(f"s-{i}", opts, opts[0]))
                             + "\n")
            spec = mixmod.MixSpec(
                version=mixmod.MIX_VERSION, seed=7, target=100,
                quotas={"banking77": 90, "swag": 10},
                supply={"banking77": 90, "swag": 10},
                keep_fractions={"banking77": 1.0, "swag": 1.0},
                seen_labels={}, shards={})
            per, counts = labelspace.collect_spaces(spec, root)
            manifest = {"corpus": "fixture", "task": "T-labelspace-div",
                        "version": mixmod.MIX_VERSION, "seed": 7,
                        "target": 100,
                        "sources": {
                            "banking77": {"quota": 90, "supply": 90},
                            "swag": {"quota": 10, "supply": 10}},
                        "selection": {"keep_fractions": {
                            "banking77": 1.0, "swag": 1.0}},
                        "holdout": {},
                        "members_sha256": counts["members_sha256"]}
            mpath = os.path.join(tmp, "m.json")
            with open(mpath, "w") as fh:
                json.dump(manifest, fh)
            inv = inventory_of(mpath, root, threshold=32)
            self.assertTrue(inv["selection_match"])
            self.assertEqual(inv["n_shared"], 1)
            self.assertEqual(inv["filas_por_espacio_shared"],
                             {"banking77": 90})
            self.assertEqual(inv["fraccion_shared"], 0.9)
            self.assertEqual(inv["veredicto"], "pobre")
            # the shared taxonomy is reusable; the 10 singletons are not
            self.assertEqual(inv["n_reutilizables"], 1)
            self.assertEqual(inv["reutilizable_min_filas"],
                             REUSABLE_MIN_ROWS)
            self.assertEqual(inv["n_espacios"], 11)


if __name__ == "__main__":
    unittest.main()
