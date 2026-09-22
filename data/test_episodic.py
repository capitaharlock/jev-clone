"""Tests for the episodic many-spaces generator (#T-labelspace-div).

Covers the generator (determinism, global label uniqueness, schema shape)
and the new mix's sampler path: the experimental source is registered but
excluded from every default, and explicitly selectable end to end.

Run:
    python3 -m unittest data.test_episodic -v
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from . import episodic
from . import mix as mixmod


def _gen(tmp, seed=99, spaces=8, rps=5, shards=2):
    return episodic.generate(seed, os.path.join(tmp, "ep"), spaces, rps,
                             shards, log=None)


def _read_shards(tmp):
    rows = []
    for f in sorted(os.listdir(os.path.join(tmp, "ep"))):
        with open(os.path.join(tmp, "ep", f)) as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    return rows


class GeneratorTest(unittest.TestCase):
    def test_deterministic_bytes(self):
        with tempfile.TemporaryDirectory() as a, \
                tempfile.TemporaryDirectory() as b:
            _gen(a)
            _gen(b)
            fa = sorted(os.listdir(os.path.join(a, "ep")))
            fb = sorted(os.listdir(os.path.join(b, "ep")))
            self.assertEqual(fa, fb)
            for f in fa:
                self.assertEqual(
                    open(os.path.join(a, "ep", f), "rb").read(),
                    open(os.path.join(b, "ep", f), "rb").read())

    def test_schema_and_answerability(self):
        with tempfile.TemporaryDirectory() as tmp:
            _gen(tmp)
            rows = _read_shards(tmp)
            self.assertEqual(len(rows), 8 * 5)
            for r in rows:
                self.assertEqual(r["split"], "train")
                (q,) = r["questions"]
                texts = [o["text"] for o in q["options"]]
                self.assertIn(q["answer"], [o["id"] for o in q["options"]])
                self.assertIn(f"= {q['answer']}", r["state"])
                self.assertEqual(2 <= len(texts), True)
                self.assertEqual(len(texts), len(set(texts)))
                if len(texts) == 2:
                    self.assertEqual(q["kind"], "boolean")
                else:
                    self.assertEqual(q["kind"], "choice")

    def test_spaces_never_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            _gen(tmp, spaces=12, rps=6)
            per_space: dict = {}
            for r in _read_shards(tmp):
                (q,) = r["questions"]
                sid = q["id"].rsplit("-", 1)[0]
                per_space.setdefault(sid, set()).update(
                    o["text"] for o in q["options"])
            self.assertEqual(len(per_space), 12)
            seen: set = set()
            for sid, texts in per_space.items():
                self.assertTrue(texts.isdisjoint(seen), sid)
                seen |= texts

    def test_manifest_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = _gen(tmp, spaces=6, rps=4)
            self.assertEqual(m["version"], episodic.EPISODIC_VERSION)
            self.assertEqual(m["recipe"]["supply_rows"], 24)
            self.assertEqual(m["labelspace"]["n_espacios"], 6)
            self.assertEqual(m["labelspace"]["n_reutilizables"], 6)
            self.assertEqual(len(m["shards"]), 2)


class SamplerPathTest(unittest.TestCase):
    """The new mix's sampler: registered, excluded by default, selectable."""

    def test_registered_but_never_default(self):
        self.assertIn("episodic-div", mixmod.SOURCES)
        self.assertTrue(mixmod.SOURCES["episodic-div"].experimental)
        self.assertNotIn("episodic-div", mixmod.default_datasets())
        self.assertNotIn("episodic-div", mixmod.clean_datasets())
        # explicit lists pass through: selectable, never default
        self.assertIn("episodic-div",
                      mixmod.default_datasets(["episodic-div"]))

    def test_plan_and_realise_over_fixture(self):
        # scan_supply reads <root>/<id>.jsonl for a non-default root, so a
        # 2-space fixture stands in for the 20-shard corpus.
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, "episodic-div.jsonl"), "w") as fh:
                for i in range(20):
                    sid = f"epdiv-0000{i // 10}"
                    opts = ([f"w{i}", f"x{i}"] if i % 2
                            else [f"w{i}", f"x{i}", f"y{i}"])
                    fh.write(json.dumps({
                        "state": f"Record e-{i}:\nf = {opts[0]}",
                        "questions": [{
                            "id": f"{sid}-{i % 10}", "kind": "choice",
                            "options": [{"id": t, "text": t} for t in opts],
                            "answer": opts[0]}],
                        "split": "train"}) + "\n")
            scan = mixmod.scan_supply(["episodic-div"], root)
            self.assertFalse(scan["episodic-div"].get("missing"))
            supply = mixmod.effective_supply(scan)
            spec = mixmod.plan_mix(supply, 20, seed=1, cap_margin=0.0,
                                   dataset_cap=1.0, family_cap=1.0)
            self.assertEqual(spec.quotas["episodic-div"], 20)
            counts = mixmod.realise(spec, root)
            self.assertEqual(counts["rows"], 20)
            mixmod.verify_mix(counts["dataset"], counts["rows"],
                              dataset_cap=1.0, family_cap=1.0)


if __name__ == "__main__":
    unittest.main()
