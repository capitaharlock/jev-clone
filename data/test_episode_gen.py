"""Tests del generador de episodios (#T-episode-gen).

El gate, ejecutable:

* el 100 % de lo publicado pasa el validador de `#T-episode-contract`
  y los rechazos quedan con motivo;
* en comparación, perturbar un atributo del estado cambia el gold de
  la regla (el generador no puede publicar un gold contradicho);
* ningún episodio lleva el marcador de plantilla de `data/labelgen.py`
  ni una pregunta igual a un ID de dataset.

Run (UN directorio cada vez — colisión de basenames en este repo):
    PYTHONPATH=. pytest data/test_episode_gen.py -q
"""

from __future__ import annotations

import random
import unittest

from . import episode_contract as EC
from . import episode_gen as EG

SEED = 20260926


def _smoke_episodes(n: int = 20) -> list[dict]:
    out = EG.run(n=n, seed=SEED, prose="local", teacher="stub",
                 out_dir="/tmp/episode-gen-test")
    assert out["n_episodes"] == n, out
    return EG.load_episodes(out["out"])


class SplitDeclaredTest(unittest.TestCase):
    def test_plan_covers_five_families_two_langs(self):
        plan = EG.declared_split(2000, SEED)
        self.assertEqual(len(plan), 10)
        for f in EC.FAMILIES:
            for lang in EC.LANGS:
                self.assertGreater(plan[(f, lang)], 0)
        self.assertEqual(sum(plan.values()), 2000)

    def test_plan_deterministic(self):
        self.assertEqual(EG.declared_split(25, SEED),
                         EG.declared_split(25, SEED))

    def test_manifest_declared_before_generating(self):
        import json
        out = EG.run(n=10, seed=SEED, prose="local", teacher="stub",
                     out_dir="/tmp/episode-gen-test-plan")
        with open(out["out"] + "/manifest.json") as fh:
            manifest = json.load(fh)
        self.assertIn("planned_split", manifest)
        self.assertEqual(sum(manifest["planned_split"].values()), 10)
        self.assertEqual(manifest["schema_sha"], EC.schema_sha())


class RuleGoldTest(unittest.TestCase):
    def test_comparison_deterministic(self):
        attrs = [{"price_eur": 42, "duration_h": 3},
                 {"price_eur": 25, "duration_h": 5}]
        self.assertEqual(EG.comparison_gold(attrs, "price"), 1)
        self.assertEqual(EG.comparison_gold(attrs, "duration"), 0)

    def test_comparison_perturb_flips_gold(self):
        rng = random.Random(0)
        for _ in range(20):
            p1, p2 = rng.randint(15, 60), rng.randint(15, 60)
            if p1 == p2:
                continue
            attrs = [{"price_eur": p1, "duration_h": 2},
                     {"price_eur": p2, "duration_h": 2}]
            before = EG.comparison_gold(attrs, "price")
            attrs[1 - before]["price_eur"] = min(p1, p2) - 3
            self.assertNotEqual(EG.comparison_gold(attrs, "price"), before)

    def test_priority_tie_goes_to_price(self):
        attrs = [{"price_eur": 180, "duration_h": 2},
                 {"price_eur": 150, "duration_h": 2}]
        self.assertEqual(EG.priority_gold(attrs), 1)

    def test_smoke_rule_golds_match_recomputation(self):
        for ep in _smoke_episodes():
            rt = ep["rule_trace"]
            cands = [c["id"] for c in ep["candidates"]]
            if ep["family"] == EC.COMPARISON:
                gold = EG.comparison_gold(rt["attrs"], rt["criterion"])
            elif ep["family"] == EC.PRIORITY:
                gold = EG.priority_gold(rt["attrs"])
            else:
                continue
            self.assertEqual(ep["answer"], cands[gold])


class GateOnSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episodes = _smoke_episodes(20)

    def test_validator_pass_100pc(self):
        per = EC.batch_validate(self.episodes)
        self.assertEqual(per["n_invalid"], 0, per["per_episode"])

    def test_no_labelgen_marker_no_dataset_id(self):
        for ep in self.episodes:
            for m in EG.LABELGEN_MARKERS:
                self.assertNotIn(m, ep["state"])
                self.assertNotIn(m, ep["question"])
            self.assertFalse(
                EC.is_dataset_id_question(ep["question"]), ep["question"])

    def test_teacher_trace_never_a_probability(self):
        for ep in self.episodes:
            trace = ep["teacher_trace"]
            self.assertIn(trace["choice"],
                          [c["id"] for c in ep["candidates"]])
            self.assertTrue(trace.get("confidence_discarded"))
            self.assertNotIn("probability", trace)
            self.assertNotIn("confidence", trace)

    def test_counterfactual_pairs_share_group(self):
        groups: dict[str, set] = {}
        for ep in self.episodes:
            groups.setdefault(ep["variant_group"], set()).add(ep["id"])
        self.assertTrue(any(len(v) >= 2 for v in groups.values()))

    def test_reproducible_by_seed(self):
        again = EG.run(n=20, seed=SEED, prose="local", teacher="stub",
                       out_dir="/tmp/episode-gen-test-repro")
        eps = EG.load_episodes(again["out"])
        self.assertEqual([e["state"] for e in eps],
                         [e["state"] for e in self.episodes])


if __name__ == "__main__":
    unittest.main()
