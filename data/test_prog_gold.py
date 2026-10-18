"""Tests for programmatic gold V1 (#T-prog-gold gate, stdlib unittest)."""
from __future__ import annotations

import random
import unittest

from data.optset import difficulty as optset_difficulty
from tools.data_factory.validator import validate_record
from tools.prog_gold.graph import (
    TripleStore,
    verify_answer_binding,
    verify_question,
    verify_record,
)
from tools.prog_gold.miner import (
    NearestLabelIndex,
    PoolTooSmall,
    adversarial_reorder,
    build_indexes,
    build_pools,
    sample_negatives,
)
from tools.prog_gold.packer import MAX_Q, MIN_Q, pack
from tools.prog_gold.pipeline import build_entity_qs, make_layer_d, state_text
from tools.prog_gold.quality import (
    MIN_ACCEPTED_SCORE,
    assign_difficulty,
    quality_block,
    score_for,
)
from tools.prog_gold.run_pilot import probe_top1, robustness_audit, tamper_probe
from tools.prog_gold.sampler_k import K_MIX, draw_k, mix_deviation
from tools.prog_gold.transforms import (
    MAX_TRANSFORM_SHARE,
    TRANSFORM_WEIGHTS,
    filler,
    irrelevant_context,
    transform_share,
)


def tiny_store() -> TripleStore:
    """The smallest legal graph: a two-value country pool, plus one film."""
    s = TripleStore()
    s.taxon_of = {"e1": "city", "e2": "city", "e3": "film"}
    s.labels = {"e1": "Alba", "e2": "Beta", "e3": "Gamma",
                "c1": "Norte", "c2": "Sur", "c3": "Este", "c4": "Oeste",
                "d1": "Dir X"}
    s.triples = {("e1", "P17"): {"c1"}, ("e2", "P17"): {"c2"},
                 ("e1", "P131"): {"c3"}, ("e2", "P131"): {"c4"},
                 ("e3", "P57"): {"d1"}}
    return s


def toy_store() -> TripleStore:
    """A graph wide enough to actually emit choice rows.

    The country pool needs more than two distinct labels or the pipeline
    emits booleans only — which is what made the first version of these
    tests measure nothing.
    """
    s = TripleStore()
    countries = {"c1": "Austria", "c2": "Australia", "c3": "Bangladesh",
                 "c4": "Austrasia", "c5": "Perú", "c6": "Portugal"}
    regions = {"r1": "Región Norte", "r2": "Región Sur", "r3": "Región Este",
               "r4": "Región Oeste", "r5": "Región Central"}
    s.taxon_of = {f"e{i}": "city" for i in range(1, 6)}
    s.taxon_of["f1"] = "film"
    s.labels = {"e1": "Alba", "e2": "Beta", "e3": "Gamma", "e4": "Delta",
                "e5": "Epsilon", "f1": "Película", "d1": "Dir X"}
    s.labels.update(countries)
    s.labels.update(regions)
    s.triples = {("f1", "P57"): {"d1"}}
    for i, (country, region) in enumerate(zip(list(countries)[:5],
                                              list(regions)), start=1):
        s.triples[(f"e{i}", "P17")] = {country}
        s.triples[(f"e{i}", "P131")] = {region}
    return s


def colour_store() -> TripleStore:
    """The §46 scenario itself: a favourite-colour taxon beside a volcano one.

    The pools exist side by side in one graph, which is exactly the situation
    in which a miner that ranks globally would offer "database" as a colour.
    """
    s = TripleStore()
    s.taxon_of = {"p1": "person", "p2": "person", "p3": "person",
                  "v1": "volcano", "v2": "volcano"}
    s.labels = {
        "p1": "Ana", "p2": "Bruno", "p3": "Carla",
        "v1": "Teide", "v2": "Etna",
        "col_rojo": "rojo", "col_verde": "verde", "col_azul": "azul",
        "col_violeta": "violeta", "col_amarillo": "amarillo",
        "thing_db": "database", "thing_volcano": "volcano",
        "thing_router": "router",
    }
    s.triples = {
        ("p1", "Pfav"): {"col_rojo"},
        ("p2", "Pfav"): {"col_verde"},
        ("p3", "Pfav"): {"col_azul"},
        # a different taxon whose property takes tech/geology values
        ("v1", "Pkind"): {"thing_volcano"},
        ("v2", "Pkind"): {"thing_db"},
    }
    # colours that exist in the graph but are not anyone's favourite yet
    s.triples[("p1", "Pseen")] = {"col_violeta", "col_amarillo"}
    return s


class TestChecker(unittest.TestCase):
    def test_accepts_graph_gold(self):
        s = toy_store()
        rec = {"gold": {"kind": "graph", "entity": "e1", "prop": "P17", "value": "c1"}}
        self.assertEqual(verify_record(rec, s), [])

    def test_rejects_tampered_gold(self):
        s = toy_store()
        rec = {"gold": {"kind": "graph", "entity": "e1", "prop": "P17", "value": "c2"}}
        errs = verify_record(rec, s)
        self.assertTrue(any("not in graph" in e for e in errs))

    def test_negated_gold_must_not_hold(self):
        s = toy_store()
        ok = {"gold": {"kind": "graph", "entity": "e1", "prop": "P17",
                       "value": "c2", "negated": True}}
        self.assertEqual(verify_record(ok, s), [])
        bad = {"gold": {"kind": "graph", "entity": "e1", "prop": "P17",
                        "value": "c1", "negated": True}}
        self.assertTrue(verify_record(bad, s))

    def test_hand_altered_answer_key_is_rejected(self):
        """The cheapest tampering: leave the gold pointer, move the answer."""
        s = toy_store()
        q = {"kind": "choice", "answer": "o0",
             "options": [{"id": "o0", "text": s.labels["c1"]},
                         {"id": "o1", "text": s.labels["c2"]}],
             "gold": {"kind": "graph", "entity": "e1", "prop": "P17", "value": "c1"}}
        self.assertEqual(verify_question(q, s), [])
        tampered = dict(q, answer="o1")
        errs = verify_question(tampered, s)
        self.assertTrue(any("answer points at" in e for e in errs), errs)

    def test_hand_altered_boolean_polarity_is_rejected(self):
        s = toy_store()
        q = {"kind": "boolean", "answer": "yes",
             "options": [{"id": "yes", "text": "Sí"}, {"id": "no", "text": "No"}],
             "gold": {"kind": "graph", "entity": "e1", "prop": "P17", "value": "c1"}}
        self.assertEqual(verify_question(q, s), [])
        self.assertTrue(verify_question(dict(q, answer="no"), s))

    def test_duplicate_gold_label_among_options_is_rejected(self):
        s = toy_store()
        s.labels["c2"] = s.labels["c1"]  # two QIDs, one wording: unanswerable
        q = {"kind": "choice", "answer": "o0",
             "options": [{"id": "o0", "text": s.labels["c1"]},
                         {"id": "o1", "text": s.labels["c1"]}],
             "gold": {"kind": "graph", "entity": "e1", "prop": "P17", "value": "c1"}}
        self.assertTrue(any("not unique" in e for e in verify_answer_binding(q, s)))

    def test_tamper_probe_on_built_records(self):
        """End-to-end: the gate's own probe must reject both hand edits."""
        s = toy_store()
        indexes = build_indexes(s)
        qs, _, _ = build_entity_qs("city", "e1", s, indexes, random.Random(3), [0])
        rec = {"state": state_text("city", "e1", s), "questions": qs,
               "split": "train", "parent_example_id": "wd:e1"}
        probe = tamper_probe([rec], s)
        self.assertTrue(probe["ran"])
        self.assertTrue(probe["clean_row_accepted"])
        self.assertTrue(probe["altered_gold_rejected"])
        self.assertTrue(probe["altered_answer_rejected"])


class TestMiner(unittest.TestCase):
    def test_rejects_database_and_volcano_for_favourite_colour(self):
        """§46 — a favourite-colour distractor is a colour. Never 'database'."""
        s = colour_store()
        indexes = build_indexes(s)
        index = indexes[("person", "Pfav")]
        colours = {"rojo", "verde", "azul"}
        rng = random.Random(0)
        for _ in range(50):
            negs = sample_negatives(index, "col_rojo", 2, rng)
            texts = {index.label[n] for n in negs}
            self.assertTrue(texts <= colours - {"rojo"},
                            f"non-colour distractor offered: {texts}")
            self.assertNotIn("database", texts)
            self.assertNotIn("volcano", texts)
            self.assertNotIn("router", texts)

    def test_pools_never_mix_taxons(self):
        s = colour_store()
        pools = build_pools(s)
        self.assertEqual({s.labels[v] for v in pools[("person", "Pfav")]},
                         {"rojo", "verde", "azul"})
        self.assertEqual({s.labels[v] for v in pools[("volcano", "Pkind")]},
                         {"volcano", "database"})

    def test_negatives_same_taxon_pool_never_gold(self):
        s = tiny_store()
        indexes = build_indexes(s)
        rng = random.Random(0)
        for _ in range(20):
            negs = sample_negatives(indexes[("city", "P17")], "c1", 1, rng)
            self.assertEqual(negs, ["c2"])

    def test_cross_taxon_isolation(self):
        s = tiny_store()
        indexes = build_indexes(s)
        rng = random.Random(1)
        for _ in range(10):
            negs = sample_negatives(indexes[("city", "P17")], "c1", 1, rng)
            self.assertNotIn("d1", negs)  # the film director is not a country

    def test_index_score_matches_optset(self):
        """The memoized score is `data.optset.difficulty`, not a second notion."""
        labels = {"a": "Reino Unido", "b": "Reino de España", "c": "Australia"}
        index = NearestLabelIndex(["a", "b", "c"], labels)
        for x in ("a", "b", "c"):
            for y in ("a", "b", "c"):
                if x == y:
                    continue
                self.assertAlmostEqual(index.score(x, y),
                                       optset_difficulty(labels[x], labels[y]),
                                       places=9)

    def test_nearest_labels_rank_above_random(self):
        labels = {"g": "Austria", "n1": "Australia", "n2": "Bangladesh",
                  "n3": "Austrasia", "n4": "Perú"}
        index = NearestLabelIndex(list(labels), labels)
        near = index.ranked("g")
        self.assertTrue(near)
        self.assertIn(labels[near[0]], ("Australia", "Austrasia"))
        self.assertGreater(index.score("g", near[0]), index.score("g", "n4"))

    def test_no_duplicate_wording_in_one_option_set(self):
        labels = {"g": "Norte", "a": "Sur", "b": "Sur", "c": "Este"}
        index = NearestLabelIndex(list(labels), labels)
        negs = sample_negatives(index, "g", 2, random.Random(4))
        self.assertEqual(len({labels[n] for n in negs}), 2)

    def test_pool_too_small_raises_instead_of_widening(self):
        s = tiny_store()
        index = build_indexes(s)[("city", "P17")]
        with self.assertRaises(PoolTooSmall):
            sample_negatives(index, "c1", 5, random.Random(0))
        self.assertEqual(len(sample_negatives(index, "c1", 1, random.Random(0))), 1)

    def test_adversary_can_never_select_the_gold(self):
        s = colour_store()
        index = build_indexes(s)[("person", "Pfav")]

        class GoldGrabber:
            def solve(self, state, question, key):
                return question["gold"]

        picks = ["col_verde", "col_azul"]
        out, changed = adversarial_reorder(index, "col_rojo", picks,
                                           GoldGrabber(), "k")
        self.assertNotIn("col_rojo", out)
        self.assertEqual(sorted(out), sorted(picks))
        self.assertFalse(changed)

    def test_adversary_only_permutes(self):
        s = colour_store()
        index = build_indexes(s)[("person", "Pfav")]

        class PicksLast:
            def solve(self, state, question, key):
                return question["options"][-1]["id"]

        picks = ["col_verde", "col_azul"]
        out, changed = adversarial_reorder(index, "col_rojo", picks,
                                           PicksLast(), "k")
        self.assertEqual(sorted(out), sorted(picks))
        self.assertTrue(changed)
        self.assertEqual(out[0], "col_azul")

    def test_adversary_failure_does_not_decide_data(self):
        s = colour_store()
        index = build_indexes(s)[("person", "Pfav")]

        class Broken:
            def solve(self, state, question, key):
                raise RuntimeError("endpoint down")

        picks = ["col_verde", "col_azul"]
        out, changed = adversarial_reorder(index, "col_rojo", picks, Broken(), "k")
        self.assertEqual(out, picks)
        self.assertFalse(changed)


class TestKAndDifficulty(unittest.TestCase):
    def test_k_mix_sums_to_100(self):
        self.assertEqual(sum(w for _, w in K_MIX), 100)

    def test_k_distribution_follows_mix(self):
        rng = random.Random(7)
        ks = [draw_k(rng) for _ in range(20000)]
        for k, w in K_MIX:
            self.assertAlmostEqual(sum(1 for x in ks if x == k) / len(ks),
                                   w / 100, delta=0.02)
        self.assertLessEqual(mix_deviation(ks), 0.02)

    def test_difficulty_25_50_25(self):
        diffs = assign_difficulty([float(i) for i in range(1000)])
        self.assertEqual(sum(1 for d in diffs if d == "easy"), 250)
        self.assertEqual(sum(1 for d in diffs if d == "medium"), 500)
        self.assertEqual(sum(1 for d in diffs if d == "hard"), 250)

    def test_difficulty_is_stable_for_ties(self):
        self.assertEqual(assign_difficulty([1.0] * 8), assign_difficulty([1.0] * 8))


class TestQuality(unittest.TestCase):
    def test_tier_table(self):
        self.assertEqual(score_for("human"), 1.0)
        self.assertEqual(score_for("programmatic"), 1.0)
        self.assertEqual(score_for("teacher", 2), 0.8)
        self.assertEqual(score_for("teacher", 1), 0.55)

    def test_unverified_is_rejected_not_scored(self):
        self.assertIsNone(score_for("unverified"))
        self.assertIsNone(score_for("teacher", 0))
        self.assertIsNone(quality_block("unverified"))

    def test_every_accepted_score_clears_the_floor(self):
        for source, n in (("human", 0), ("programmatic", 0),
                          ("teacher", 2), ("teacher", 1)):
            self.assertGreaterEqual(score_for(source, n), MIN_ACCEPTED_SCORE)

    def test_graph_rows_are_programmatic_1_0(self):
        block = quality_block("programmatic", decider="graph")
        self.assertEqual(block["score"], 1.0)
        self.assertEqual(block["decider"], "graph")


class TestPacks(unittest.TestCase):
    def test_packs_3_to_10_single_split(self):
        qs = [{"id": f"q{i}", "kind": "boolean"} for i in range(23)]
        packs = pack(qs, "state", "wd:X", 0)
        self.assertTrue(packs)
        for p in packs:
            self.assertGreaterEqual(len(p["questions"]), MIN_Q)
            self.assertLessEqual(len(p["questions"]), MAX_Q)
        self.assertEqual(len({p["split"] for p in packs}), 1)

    def test_skips_thin_entities(self):
        self.assertEqual(pack([{"id": "q1"}], "s", "wd:Y", 0), [])

    def test_transform_group_shares_the_parent_split(self):
        s = toy_store()
        indexes = build_indexes(s)
        qs, _, _ = build_entity_qs("city", "e1", s, indexes, random.Random(3), [0])
        packs = pack(qs, state_text("city", "e1", s), "wd:e1", 0)
        self.assertTrue(packs)
        layer_d = make_layer_d(packs[0], "wd:e1", [100], random.Random(1))
        splits = {r["split"] for r in packs} | {r["split"] for r in layer_d}
        self.assertEqual(len(splits), 1)
        for r in layer_d:
            self.assertEqual(r["parent_example_id"], "wd:e1")
            self.assertGreaterEqual(len(r["questions"]), MIN_Q)


class TestTransforms(unittest.TestCase):
    def test_weights_are_controlled(self):
        self.assertEqual(TRANSFORM_WEIGHTS["base"], 1.0)
        for name, w in TRANSFORM_WEIGHTS.items():
            if name != "base":
                self.assertLess(w, 1.0, name)

    def test_transform_share_under_cap(self):
        variants = (["base"] * 100 + ["paraphrase"] * 10
                    + ["option-shuffle"] * 10 + ["irrelevant-context"] * 10)
        self.assertLessEqual(transform_share(variants), MAX_TRANSFORM_SHARE)

    def test_counterfactual_is_not_charged_as_a_transform(self):
        """It changes the answer, so it is a new example, not surface variation."""
        self.assertEqual(transform_share(["base", "counterfactual"]), 0.0)

    def test_counterfactual_half_is_graph_false(self):
        s = toy_store()
        indexes = build_indexes(s)
        qs, _, _ = build_entity_qs("city", "e1", s, indexes, random.Random(3), [0])
        pairs = [q for q in qs if q.get("variant") == "counterfactual"]
        self.assertTrue(pairs)
        for q in pairs:
            self.assertEqual(q["answer"], "no")
            self.assertTrue(q["gold"]["negated"])
            self.assertFalse(s.check(q["gold"]["entity"], q["gold"]["prop"],
                                     q["gold"]["value"]))
            self.assertEqual(verify_question(q, s), [])
            self.assertIn("counterfactual_of", q)

    def test_layer_d_keeps_answers_and_parent(self):
        s = toy_store()
        indexes = build_indexes(s)
        qs, _, _ = build_entity_qs("city", "e1", s, indexes, random.Random(3), [0])
        packs = pack(qs, state_text("city", "e1", s), "wd:e1", 0)
        for rec in make_layer_d(packs[0], "wd:e1", [500], random.Random(2)):
            for src, out in zip(packs[0]["questions"], rec["questions"]):
                self.assertEqual(src["answer"], out["answer"])
                self.assertEqual(src["id"], out["parent_question_id"])
                self.assertEqual(verify_question(out, s), [])

    def test_option_shuffle_keeps_the_answer_binding(self):
        s = toy_store()
        indexes = build_indexes(s)
        qs, _, _ = build_entity_qs("city", "e1", s, indexes, random.Random(3), [0])
        packs = pack(qs, state_text("city", "e1", s), "wd:e1", 0)
        shuffled = next(r for r in make_layer_d(packs[0], "wd:e1", [700],
                                                random.Random(9))
                        if r["variant"] == "option-shuffle")
        for q in shuffled["questions"]:
            self.assertEqual(verify_question(q, s), [])

    def test_filler_is_2k_tokens(self):
        self.assertGreaterEqual(len(filler(2000).split()), 2000)


class TestBuild(unittest.TestCase):
    def test_entity_qs_validate_and_verify(self):
        s = toy_store()
        indexes = build_indexes(s)
        qs, _, _ = build_entity_qs("city", "e1", s, indexes, random.Random(3), [0])
        self.assertGreaterEqual(len(qs), MIN_Q)
        rec = {"state": state_text("city", "e1", s),
               "questions": [{k: v for k, v in q.items()
                              if k in ("id", "kind", "options", "answer")}
                             for q in qs],
               "split": "train", "parent_example_id": "wd:e1"}
        self.assertEqual(validate_record(rec), [])
        for q in qs:
            self.assertEqual(verify_question(q, s), [])
            self.assertEqual(q["quality"]["score"], 1.0)

    def test_validation_keeps_parent_example_id(self):
        """The first pilot rejected 100 % of its rows by dropping this field."""
        from tools.prog_gold.pipeline import strip_for_validation

        rec = {"state": "s", "split": "train", "parent_example_id": "wd:e1",
               "questions": [{"id": "q1", "kind": "boolean", "answer": "yes",
                              "options": [{"id": "yes", "text": "Sí"},
                                          {"id": "no", "text": "No"}]}]}
        self.assertIn("parent_example_id", strip_for_validation(rec))
        self.assertEqual(validate_record(strip_for_validation(rec)), [])

    def test_unlabelled_values_never_become_options(self):
        """Wikidata returns the bare QID when an item has no es/en label."""
        s = toy_store()
        s.labels["Q99999"] = "Q99999"
        s.triples[("e2", "P17")] = {"Q99999"}
        pools = build_pools(s)
        self.assertNotIn("Q99999", pools.get(("city", "P17"), []))
        self.assertFalse(s.has_label("Q99999"))
        self.assertTrue(s.has_label("c1"))
        self.assertEqual(s.labelled_values("e2", "P17"), [])

    def test_deterministic_rebuild(self):
        s = toy_store()
        indexes = build_indexes(s)
        a, _, _ = build_entity_qs("city", "e1", s, indexes, random.Random(3), [0])
        b, _, _ = build_entity_qs("city", "e1", s, indexes, random.Random(3), [0])
        self.assertEqual(a, b)


class TestRobustness(unittest.TestCase):
    def test_irrelevant_context_keeps_lexical_top1(self):
        """§57 — 2k irrelevant tokens must not move the probe's top-1."""
        s = toy_store()
        indexes = build_indexes(s)
        rng = random.Random(5)
        keep = moved = 0
        for ent in ("e1", "e2"):
            qs, _, _ = build_entity_qs("city", ent, s, indexes, rng, [0])
            state = state_text("city", ent, s)
            for q in qs:
                if q["kind"] != "choice":
                    continue
                gold_text = next(o["text"] for o in q["options"]
                                 if o["id"] == q["answer"])
                if probe_top1(state, q["options"]) != gold_text:
                    continue
                if probe_top1(irrelevant_context(state), q["options"]) == gold_text:
                    keep += 1
                else:
                    moved += 1
        self.assertGreater(keep, 0)
        self.assertLessEqual(moved / max(1, keep + moved), 0.05)

    def test_stopwords_are_not_evidence(self):
        """'de' in the filler must not promote a long distractor over the gold."""
        options = [{"id": "o0", "text": "Austria"},
                   {"id": "o1", "text": "Costa de Marfil"}]
        state = "Alba es una ciudad. Su país es Austria."
        self.assertEqual(probe_top1(state, options), "Austria")
        self.assertEqual(probe_top1(irrelevant_context(state), options), "Austria")

    def test_robustness_audit_ignores_rows_the_probe_never_had(self):
        audit = robustness_audit([])
        self.assertEqual(audit["probed"], 0)
        self.assertEqual(audit["top1_shift"], 0.0)


if __name__ == "__main__":
    unittest.main()
