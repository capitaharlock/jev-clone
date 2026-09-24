"""Unit tests for eval.optiontext — the three arms and the R8 budget.

No torch and no checkpoint: everything here is the part of #T-option-text
that decides what the model is SHOWN, which is exactly the part a reader has
to be able to check without a GPU. The scoring itself is `eval.fullspace`'s
`tally`, tested there.
"""
import unittest
from dataclasses import replace

from data import taxonomy as TX
from data.optset import Sample

from . import optiontext as O


def toy_sample(state="my card has not arrived yet", gold="card_arrival"):
    labels = sorted(["card_arrival", "card_linking", "exchange_rate"])
    return Sample(dataset="banking77", row_id="banking77-0",
                  question_id="banking77-intent-0", state=state,
                  question="banking77-intent-0",
                  options=[{"id": lab, "text": lab} for lab in labels],
                  answer=gold, gold_index=labels.index(gold))


def toy_support(n=40):
    labels = ["card_arrival", "card_linking", "exchange_rate"]
    return [{"id": f"banking77:banking77-{i}:q", "label": labels[i % 3],
             "text": f"row {i} about my card arriving late and the rate",
             "norm": O.normalise(f"row {i} about my card arriving late "
                                 "and the rate")}
            for i in range(n)]


class TestRetrieval(unittest.TestCase):
    def test_terms_are_words_and_adjacent_pairs(self):
        self.assertEqual(O.terms("my card"), ["my", "card", "my_card"])

    def test_retrieval_caps_examples_per_class(self):
        idx = O.BM25(toy_support())
        got = idx.retrieve("card arriving late", want=24, per_class=4)
        per = {}
        for ex in got:
            per[ex["label"]] = per.get(ex["label"], 0) + 1
        self.assertTrue(all(v <= 4 for v in per.values()), per)
        self.assertLessEqual(len(got), 24)

    def test_retrieval_never_returns_the_query_back_as_a_demonstration(self):
        pool = toy_support()
        idx = O.BM25(pool)
        query = pool[0]["text"]
        got = idx.retrieve(query, banned=O.normalise(query))
        self.assertNotIn(pool[0]["id"], [e["id"] for e in got])

    def test_retrieval_is_deterministic(self):
        idx = O.BM25(toy_support())
        self.assertEqual([e["id"] for e in idx.retrieve("card rate")],
                         [e["id"] for e in idx.retrieve("card rate")])


class TestTheArms(unittest.TestCase):
    def setUp(self):
        self.base = [toy_sample()]
        self.desc = {"card_arrival": "A card has not arrived.",
                     "card_linking": "Linking an existing card.",
                     "exchange_rate": "The rate applied to a conversion."}
        self.index = O.BM25(toy_support())

    def test_arm_a_is_the_repo_as_it_ships(self):
        samples, segs = O.arm_samples("A", self.base, self.desc, None)
        self.assertTrue(all(o["id"] == o["text"]
                            for o in samples[0].options))
        self.assertEqual(segs, [None])

    def test_arm_b_changes_only_the_option_text(self):
        samples, _ = O.arm_samples("B", self.base, self.desc, self.index)
        a, b = self.base[0], samples[0]
        self.assertEqual(a.state, b.state)
        self.assertEqual([o["id"] for o in a.options],
                         [o["id"] for o in b.options])
        self.assertEqual(a.gold_index, b.gold_index)
        self.assertIn("A card has not arrived.", b.options[0]["text"])

    def test_every_arm_keeps_the_same_gold_in_the_same_position(self):
        for arm in ("A", "B", "C", "C-teacher-order"):
            samples, _ = O.arm_samples(arm, self.base, self.desc, self.index)
            self.assertEqual(samples[0].gold_index, self.base[0].gold_index)
            self.assertEqual(samples[0].answer, self.base[0].answer)
            self.assertEqual(len(samples[0].options), 3)

    def test_arm_c_puts_the_query_first_and_the_teacher_order_last(self):
        c, segs = O.arm_samples("C", self.base, self.desc, self.index)
        self.assertTrue(c[0].state.startswith(self.base[0].state))
        self.assertEqual(segs[0]["spans"][0]["kind"], "query")
        t, tsegs = O.arm_samples("C-teacher-order", self.base, self.desc,
                                 self.index)
        self.assertTrue(t[0].state.endswith(self.base[0].state))
        self.assertEqual(tsegs[0]["spans"][-1]["kind"], "query")
        # same material, different order: the budget is the only difference
        self.assertEqual(sorted(x for x in t[0].state.split("\n") if x),
                         sorted(x for x in c[0].state.split("\n") if x))

    def test_an_unknown_arm_is_refused(self):
        with self.assertRaises(ValueError):
            O.arm_samples("D", self.base, self.desc, self.index)

    def test_arm_c_without_an_index_is_refused(self):
        with self.assertRaises(ValueError):
            O.arm_samples("C", self.base, self.desc, None)


class FakeTokenizer:
    """One token per whitespace word, with exact char offsets.

    The budget has to be measured on the SAME strings that are scored; what
    it must not depend on is which tokenizer measured them, so the test
    supplies a trivial one and asserts the accounting, not the vocabulary.
    """

    def __call__(self, text, truncation=False, max_length=None,
                 return_offsets_mapping=False, add_special_tokens=True):
        spans, at = [], 0
        for word in text.split(" "):
            if word:
                start = text.index(word, at)
                spans.append((start, start + len(word)))
                at = start + len(word)
        if truncation and max_length is not None:
            spans = spans[:max_length]
        out = {"input_ids": list(range(len(spans)))}
        if return_offsets_mapping:
            out["offset_mapping"] = spans
        return out


class TestTheBudget(unittest.TestCase):
    def setUp(self):
        self.tok = FakeTokenizer()
        self.desc = {lab: "definition" for lab in
                     ("card_arrival", "card_linking", "exchange_rate")}
        self.index = O.BM25(toy_support())

    def test_a_state_that_fits_loses_nothing(self):
        base = [toy_sample()]
        samples, segs = O.arm_samples("A", base, self.desc, None)
        rep = O.budget(self.tok, samples, segs, window=256)
        self.assertEqual(rep["rows_truncated"], 0)
        self.assertEqual(rep["tokens_dropped_total"], 0)
        self.assertEqual(rep["query_survives_share"], 1.0)
        self.assertEqual(rep["examples_offered_total"], 0)

    def test_query_first_keeps_the_query_when_the_window_bites(self):
        base = [toy_sample()]
        samples, segs = O.arm_samples("C", base, self.desc, self.index)
        rep = O.budget(self.tok, samples, segs, window=12)
        self.assertEqual(rep["query_survives_share"], 1.0)
        self.assertGreater(rep["rows_truncated"], 0)
        self.assertLess(rep["examples_complete_total"],
                        rep["examples_offered_total"])
        self.assertIn("first", rep["query_position"])

    def test_teacher_order_loses_the_query_to_the_same_window(self):
        base = [toy_sample()]
        samples, segs = O.arm_samples("C-teacher-order", base, self.desc,
                                      self.index)
        rep = O.budget(self.tok, samples, segs, window=12)
        self.assertEqual(rep["query_survives_whole"], 0)
        self.assertIn("after the examples", rep["query_position"])

    def test_an_example_counts_only_when_it_survives_WHOLE(self):
        base = [toy_sample()]
        samples, segs = O.arm_samples("C", base, self.desc, self.index)
        full = O.budget(self.tok, samples, segs, window=10_000)
        self.assertEqual(full["examples_complete_total"],
                         full["examples_offered_total"])
        self.assertEqual(full["examples_complete_share"], 1.0)


class TestTheVerdict(unittest.TestCase):
    @staticmethod
    def arm(name, acc, lo, hi, beats=False):
        return {"arm": name, "accuracy": acc, "accuracy_ci95": [lo, hi],
                "chance": 0.012987, "cardinality": 77, "n": 1000,
                "beats_chance": beats}

    def test_flat_arms_leave_the_denominator_first(self):
        arms = {"A": self.arm("A", 0.009, 0.0047, 0.017),
                "B": self.arm("B", 0.011, 0.006, 0.020),
                "C": self.arm("C", 0.010, 0.005, 0.019)}
        v = O.verdict(arms)
        self.assertEqual(v["arms_that_move_the_number"], [])
        self.assertFalse(v["any_arm_beats_chance"])
        self.assertIn("first", v["fullspace_objective_priority"])
        self.assertIn("explains none", v["line"])
        self.assertIn("K=77", v["line"])
        self.assertIn("chance", v["line"])

    def test_an_arm_that_clears_the_baseline_shares_the_explanation(self):
        arms = {"A": self.arm("A", 0.009, 0.0047, 0.017),
                "B": self.arm("B", 0.300, 0.272, 0.330, beats=True)}
        v = O.verdict(arms)
        self.assertEqual(v["arms_that_move_the_number"], ["B"])
        self.assertEqual(v["fullspace_objective_priority"], "shared")
        self.assertAlmostEqual(v["share_of_gap_explained_by_input"],
                               (0.300 - 0.009) / (0.924 - 0.009), places=5)

    def test_the_diagnostic_arm_never_becomes_the_best_arm(self):
        arms = {"A": self.arm("A", 0.009, 0.0047, 0.017),
                "C": self.arm("C", 0.010, 0.005, 0.019),
                "C-teacher-order": self.arm("C-teacher-order", 0.9,
                                            0.88, 0.92, beats=True)}
        self.assertEqual(O.verdict(arms)["best_measured_arm"], "C")


class TestTheOptionTextItself(unittest.TestCase):
    def test_arm_b_uses_the_pinned_taxonomy_not_an_invented_string(self):
        desc = TX.load()
        base = [replace(toy_sample(),
                        options=[{"id": lab, "text": lab}
                                 for lab in sorted(desc)[:3]])]
        samples, _ = O.arm_samples("B", base, desc, None)
        for opt in samples[0].options:
            self.assertIn(desc[opt["id"]], opt["text"])


if __name__ == "__main__":
    unittest.main()


class TestTheArtifactText(unittest.TestCase):
    """The derived prose is arithmetic over the numbers, not a narration."""

    @staticmethod
    def doc(with_confirmation=False):
        arm = {"arm": "A", "accuracy": 0.009, "accuracy_ci95": [0.0047, 0.017],
               "chance": 0.012987, "cardinality": 77, "n": 1000, "hits": 9,
               "beats_chance": False}
        armb = {**arm, "arm": "B", "accuracy": 0.014,
                "accuracy_ci95": [0.0084, 0.0234], "hits": 14}
        armc = {**arm, "arm": "C", "accuracy": 0.013,
                "accuracy_ci95": [0.0076, 0.0221], "hits": 13,
                "context_budget": {"examples_complete_mean": 11.9,
                                   "examples_offered_mean": 24.0,
                                   "window_tokens": 256}}
        out = {"cut": {"name": "banking77-dev"},
               "arms": {"A": arm, "B": armb, "C": armc}}
        out["verdict"] = O.verdict(out["arms"])
        if with_confirmation:
            test = {"arm": "A", "accuracy": 0.012338, "hits": 38, "n": 3080,
                    "accuracy_ci95": [0.009, 0.016888], "chance": 0.012987,
                    "cardinality": 77, "beats_chance": False}
            out["reserved_cut_confirmation"] = {
                "cut": {"name": "banking77-test"}, "arms": {"A": test}}
        return out

    def test_a_dev_only_line_names_the_cut_it_was_measured_on(self):
        line = O.final_verdict(self.doc())
        self.assertTrue(line.startswith("[banking77-dev]"))
        self.assertIn("#T-fullspace-objective", line)

    def test_the_line_carries_the_reserved_cut_when_there_is_one(self):
        line = O.final_verdict(self.doc(with_confirmation=True))
        self.assertIn("banking77-test", line)
        self.assertIn("RESERVED", line)
        self.assertIn("38/3080", line)
        # the budget finding travels with it, not only the accuracies
        self.assertIn("256-token window", line)

    def test_examples_source_does_not_claim_an_empty_support_pool(self):
        card = {"examples_source": {"dataset": "banking77"}}
        with_c = O.examples_source(card, {"rows": 9003, "labels": 77,
                                          "excluded": 1000}, ("A", "C"))
        self.assertTrue(with_c["retrieved_in_this_read"])
        self.assertEqual(with_c["support_rows_indexed"], 9003)
        without = O.examples_source(card, {"rows": 0}, ("A", "B"))
        self.assertFalse(without["retrieved_in_this_read"])
        self.assertNotIn("support_rows_indexed", without)

    def test_finalise_puts_the_verdict_line_last_and_reads_nothing(self):
        doc = self.doc(with_confirmation=True)
        doc["sources"] = {"examples": {}}
        doc["reserved_cut_confirmation"]["sources"] = {"examples": {}}
        doc["verdict_line"] = "stale"
        doc["trailing"] = "x"
        out = O.finalise(doc)
        self.assertEqual(list(out)[-1], "verdict_line")
        self.assertNotEqual(out["verdict_line"], "stale")
        self.assertIn("banking77-test", out["verdict_line"])
