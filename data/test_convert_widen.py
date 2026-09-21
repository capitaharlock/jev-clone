"""Tests for the #T-mix-1m corpus widening (stdlib unittest).

Three properties carry the weight here:

* every new source produces rows the V1 schema accepts and the ONE mixture
  authority (`data/mix.py`) can already tag — family, layer, question type,
  hard/OOD — without a second registry;
* `score` is a real question type, recognised by `data.mix.question_type`
  from the option texts, and its gold is the annotator count the raw TSV
  carries, not a value the converter chose;
* `hard` is MEASURED. A wide option set whose distractors were drawn at
  random would satisfy `is_hard`'s K >= 9 clause and teach nothing, so the
  test asserts the hard tier's distractors are the TOP of the difficulty
  ranking and that the measurement travels with the row.
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path

from data import mix
from data.convert_widen import (
    DBPEDIA_K_HARD,
    GOEMOTIONS_K_HARD,
    _Ranker,
    convert_dbpedia,
    convert_detox,
    convert_goemotions,
    convert_snli,
    convert_swag,
    detox_bucket,
)
from data.schema import Example, Question, validate
from tools.mix_1m import run_mix, sampler
from tools.mix_1m.fence import FENCED_DATASETS

WIDENED = ("dbpedia14", "snli", "goemotions", "detox-attack", "swag")


def read(path) -> list:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def as_example(row: dict) -> Example:
    """Re-validate a written row against the V1 schema from scratch."""
    return Example(
        state=row["state"], split=row["split"],
        questions=[Question(id=q["id"], kind=q["kind"], answer=q["answer"],
                            options=[mix.SOURCES and __import__(
                                "data.schema", fromlist=["Option"]).Option(
                                    id=o["id"], text=o["text"])
                                for o in q["options"]])
                   for q in row["questions"]])


class TestRanker(unittest.TestCase):
    POOL = {"CRIME": "CRIME", "CRIMES": "CRIMES", "PARENTS": "PARENTS",
            "PARENTING": "PARENTING", "MONEY": "MONEY", "TASTE": "TASTE"}

    def test_ranking_is_by_measured_difficulty(self):
        ranker = _Ranker(self.POOL)
        ranked = ranker.ranked("CRIME")
        self.assertEqual(ranked[0][0], "CRIMES")
        self.assertEqual(ranked, sorted(ranked, key=lambda p: (-p[1], p[0])))

    def test_hard_draw_takes_the_nearest_and_records_them(self):
        ranker = _Ranker(self.POOL)
        ids, quality = ranker.draw("CRIME", 3, True, "k")
        self.assertEqual(set(ids), {"CRIME", "CRIMES", ranker.ranked("CRIME")[1][0]})
        self.assertEqual(quality["difficulty"], "hard")
        self.assertGreaterEqual(quality["nn_min"], ranker.ranked("CRIME")[2][1])
        self.assertEqual(quality["pool_size"], len(self.POOL))

    def test_easy_draw_is_not_marked_hard(self):
        _ids, quality = _Ranker(self.POOL).draw("CRIME", 4, False, "k")
        self.assertEqual(quality["difficulty"], "easy")
        self.assertFalse(mix.is_hard({"options": [], "quality": quality}))

    def test_draw_is_deterministic(self):
        a = _Ranker(self.POOL).draw("MONEY", 4, False, "row", 7)
        b = _Ranker(self.POOL).draw("MONEY", 4, False, "row", 7)
        self.assertEqual(a, b)


class TestDetoxBucket(unittest.TestCase):
    def test_zero_annotators_is_zero(self):
        self.assertEqual(detox_bucket(0.0), 0)

    def test_quartiles(self):
        self.assertEqual([detox_bucket(f) for f in (0.1, 0.25, 0.26, 0.5,
                                                    0.51, 0.75, 0.76, 1.0)],
                         [1, 1, 2, 2, 3, 3, 4, 4])


class _Fixtures(unittest.TestCase):
    """One tiny raw corpus per source, written to a temp tree."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.out = self.root / "out"
        self.out.mkdir()
        self.addCleanup(self.tmp.cleanup)

    # -- raw builders ------------------------------------------------------

    def dbpedia(self, n: int = 40) -> Path:
        base = self.root / "dbpedia_csv"
        base.mkdir()
        classes = ["Company", "EducationalInstitution", "Artist", "Athlete",
                   "OfficeHolder", "MeanOfTransportation", "Building",
                   "NaturalPlace", "Village", "Animal", "Plant", "Album",
                   "Film", "WrittenWork"]
        (base / "classes.txt").write_text("\n".join(classes) + "\n")
        with open(base / "train.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            for i in range(n):
                w.writerow([i % 14 + 1, f"Title {i}", f" Abstract number {i}."])
        return base

    def snli(self) -> Path:
        path = self.root / "snli.zip"
        rows = [
            {"gold_label": "entailment", "sentence1": "A man plays a guitar.",
             "sentence2": "A person plays an instrument."},
            {"gold_label": "contradiction", "sentence1": "A man sleeps.",
             "sentence2": "A man runs a marathon."},
            {"gold_label": "-", "sentence1": "No consensus here.",
             "sentence2": "Nobody agreed."},
        ]
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("snli_1.0/snli_1.0_train.jsonl",
                        "\n".join(json.dumps(r) for r in rows))
        return path

    def goemotions(self) -> Path:
        base = self.root / "goemotions"
        base.mkdir()
        emotions = ["admiration", "amusement", "anger", "annoyance",
                    "approval", "caring", "confusion", "curiosity", "desire",
                    "disappointment", "disapproval", "disgust", "excitement",
                    "fear", "gratitude", "joy", "love", "optimism", "pride",
                    "relief", "remorse", "sadness", "surprise", "neutral"]
        head = (["text", "id", "author", "subreddit", "link_id", "parent_id",
                 "created_utc", "rater_id", "example_very_unclear"] + emotions)

        def vote(eid, text, emotion, rater, unclear="False"):
            row = dict.fromkeys(head, "0")
            row.update({"text": text, "id": eid, "author": "a",
                        "subreddit": "s", "link_id": "l", "parent_id": "p",
                        "created_utc": "1", "rater_id": str(rater),
                        "example_very_unclear": unclear})
            if emotion:
                row[emotion] = "1"
            return row

        rows = []
        for r in range(3):                      # clean majority
            rows.append(vote("aaa", "What a lovely thing to say", "love", r))
        rows.append(vote("bbb", "tie between two", "anger", 1))
        rows.append(vote("bbb", "tie between two", "joy", 2))
        for r in range(3):                      # marked unclear
            rows.append(vote("ccc", "???", "anger", r, unclear="True"))
        with open(base / "goemotions_1.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=head)
            w.writeheader()
            w.writerows(rows)
        return base

    def detox(self) -> Path:
        base = self.root / "detox"
        base.mkdir()
        with open(base / "7554634.tsv", "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["rev_id", "comment", "year", "logged_in", "ns",
                        "sample", "split"])
            w.writerow(["1", "you are NEWLINE_TOKEN wrong", "2015", "True",
                        "user", "random", "train"])
            w.writerow(["2", "thanks for the edit", "2015", "True", "user",
                        "random", "train"])
        with open(base / "7554637.tsv", "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["rev_id", "worker_id", "quoting_attack",
                        "recipient_attack", "third_party_attack",
                        "other_attack", "attack"])
            for worker in range(4):             # 3 of 4 call rev 1 an attack
                hit = "1.0" if worker < 3 else "0.0"
                w.writerow(["1", str(worker), "0.0", hit, "0.0", "0.0", hit])
            for worker in range(4):
                w.writerow(["2", str(worker), "0.0", "0.0", "0.0", "0.0",
                            "0.0"])
        return base

    def swag(self) -> Path:
        base = self.root / "swag"
        base.mkdir()
        head = ["", "video-id", "fold-ind", "startphrase", "sent1", "sent2",
                "gold-source", "ending0", "ending1", "ending2", "ending3",
                "label"]
        with open(base / "train.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(head)
            # two rows that SHARE a startphrase and differ in everything
            # else: the dedup must keep both.
            w.writerow([0, "v", 1, "He lifts the bar. He", "He lifts the bar.",
                        "He", "gold", "grunts.", "flies.", "melts.",
                        "evaporates.", 0])
            w.writerow([1, "v", 1, "He lifts the bar. He", "He lifts the bar.",
                        "He", "gold", "sets it down.", "sings opera.",
                        "dissolves.", "teleports.", 0])
        return base


class TestConverters(_Fixtures):
    def test_dbpedia_rows_are_schema_valid_and_tagged(self):
        report = convert_dbpedia(self.dbpedia(), self.out)
        rows = read(self.out / "dbpedia14.jsonl")
        self.assertEqual(report["rows"], len(rows))
        for row in rows:
            for q in row["questions"]:
                texts = [o["text"] for o in q["options"]]
                self.assertIn(q["answer"], [o["id"] for o in q["options"]])
                self.assertEqual(mix.question_type(q["kind"], texts), "choice")

    def test_dbpedia_hard_tier_is_wide_and_measured(self):
        convert_dbpedia(self.dbpedia(120), self.out)
        rows = read(self.out / "dbpedia14.jsonl")
        hard = [q for r in rows for q in r["questions"]
                if q["quality"]["difficulty"] == "hard"]
        easy = [q for r in rows for q in r["questions"]
                if q["quality"]["difficulty"] == "easy"]
        self.assertTrue(hard and easy, "both tiers must be produced")
        for q in hard:
            self.assertGreaterEqual(len(q["options"]), DBPEDIA_K_HARD[0])
            self.assertLessEqual(len(q["options"]), DBPEDIA_K_HARD[1])
            self.assertTrue(mix.is_hard(q))
            self.assertGreater(q["quality"]["nn_mean"], 0.0)
        for q in easy:
            self.assertEqual(len(q["options"]), 4)
            self.assertFalse(mix.is_hard(q))

    def test_dbpedia_never_emits_the_whole_label_space(self):
        convert_dbpedia(self.dbpedia(120), self.out)
        rows = read(self.out / "dbpedia14.jsonl")
        pool = {o["id"] for r in rows for q in r["questions"]
                for o in q["options"]}
        self.assertEqual(len(pool), 14)
        for row in rows:
            for q in row["questions"]:
                self.assertNotEqual({o["id"] for o in q["options"]}, pool)

    def test_snli_emits_a_choice_and_a_boolean_and_skips_no_consensus(self):
        report = convert_snli(self.snli(), self.out)
        rows = read(self.out / "snli.jsonl")
        self.assertEqual(report["rows"], 2)         # the "-" pair is dropped
        for row in rows:
            kinds = [q["kind"] for q in row["questions"]]
            self.assertEqual(kinds, ["choice", "boolean"])
            types = [mix.question_type(q["kind"],
                                       [o["text"] for o in q["options"]])
                     for q in row["questions"]]
            self.assertEqual(types, ["choice", "noul"])
        entail = rows[0]["questions"]
        self.assertEqual(entail[0]["answer"], "entailment")
        self.assertEqual(entail[1]["answer"], "yes")
        self.assertEqual(rows[1]["questions"][1]["answer"], "no")

    def test_goemotions_drops_ties_and_unclear_rows(self):
        report = convert_goemotions(self.goemotions(), self.out)
        rows = read(self.out / "goemotions.jsonl")
        self.assertEqual(report["rows"], 1)
        self.assertEqual(rows[0]["questions"][0]["answer"], "love")

    def test_goemotions_hard_question_is_wide_and_near(self):
        convert_goemotions(self.goemotions(), self.out)
        easy, hard = read(self.out / "goemotions.jsonl")[0]["questions"]
        self.assertEqual(len(easy["options"]), 4)
        self.assertEqual(len(hard["options"]), GOEMOTIONS_K_HARD)
        self.assertFalse(mix.is_hard(easy))
        self.assertTrue(mix.is_hard(hard))
        self.assertEqual(hard["quality"]["votes"], 3)
        self.assertEqual(hard["quality"]["raters"], 3)

    def test_detox_gold_is_the_annotator_count(self):
        convert_detox(self.detox(), self.out)
        rows = {r["state"]: r for r in read(self.out / "detox-attack.jsonl")}
        attacked = rows["you are \n wrong"]
        answers = {q["id"].split("-", 1)[1].rsplit("-", 1)[0]: q["answer"]
                   for q in attacked["questions"]}
        # 3 of 4 annotators -> 0.75 -> the third bucket
        self.assertEqual(answers["attack"], "3")
        self.assertEqual(answers["recipient_attack"], "3")
        clean = rows["thanks for the edit"]
        self.assertTrue(all(q["answer"] == "0" for q in clean["questions"]))

    def test_detox_questions_are_score_type(self):
        convert_detox(self.detox(), self.out)
        for row in read(self.out / "detox-attack.jsonl"):
            for q in row["questions"]:
                texts = [o["text"] for o in q["options"]]
                self.assertEqual(texts, [f"score {i}" for i in range(5)])
                self.assertEqual(mix.question_type(q["kind"], texts), "score")

    def test_swag_keeps_two_questions_that_share_a_state(self):
        report = convert_swag(self.swag(), self.out)
        rows = read(self.out / "swag.jsonl")
        self.assertEqual(report["duplicates_dropped"], 0)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["state"], rows[1]["state"])
        for row in rows:
            q = row["questions"][0]
            self.assertEqual(len(q["options"]), 4)
            self.assertEqual(q["answer"], "o0")
            self.assertEqual(q["quality"]["source"],
                             "adversarial filtering (SWAG)")

    def test_every_written_row_revalidates_against_the_v1_schema(self):
        convert_dbpedia(self.dbpedia(20), self.out)
        convert_snli(self.snli(), self.out)
        convert_goemotions(self.goemotions(), self.out)
        convert_detox(self.detox(), self.out)
        convert_swag(self.swag(), self.out)
        for name in ("dbpedia14", "snli", "goemotions", "detox-attack",
                     "swag"):
            for row in read(self.out / f"{name}.jsonl"):
                self.assertEqual(validate(as_example(row)), [], name)


class TestRegistry(unittest.TestCase):
    def test_every_widened_source_is_registered(self):
        for name in WIDENED:
            self.assertIn(name, mix.SOURCES)
            self.assertIn(name, mix.TRAINABLE_DATASETS)
            self.assertTrue(mix.SOURCES[name].layer)
            self.assertTrue(mix.SOURCES[name].note)

    def test_no_widened_source_touches_the_benchmark_fence(self):
        for name in WIDENED:
            self.assertNotIn(name, mix.MIX_1M_FENCED)
            self.assertNotIn(name, FENCED_DATASETS)
            self.assertIn(name, mix.clean_datasets())

    def test_the_widening_lifts_the_cap_units_past_one(self):
        before = mix.cap_units([d for d in mix.clean_datasets()
                                if d not in WIDENED])
        after = mix.cap_units(mix.clean_datasets())
        self.assertGreater(after, before)
        self.assertGreaterEqual(after, 1.0)

    def test_the_ordinal_and_nli_layers_now_have_supply(self):
        by_layer = mix.layers(mix.clean_datasets())
        self.assertIn("snli", by_layer["nli"])
        self.assertIn("detox-attack", by_layer["preference"])
        self.assertIn("swag", by_layer["adversarial"])

    def test_fixed_option_sources_are_declared_fixed(self):
        # the scale and the 3-way relation ARE the option set: declaring
        # them dynamic would put them through the global-label-space check
        # they cannot pass.
        self.assertFalse(mix.SOURCES["detox-attack"].dynamic_options)
        self.assertFalse(mix.SOURCES["snli"].dynamic_options)
        self.assertTrue(mix.SOURCES["dbpedia14"].dynamic_options)


class TestFeasibilityArithmetic(unittest.TestCase):
    #: real registry ids, because `binding_sources` reports the cap slack
    #: of the FAMILIES these sources sit in — four different families here,
    #: plus the small source that eats the slack.
    SUPPLY = {"civil-comments": 500_000, "dbpedia14": 500_000,
              "snli": 500_000, "goemotions": 500_000, "email-triage": 4_000}

    def test_binding_sources_names_who_is_short(self):
        report = run_mix.binding_sources(self.SUPPLY, 1_000_000)
        self.assertEqual(report["dataset_cap_rows"], 150_000)
        self.assertEqual(report["at_cap"], ["civil-comments", "dbpedia14",
                                            "goemotions", "snli"])
        self.assertEqual(report["short_of_cap"], {"email-triage": 146_000})
        self.assertEqual(report["shortfall_total"], 146_000)
        # the slack the shortfall is measured against is the cap units the
        # registry actually carries over 100 %, not a number of our own
        self.assertEqual(report["slack_rows"],
                         int(1_000_000 * (mix.cap_units(
                             sorted(self.SUPPLY)) - 1.0)))

    def test_ceiling_is_the_largest_target_the_caps_admit(self):
        supply = {d: mix.SOURCES and n for d, n in
                  (("boolq", 9_427), ("email-triage", 4_000),
                   ("civil-comments", 1_804_874), ("huffpost", 162_935),
                   ("massive", 69_084), ("prog-gold", 60_631),
                   ("synth-v1", 34_847))}
        ceiling = mix.max_feasible_target(supply)
        # the finding this task started from: seven sources, 1.05 cap units
        # and still only ~40 k rows, because `email-triage` eats the slack.
        self.assertLess(ceiling, 41_000)
        widened = dict(supply, dbpedia14=560_000, snli=1_097_590,
                       goemotions=87_076, **{"detox-attack": 166_000},
                       swag=73_546)
        self.assertGreater(mix.max_feasible_target(widened), 900_000)


class TestCleanSupplyIsReal(unittest.TestCase):
    """Guards that only run where the converted corpus is on disk."""

    def setUp(self) -> None:
        if not os.path.exists(os.path.join(
                mix.PREFETCH_DIR, "dbpedia14.jsonl")):
            self.skipTest("widened corpus not converted on this machine")

    def test_the_fenced_registry_now_reaches_the_1m_target(self):
        _scan, supply = sampler.clean_supply()
        feas = run_mix.corpus_feasibility(supply)
        self.assertGreaterEqual(feas["achievable_rows"], 900_000)
        self.assertTrue(feas["why"])


if __name__ == "__main__":
    unittest.main()
