"""Unit tests for eval.parity — #T-jev-parity.

The subject here is almost never arithmetic: it is what the artifact CLAIMS.
Four things this task exists to stop, each nailed down below:

* `same_rows: false` read as "different rows". Nobody compared the row sets;
  one side publishes no row list. Not-verified is a third state and the
  artifact has to say so in words.
* a protocol difference described in prose instead of priced. The three
  #T-option-text arms are numbers, and they only subtract inside one cut.
* the contamination claim widened from what the evidence carries. Excluding
  the banking77 DATASET is not the absence of its label STRINGS, and a label
  identifier in another corpus is not the same finding as the English phrase
  turning up in somebody's sentence.
* a report quoting somebody else's parity number because none was measured
  for the checkpoint it is about.

No model is loaded: the scorer is stubbed and the corpus is a tmpdir.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from data.optset import Sample

from . import fullspace as F
from . import gate_rules as G
from . import parity as P

LABELS = [f"label_{i:02d}" for i in range(P.CARDINALITY)]


class FakeCut:
    name = "banking77-test"
    reserved = True
    dataset = "banking77"
    split = "test"


def samples(n: int, golds: list | None = None) -> list:
    """`n` rows over the full 77-label space, gold chosen per row."""
    options = [{"id": lab, "text": lab} for lab in LABELS]
    golds = golds or [LABELS[0]] * n
    return [Sample(dataset="banking77", row_id=f"banking77-{i}",
                   question_id=f"banking77-intent-{i}", state=f"state {i}",
                   question="question", options=list(options),
                   answer=golds[i], gold_index=LABELS.index(golds[i]))
            for i in range(n)]


def entries(rows: list, correct: int = 0) -> list:
    """One scored entry per row; the first `correct` of them are right."""
    out = []
    for i, row in enumerate(rows):
        probs = [0.001] * (P.CARDINALITY + 1)
        pred = row.gold_index if i < correct else (row.gold_index + 1) % \
            P.CARDINALITY
        probs[pred] = 0.9
        out.append({"probs": probs, "pred": pred, "id": P.sample_id(row)})
    return out


def seal(ids: list) -> dict:
    return {
        "rows": len(ids),
        "ids": ids,
        "manifest": "artifacts/splits/T-eval-cardinality-banking77-test/"
                    "manifest.json",
        "manifest_sha256": "a" * 64,
        "split_sha256": "b" * 64,
    }


RUN_FENCED = {
    "run_id": "a-run",
    "datasets": ["boolq", "massive", "snli"],
    "fence": {"clean_1m": True, "fenced_datasets": ["banking77"],
              "why": "§§18, 77"},
    "data_manifest": {},
}
CONTAM_CLEAN = {
    "generated_utc": "2026-09-24T00:00:00Z",
    "labels_found": {}, "labels_found_total": 0,
    "labels_found_as_identifier": {},
    "labels_found_as_identifier_total": 0,
    "labels_found_as_answer": {}, "labels_found_as_answer_total": 0,
    "declared_fence": RUN_FENCED["fence"],
    "reading": "nothing found",
    "not_auditable": {"what": "the backbone"},
}


def doc(n: int = 20, correct: int = 1, golds: list | None = None,
        run: dict | None = None, contamination: dict | None = None,
        regime: dict | None = None) -> dict:
    rows = samples(n, golds)
    scored = entries(rows, correct)
    ids = [P.sample_id(r) for r in rows]
    return P.compose(FakeCut(), "artifacts/checkpoints/decision/x/stage-1",
                     {"model_version": "mv", "weights_sha256": "c" * 64,
                      "run_id": "a-run"},
                     seal(ids), run or RUN_FENCED, rows, scored,
                     contamination or CONTAM_CLEAN,
                     regime if regime is not None else {"measured": False},
                     None, {"device": "cpu", "rows": n, "seconds": 1.0})


class TestSameRowsSaysWhatIsNotVerified(unittest.TestCase):
    def test_false_does_not_mean_different_rows(self):
        reading = doc()["same_rows_reading"]
        self.assertFalse(reading["verified"])
        self.assertIn("NOT ESTABLISHED", reading["value_means"])
        self.assertIn("nobody has compared them", reading["value_means"])

    def test_the_phrase_the_task_struck_out_is_gone(self):
        """`eval/fullspace.py` used to say "different rows" — a claim."""
        blob = json.dumps(doc(), ensure_ascii=False).lower()
        self.assertNotIn("different rows may have been sampled", blob)
        for source in (json.dumps(F.SAME_ROWS_READING).lower(),
                       " ".join(F.SAME_ROWS_NOT_VERIFIED).lower()):
            self.assertNotIn("different rows, and", source)

    def test_it_names_the_two_things_that_are_unverified(self):
        reading = doc()["same_rows_reading"]
        blob = " ".join(reading["what_is_NOT_verified"])
        self.assertIn("never been scored with our pipeline", blob)
        self.assertIn("no row-by-row evidence", blob)
        self.assertIn("#T-teacher-probe", reading["what_would_verify_it"])

    def test_the_two_gates_tell_the_same_story(self):
        """One list, shared, so `eval.fullspace` and `eval.parity` cannot
        drift into two accounts of the same absence of evidence."""
        self.assertEqual(doc()["same_rows_reading"]["what_is_NOT_verified"],
                         list(F.SAME_ROWS_NOT_VERIFIED))

    def test_the_value_stays_a_boolean_false(self):
        """R3 stamps `same_rows: false` and readers test it as a bool; a
        dict there would be TRUTHY and would invert every such test."""
        self.assertIs(doc()["same_rows"], False)


class TestProtocolDifferencesAreEnumerated(unittest.TestCase):
    def setUp(self):
        self.diffs = doc()["protocol_differences"]

    def test_every_difference_carries_both_sides_and_a_verdict(self):
        for diff in self.diffs:
            with self.subTest(diff=diff["id"]):
                for key in ("id", "dimension", "teacher", "ours", "matched"):
                    self.assertTrue(diff.get(key) is not None, key)
                self.assertIn(diff["matched"], (True, False, "unverifiable"))
                self.assertTrue(diff.get("evidence") or diff.get("measured"))

    def test_the_dimensions_the_task_names_are_all_there(self):
        ids = {d["id"] for d in self.diffs}
        self.assertLessEqual(
            {"label-space", "split", "scored-by", "row-evidence",
             "option-text", "labelled-examples", "context-budget",
             "abstention", "training-data-audit", "backbone-exposure"}, ids)

    def test_cardinality_is_the_one_dimension_that_matches(self):
        by_id = {d["id"]: d for d in self.diffs}
        self.assertIs(by_id["label-space"]["matched"], True)
        self.assertIn(str(P.CHANCE), by_id["label-space"]["ours"])

    def test_unverifiable_is_not_rounded_to_false(self):
        by_id = {d["id"]: d for d in self.diffs}
        self.assertEqual(by_id["split"]["matched"], "unverifiable")
        self.assertEqual(by_id["backbone-exposure"]["matched"],
                         "unverifiable")

    def test_the_backbone_is_declared_apart_from_the_corpus_fence(self):
        by_id = {d["id"]: d for d in self.diffs}
        self.assertIn("unauditable", by_id["backbone-exposure"]["ours"])
        self.assertIn("separately", by_id["backbone-exposure"]["evidence"])


class TestRowsAreIdentified(unittest.TestCase):
    def test_the_ids_are_stable_and_digested(self):
        first, second = doc(12), doc(12)
        self.assertEqual(first["rows"]["ids_sha256"],
                         second["rows"]["ids_sha256"])
        self.assertEqual(first["rows"]["n"], 12)
        self.assertTrue(first["rows"]["unique"])

    def test_the_digest_is_of_a_SET_not_an_order(self):
        """Scoring the same rows in another order is the same row-set."""
        rows = samples(8)
        ids = [P.sample_id(r) for r in rows]
        one = P.row_identity([{"id": i} for i in ids], seal(ids), None)
        other = P.row_identity([{"id": i} for i in reversed(ids)],
                               seal(ids), None)
        self.assertEqual(one["ids_sha256"], other["ids_sha256"])

    def test_scoring_a_subset_of_the_seal_is_visible(self):
        rows = samples(4)
        ids = [P.sample_id(r) for r in rows]
        report = P.row_identity([{"id": i} for i in ids[:2]], seal(ids), None)
        self.assertFalse(report["sealed_cut"]["covers_the_whole_sealed_cut"])

    def test_a_row_id_names_a_row_of_the_official_split(self):
        self.assertEqual(P.sample_id(samples(1)[0]),
                         "banking77:banking77-0:banking77-intent-0")


class TestGoldExposureSplitsInsideTheSameK(unittest.TestCase):
    def test_a_fenced_run_has_every_gold_on_the_unseen_side(self):
        exposure = doc(10)["gold_exposure"]
        self.assertEqual(exposure["n_seen"], 0)
        self.assertEqual(exposure["n_unseen"], P.CARDINALITY)
        self.assertEqual(exposure["by_side"]["unseen_gold"]["n"], 10)
        self.assertEqual(exposure["by_side"]["seen_gold"]["n"], 0)
        self.assertIn("why_empty", exposure["by_side"]["seen_gold"])

    def test_a_label_that_is_a_gold_answer_elsewhere_counts_as_seen(self):
        contam = dict(CONTAM_CLEAN,
                      labels_found_as_answer={LABELS[0]: {"massive": 3}},
                      labels_found_as_answer_total=1)
        golds = [LABELS[0]] * 6 + [LABELS[1]] * 4
        exposure = doc(10, golds=golds, contamination=contam)["gold_exposure"]
        self.assertEqual(exposure["seen"], [LABELS[0]])
        self.assertEqual(exposure["by_side"]["seen_gold"]["n"], 6)
        self.assertEqual(exposure["by_side"]["unseen_gold"]["n"], 4)
        self.assertTrue(exposure["by_side"]["covers_every_row"])

    def test_training_on_the_dataset_makes_every_label_seen(self):
        run = {"datasets": ["banking77", "massive"],
               "fence": {"clean_1m": False, "fenced_datasets": []}}
        exposure = P.seen_labels(run, CONTAM_CLEAN, LABELS)
        self.assertEqual(exposure["n_seen"], P.CARDINALITY)
        self.assertTrue(exposure["basis"]["dataset_in_mixture"])

    def test_both_sides_keep_the_same_cardinality_and_chance(self):
        """Not two cuts and not two Ks: the same rows, partitioned."""
        contam = dict(CONTAM_CLEAN,
                      labels_found_as_answer={LABELS[0]: {"massive": 1}})
        golds = [LABELS[0]] * 5 + [LABELS[1]] * 5
        sides = doc(10, golds=golds,
                    contamination=contam)["gold_exposure"]["by_side"]
        for side in ("seen_gold", "unseen_gold"):
            self.assertEqual(sides[side]["cardinality"], P.CARDINALITY)
            self.assertEqual(sides[side]["chance"], P.CHANCE)


class TestTheDistancesAreBothPublished(unittest.TestCase):
    def test_the_old_number_is_frozen_as_the_baseline(self):
        self.assertEqual(P.BASELINE["accuracy"], 0.012338)
        self.assertEqual(P.BASELINE["hits"], 38)
        self.assertEqual(P.BASELINE["n"], P.OFFICIAL_TEST_ROWS)
        self.assertIn("0,0123", P.BASELINE["published_as"])
        self.assertIn("T-teacher-probe", P.BASELINE["artifact"])

    def test_every_run_publishes_its_distance_to_both(self):
        dist = doc(100, correct=50)["distances"]
        self.assertAlmostEqual(dist["to_baseline"]["delta"],
                               0.5 - 0.012338, places=6)
        self.assertAlmostEqual(dist["to_teacher"]["delta"],
                               0.924 - 0.5, places=6)
        self.assertTrue(dist["to_teacher"]["teacher_is_a_citation"])

    def test_overlapping_intervals_are_not_called_a_move(self):
        dist = doc(3080, correct=38)["distances"]["to_baseline"]
        self.assertTrue(dist["intervals_overlap"])
        self.assertIn("has not been shown to differ", dist["reading"])

    def test_chance_is_the_first_distance(self):
        dist = doc()["distances"]["chance"]
        self.assertEqual(dist["chance"], P.CHANCE)
        self.assertIn("R2", dist["reading"])


class TestTheInformationRegimeIsANumber(unittest.TestCase):
    #: the shape `artifacts/gates/T-option-text/optiontext.json` has
    ARTIFACT = {
        "checkpoint": "ckpt",
        "cut": {"name": "banking77-dev", "reserved": False},
        "arms": {
            "A": {"arm": "A", "n": 1000, "hits": 9, "cardinality": 77,
                  "chance": 0.012987, "accuracy": 0.009,
                  "accuracy_ci95": [0.004742, 0.017016], "beats_chance": False,
                  "protocol": {"option_text": "raw identifier",
                               "labelled_examples_in_state": 0},
                  "context_budget": {}},
            "C": {"arm": "C", "n": 1000, "hits": 13, "cardinality": 77,
                  "chance": 0.012987, "accuracy": 0.013,
                  "accuracy_ci95": [0.007613, 0.022114], "beats_chance": False,
                  "protocol": {"option_text": "definitions",
                               "labelled_examples_in_state": 24},
                  "context_budget": {"window_tokens": 256,
                                     "examples_offered_mean": 24.0,
                                     "examples_complete_mean": 11.9,
                                     "query_survives_share": 1.0}},
        },
        "verdict": {"point_shift_best_vs_baseline": 0.005,
                    "share_of_gap_explained_by_input": 0.005464,
                    "arms_that_move_the_number": [], "line": "a line"},
        "reserved_cut_confirmation": {
            "checkpoint": "ckpt",
            "cut": {"name": "banking77-test", "reserved": True},
            "arms": {
                "A": {"arm": "A", "n": 3080, "hits": 38, "cardinality": 77,
                      "chance": 0.012987, "accuracy": 0.012338,
                      "accuracy_ci95": [0.009002, 0.016888],
                      "beats_chance": False,
                      "protocol": {"option_text": "raw identifier",
                                   "labelled_examples_in_state": 0}},
                "B": {"arm": "B", "n": 3080, "hits": 37, "cardinality": 77,
                      "chance": 0.012987, "accuracy": 0.012013,
                      "accuracy_ci95": [0.008728, 0.016514],
                      "beats_chance": False,
                      "protocol": {"option_text": "definitions",
                                   "labelled_examples_in_state": 0}},
            },
        },
    }

    def regime(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "optiontext.json"
            path.write_text(json.dumps(self.ARTIFACT))
            return P.information_regime(path)

    def test_every_arm_arrives_with_its_cut(self):
        regime = self.regime()
        self.assertTrue(regime["measured"])
        self.assertEqual(regime["arms_by_cut"]["banking77-test"]["A"]["n"],
                         3080)
        self.assertEqual(regime["arms_by_cut"]["banking77-dev"]["C"]["n"],
                         1000)

    def test_arm_c_is_not_dropped_for_being_dev_only(self):
        """It is the whole labelled-examples difference; without it that
        row of the table goes back to being prose."""
        regime = self.regime()
        self.assertIn("C", regime["arms"])
        self.assertEqual(
            regime["arms"]["C"]["labelled_examples_in_state"], 24)

    def test_a_delta_is_only_computed_inside_one_cut(self):
        arithmetic = self.regime()["arithmetic"]
        self.assertIn("C - A on banking77-dev", arithmetic["deltas"])
        self.assertIn("B - A on banking77-test", arithmetic["deltas"])
        for name in arithmetic["deltas"]:
            self.assertRegex(name, r" on banking77-(dev|test)$")

    def test_a_shift_inside_overlapping_intervals_is_not_a_move(self):
        delta = self.regime()["arithmetic"]["deltas"]["C - A on banking77-dev"]
        self.assertAlmostEqual(delta["point_shift"], 0.004, places=6)
        self.assertTrue(delta["intervals_overlap"])
        self.assertIn("is not a move", delta["reading"])

    def test_the_budget_finding_travels_with_it(self):
        budget = self.regime()["budget_finding"]
        self.assertEqual(budget["window_tokens"], 256)
        self.assertEqual(budget["examples_complete_mean"], 11.9)
        self.assertIn("R8", budget["reading"])

    def test_the_arms_reach_the_protocol_table(self):
        diffs = doc(regime=self.regime())["protocol_differences"]
        by_id = {d["id"]: d for d in diffs}
        self.assertIn("37/3080", by_id["option-text"]["measured"])
        self.assertIn("13/1000", by_id["labelled-examples"]["measured"])
        self.assertIn("256-token", by_id["context-budget"]["measured"])

    def test_a_missing_artifact_says_so_instead_of_guessing(self):
        regime = P.information_regime(Path("/nowhere/optiontext.json"))
        self.assertFalse(regime["measured"])
        self.assertIn("eval.optiontext", regime["how"])


class TestContaminationIsDeclaredWithPrecision(unittest.TestCase):
    def corpus(self, tmp: Path, rows: list) -> Path:
        path = tmp / "other.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return path

    def test_the_identifier_and_the_english_phrase_are_counted_apart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.corpus(Path(tmp), [
                {"state": "nothing to see", "questions": [
                    {"id": "q", "options": [{"id": "label_00"}],
                     "answer": "label_00"}]},
                {"state": "the label 00 of this sentence", "questions": []},
            ])
            rep = P.scan_files([path], LABELS)
        self.assertEqual(rep["occurrences"]["label_00"],
                         {P.FORM_IDENTIFIER: 2, P.FORM_PHRASE: 1})
        self.assertEqual(rep["as_gold_answer"], {"label_00": 1})
        self.assertEqual(rep["as_option_id"], {"label_00": 1})

    def test_a_clean_corpus_reports_absence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.corpus(Path(tmp), [
                {"state": "an ordinary sentence", "questions": []}])
            rep = P.scan_files([path], LABELS)
        self.assertEqual(rep["occurrences"], {})
        self.assertEqual(rep["labels_found_as_identifier"], [])

    def test_a_label_is_not_matched_inside_a_longer_identifier(self):
        rx = P.label_regex(["card_arrival"])
        self.assertEqual(rx.findall(b"card_arrival_pending"), [])
        self.assertEqual(rx.findall(b"xcard_arrival"), [])
        self.assertEqual(rx.findall(b"my card_arrival now"), [b"card_arrival"])
        self.assertEqual(rx.findall(b"my card arrival now"), [b"card arrival"])

    def test_the_reading_leads_with_the_finding_that_would_matter(self):
        labels = ["a_b", "c_d"]
        phrase_only = P._contamination_reading(
            labels, {"massive": {}}, {"a_b": {}}, {}, {})
        self.assertIn("no banking77 label occurs as an IDENTIFIER",
                      phrase_only)
        self.assertIn("collision of language", phrase_only)
        found = P._contamination_reading(
            labels, {"massive": {}}, {"a_b": {}}, {"a_b": {}},
            {"a_b": {"massive": 2}})
        self.assertIn("occur as IDENTIFIERS", found)
        self.assertIn("GOLD ANSWERS", found)

    def test_the_fence_and_the_string_scan_are_different_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.corpus(Path(tmp), [{"state": "clean", "questions": []}])
            with mock.patch("data.optset.source_paths",
                            return_value=[str(Path(tmp) / "other.jsonl")]), \
                 mock.patch("data.taxonomy.load",
                            return_value={k: "" for k in LABELS}):
                scan = P.contamination_scan(RUN_FENCED, write=False,
                                            log=lambda *a: None)
        fence = scan["declared_fence"]
        self.assertIn("banking77", fence["fenced_datasets"])
        self.assertIn("zero rows", fence["proves"])
        self.assertIn("LABEL STRING", fence["does_not_prove"])
        self.assertIn("unmeasurable", scan["not_auditable"]["why"])

    def test_the_multi_shard_sources_are_not_silently_skipped(self):
        """`prog-gold` is four files and `synth-v1` six; looking for one
        `<name>.jsonl` quietly dropped two of the twelve sources."""
        with tempfile.TemporaryDirectory() as tmp:
            a = self.corpus(Path(tmp), [{"state": "x", "questions": []}])
            b = Path(tmp) / "shard-b.jsonl"
            b.write_text(json.dumps({"state": "label 00 here",
                                     "questions": []}) + "\n")
            rep = P.scan_files([a, b], LABELS)
        self.assertEqual(len(rep["paths"]), 2)
        self.assertEqual(rep["occurrences"]["label_00"],
                         {P.FORM_PHRASE: 1})


class TestTheArtifactIsReadableAndCoherent(unittest.TestCase):
    def test_no_accuracy_is_published_without_its_companions(self):
        """C4 of `eval.gate_rules` — rule R2, on our own output."""
        claims = G.accuracy_claims(doc(100, correct=3))
        self.assertTrue(claims)
        bare = [c for c in claims if c["missing"] and not c["citation"]]
        self.assertEqual(bare, [], bare)

    def test_the_line_carries_k_chance_n_and_the_interval(self):
        line = doc(3080, correct=38)["parity_line"]
        self.assertIn("K=77", line)
        self.assertIn("chance 0.012987", line)
        self.assertIn("38/3080", line)
        self.assertIn("CI [0.009002, 0.016888]", line)
        self.assertIn("UNVERIFIED", line)

    def test_the_artifact_claims_no_green_verdict(self):
        """A parity artifact publishes a distance, never a PASS: a green
        claim would drag C1-C3 in behind a number that decides nothing."""
        self.assertFalse(G.green_claim(doc())["green"])

    def test_the_render_survives_a_real_artifact(self):
        text = P.render(doc(50, correct=2))
        self.assertIn("PARITY", text)
        self.assertIn("SAME ROWS: false", text)
        self.assertIn("PROTOCOL DIFFERENCES", text)

    def test_a_partial_row_set_is_called_out(self):
        text = P.render(doc(50, correct=2))
        self.assertIn("NOT the parity row-set", text)
        self.assertNotIn("NOT the parity row-set",
                         P.render(doc(P.OFFICIAL_TEST_ROWS, correct=38)))


class TestItEntersEveryReportByItself(unittest.TestCase):
    def test_a_checkpoint_with_no_parity_read_never_borrows_one(self):
        attached = P.attach("artifacts/checkpoints/decision/nope/stage-9")
        self.assertFalse(attached["measured"])
        self.assertIn("eval.parity gate", attached["how"])
        self.assertEqual(attached["baseline"]["accuracy"], 0.012338)
        self.assertEqual(attached["teacher"]["accuracy"], 0.924)

    def test_the_published_headline_is_attached_when_it_is_that_checkpoint(
            self):
        published = doc(3080, correct=38)
        published.pop("_records")
        with mock.patch.object(P, "published", return_value=published):
            attached = P.attach(published["checkpoint"])
        self.assertTrue(attached["measured"])
        self.assertIn("K=77", attached["line"])
        self.assertIs(attached["same_rows"], False)
        self.assertTrue(attached["protocol_differences"])

    def test_the_fullspace_hook_refuses_a_row_set_that_is_not_the_parity_one(
            self):
        rows = samples(40)
        short = F.emit_parity("ckpt", {}, {"banking77": (rows,
                                                        entries(rows), 1.0,
                                                        "cpu")})
        self.assertFalse(short["emitted"])
        self.assertIn("not the 3080 official test rows".replace("3080",
                                                                "3080"),
                      short["why"].replace("3 080", "3080"))
        self.assertIn("eval.parity gate", short["how"])

    def test_the_fullspace_hook_refuses_another_dataset(self):
        other = F.emit_parity("ckpt", {}, {"huffpost": ([1], [1], 1.0, "cpu")})
        self.assertFalse(other["emitted"])
        self.assertIn("banking77", other["why"])


if __name__ == "__main__":
    unittest.main()
