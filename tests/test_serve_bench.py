import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import torch

from asimovia.bench import report, rules
from asimovia.bench.run import cell_keys, choice_of, stratified, system_one
from asimovia.data.io import read_jsonl
from asimovia.model.backbone import DecisionModel, ModelConfig
from asimovia.model.checkpoint import save
from asimovia.serve.server import Decider, check_request, examples, handler

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
REQUEST = {
    "state": "Comparing soups.",
    "question": "Which has the least salt?",
    "candidates": [{"id": "a", "text": "salt 1.2 g"}, {"id": "b", "text": "salt 0.3 g"}],
}


class RequestTest(unittest.TestCase):
    def test_a_good_request_passes(self):
        self.assertIsNone(check_request(REQUEST))

    def test_bad_requests_name_the_problem(self):
        self.assertIn("state", check_request({**REQUEST, "state": ""}))
        self.assertIn("2 to", check_request({**REQUEST, "candidates": []}))
        twins = [{"id": "a", "text": "x"}, {"id": "a", "text": "y"}]
        self.assertIn("unique", check_request({**REQUEST, "candidates": twins}))

    def test_every_example_is_a_valid_request_with_its_answer(self):
        samples = examples(EXAMPLES)
        self.assertGreaterEqual(len(samples), 6)
        for sample in samples:
            self.assertIsNone(check_request(sample["request"]), sample["name"])
            self.assertEqual(
                rules.decide({**sample["request"]}), sample["expected"], sample["name"]
            )


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        torch.manual_seed(0)
        save(
            DecisionModel(
                ModelConfig(
                    max_length=512, d_model=32, n_heads=4, n_layers=1, d_ff=64, pointer_layers=1
                )
            ),
            Path(cls.folder.name),
        )
        quiet = handler(Decider(Path(cls.folder.name), "cpu"), EXAMPLES)
        quiet.log_message = lambda *args: None
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), quiet)
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.folder.cleanup()

    def call(
        self, path: str, body: dict | None = None, headers: dict | None = None
    ) -> tuple[int, dict]:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            self.url + path,
            data=data,
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            with error:
                return error.code, json.loads(error.read())

    def test_decide_returns_a_distribution_over_the_ids(self):
        status, answer = self.call("/v1/decide", REQUEST)
        self.assertEqual(status, 200)
        self.assertIn(answer["choice"], ("a", "b"))
        self.assertAlmostEqual(sum(answer["probabilities"].values()), 1.0, places=4)
        self.assertGreaterEqual(answer["latency_ms"], 0)

    def test_invalid_input_is_a_422(self):
        status, answer = self.call("/v1/decide", {"state": "x"})
        self.assertEqual(status, 422)
        self.assertIn("question", answer["error"])

    def test_health_and_examples(self):
        self.assertEqual(self.call("/v1/health")[1]["format"], "asimovia-decision-v1")
        self.assertTrue(self.call("/examples")[1])

    def test_a_query_string_does_not_change_the_route(self):
        self.assertEqual(self.call("/v1/health?verbose=1")[0], 200)
        self.assertEqual(self.call("/v1/decide?trace=1", REQUEST)[0], 200)

    def test_a_malformed_content_length_is_rejected(self):
        status, answer = self.call("/v1/decide", REQUEST, headers={"Content-Length": "many"})
        self.assertEqual(status, 413)
        self.assertIn("bytes", answer["error"])

    def test_unknown_routes_are_404(self):
        self.assertEqual(self.call("/nope")[0], 404)
        self.assertEqual(self.call("/v1/nope", REQUEST)[0], 404)


class BenchTest(unittest.TestCase):
    def test_stratified_subset_keeps_every_cell(self):
        rows = read_jsonl(ROOT / "benchmark" / "product_specs.dev.jsonl")
        subset = stratified(rows, 600)
        self.assertLess(abs(len(subset) - 600), 40)

        def cells(items: list[dict]) -> set[tuple]:
            return {(r["difficulty"], len(r["candidates"]), r["lang"]) for r in items}

        self.assertEqual(cells(subset), cells(rows))

    def test_system_one_request_keeps_every_option(self):
        row = {"state": "s", "question": "q", "candidates": REQUEST["candidates"]}
        criteria = system_one(row)["questions"]["pick"]["criteria"]
        self.assertEqual(criteria, {"a": "salt 1.2 g", "b": "salt 0.3 g"})

    def test_rules_read_sodium_in_mg(self):
        row = {
            "question": "Which product gives the most salt?",
            "candidates": [
                {"id": "a", "text": "X. energy 1 kcal, salt 1 g."},
                {"id": "b", "text": "Y. energy 1 kcal, sodium 600 mg."},
            ],
        }
        self.assertEqual(rules.decide(row), "b")
        self.assertIn("units:sodium_mg", cell_keys({**row, "difficulty": "extreme", "lang": "en"}))

    def test_choice_is_read_from_flat_and_nested_replies(self):
        flat = {"answers": {"pick": {"choice": "a"}}}
        nested = {"states": [{"answers": {"pick": {"probabilities": {"a": 0.2, "b": 0.8}}}}]}
        self.assertEqual(choice_of(flat), "a")
        self.assertEqual(choice_of(nested), "b")
        self.assertIsNone(choice_of({}))

    def test_the_rules_reach_the_published_ceiling(self):
        rows = read_jsonl(ROOT / "benchmark" / "product_specs.dev.jsonl")
        hits = sum(rules.decide(row) == row["answer"] for row in rows)
        self.assertEqual(hits, 2975)

    def test_the_committed_report_matches_the_raw_results(self):
        raw = [
            json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((ROOT / "results" / "raw").glob("*.json"))
        ]
        self.assertEqual(
            report.render(raw), (ROOT / "results" / "README.md").read_text(encoding="utf-8")
        )


if __name__ == "__main__":
    unittest.main()
