import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import torch

from bench import rules
from bench.run import cell_keys, read_rows, stratified, system_one
from model.backbone import DecisionModel, ModelConfig
from model.checkpoint import save
from serve.server import Decider, check_request, examples, handler

ROOT = Path(__file__).resolve().parents[1]
REQUEST = {"state": "Comparing soups.", "question": "Which has the least salt?",
           "candidates": [{"id": "a", "text": "salt 1.2 g"},
                          {"id": "b", "text": "salt 0.3 g"}]}


class RequestTest(unittest.TestCase):

    def test_a_good_request_passes(self):
        self.assertIsNone(check_request(REQUEST))

    def test_bad_requests_name_the_problem(self):
        self.assertIn("state", check_request({**REQUEST, "state": ""}))
        self.assertIn("2 to", check_request({**REQUEST, "candidates": []}))
        twins = [{"id": "a", "text": "x"}, {"id": "a", "text": "y"}]
        self.assertIn("unique", check_request({**REQUEST, "candidates": twins}))

    def test_every_example_is_a_valid_request_with_its_answer(self):
        samples = examples()
        self.assertGreaterEqual(len(samples), 6)
        for sample in samples:
            self.assertIsNone(check_request(sample["request"]), sample["name"])
            self.assertEqual(rules.decide({**sample["request"]}),
                             sample["expected"], sample["name"])


class ServerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        torch.manual_seed(0)
        save(DecisionModel(ModelConfig(max_length=512, d_model=32, n_heads=4,
                                       n_layers=1, d_ff=64, pointer_layers=1)),
             Path(cls.folder.name))
        cls.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), handler(Decider(Path(cls.folder.name), "cpu")))
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.folder.cleanup()

    def call(self, path: str, body: dict | None = None) -> tuple[int, dict]:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            self.url + path, data=data,
            headers={"Content-Type": "application/json"})
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
        self.assertAlmostEqual(sum(answer["probabilities"].values()), 1.0,
                               places=4)
        self.assertGreaterEqual(answer["latency_ms"], 0)

    def test_invalid_input_is_a_422(self):
        status, answer = self.call("/v1/decide", {"state": "x"})
        self.assertEqual(status, 422)
        self.assertIn("question", answer["error"])

    def test_health_and_examples(self):
        self.assertEqual(self.call("/v1/health")[1]["format"],
                         "asimovia-decision-v1")
        self.assertTrue(self.call("/examples")[1])


class BenchTest(unittest.TestCase):

    def test_stratified_subset_keeps_every_cell(self):
        rows = read_rows(ROOT / "benchmark" / "product_specs.dev.jsonl")
        subset = stratified(rows, 600)
        self.assertLess(abs(len(subset) - 600), 40)
        cells = lambda items: {(r["difficulty"], len(r["candidates"]), r["lang"])
                               for r in items}
        self.assertEqual(cells(subset), cells(rows))

    def test_system_one_request_keeps_every_option(self):
        row = {"state": "s", "question": "q", "candidates": REQUEST["candidates"]}
        criteria = system_one(row)["questions"]["pick"]["criteria"]
        self.assertEqual(criteria, {"a": "salt 1.2 g", "b": "salt 0.3 g"})

    def test_rules_read_sodium_in_mg(self):
        row = {"question": "Which product gives the most salt?",
               "candidates": [{"id": "a", "text": "X. energy 1 kcal, salt 1 g."},
                              {"id": "b", "text": "Y. energy 1 kcal, sodium 600 mg."}]}
        self.assertEqual(rules.decide(row), "b")
        self.assertIn("units:sodium_mg", cell_keys(
            {**row, "difficulty": "extreme", "lang": "en"}))


if __name__ == "__main__":
    unittest.main()
