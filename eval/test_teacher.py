"""Suite for the teacher client and the distance probe (#T-teacher-probe).

Nothing here talks to the real API. The transport is exercised against a
local `http.server` that speaks the documented System-One shape, so the
retry, the budget, the cache and the parse are all covered without spending
a cent of the operator's balance.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import teacher as TE  # noqa: E402
from eval import teacher_probe as TP  # noqa: E402

KEY = "apikey_testonly_deadbeef"
OPTIONS = [{"id": "a", "text": "Alpha"}, {"id": "b", "text": "Beta"},
           {"id": "c", "text": "Gamma"}]


class _Handler(BaseHTTPRequestHandler):
    """Scripted teacher. `server.script` decides each response."""

    def log_message(self, *a):  # silence the test run
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(
            int(self.headers.get("Content-Length", 0))) or b"{}")
        self.server.seen.append((dict(self.headers), body))
        status, payload = self.server.script(body, len(self.server.seen))
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def ok(choice="a", **extra):
    def script(body, n):
        qid = next(iter(body["questions"]))
        ans = {"choice": choice, "confidence": 0.8}
        ans.update(extra)
        return 200, {"answers": {qid: ans}, "usage": {"input_tokens": 12}}
    return script


class ServerCase(unittest.TestCase):
    """A live local endpoint plus a scratch cache, per test."""

    def setUp(self):
        self.srv = HTTPServer(("127.0.0.1", 0), _Handler)
        self.srv.seen = []
        self.srv.script = ok()
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.tmp = Path(os.environ.get("TMPDIR", "/tmp")) / (
            f"teacher-test-{os.getpid()}-{id(self)}")
        self._env = dict(os.environ)
        os.environ[TE.ENDPOINT_ENV] = (
            f"http://127.0.0.1:{self.srv.server_address[1]}/v1/decisions")
        os.environ[TE.KEY_ENV] = KEY
        os.environ[TE.MODEL_ENV] = "jev-test"

    def tearDown(self):
        self.srv.shutdown()
        os.environ.clear()
        os.environ.update(self._env)
        if self.tmp.exists():
            for p in sorted(self.tmp.rglob("*"), reverse=True):
                p.rmdir() if p.is_dir() else p.unlink()
            self.tmp.rmdir()

    def client(self, **kw):
        budget = TE.Budget(**{"max_calls": 50, "max_usd": 1.0,
                              **kw.pop("budget", {})})
        return TE.TeacherClient(budget, cache_dir=self.tmp, log=lambda *a: None,
                                **kw)


class TestConfig(unittest.TestCase):
    def test_key_never_leaks_into_the_card(self):
        os.environ[TE.KEY_ENV] = KEY
        try:
            card = TE.config_card()
            self.assertNotIn(KEY, json.dumps(card))
            self.assertEqual(card["key_fingerprint"], TE.fingerprint())
            self.assertEqual(len(card["key_fingerprint"]), 8)
        finally:
            os.environ.pop(TE.KEY_ENV, None)

    def test_redact_scrubs_the_key_from_error_text(self):
        os.environ[TE.KEY_ENV] = KEY
        try:
            msg = TE.redact(f"HTTP 401 for Bearer {KEY} at /v1")
            self.assertNotIn(KEY, msg)
            self.assertIn("<key:", msg)
        finally:
            os.environ.pop(TE.KEY_ENV, None)

    def test_key_file_is_read_when_env_is_unset(self):
        path = Path(os.environ.get("TMPDIR", "/tmp")) / "jev-key-test"
        path.write_text(f"  {KEY}\n")
        os.environ.pop(TE.KEY_ENV, None)
        os.environ[TE.KEY_FILE_ENV] = str(path)
        try:
            self.assertEqual(TE.api_key(), KEY)
        finally:
            os.environ.pop(TE.KEY_FILE_ENV, None)
            path.unlink()

    def test_unconfigured_client_refuses_instead_of_stubbing(self):
        for var in (TE.ENDPOINT_ENV, TE.KEY_ENV, TE.KEY_FILE_ENV):
            os.environ.pop(var, None)
        os.environ[TE.KEY_FILE_ENV] = "/nonexistent/jev-key"
        try:
            c = TE.TeacherClient(cache_dir=Path("/nonexistent/cache"))
            with self.assertRaises(TE.NotConfigured):
                c.decide("s", "q", OPTIONS)
        finally:
            os.environ.pop(TE.KEY_FILE_ENV, None)


class TestParse(unittest.TestCase):
    def test_choice_and_distribution(self):
        doc = {"answers": {"q": {"choice": "b", "confidence": 0.7,
                                 "probabilities": {"a": 0.2, "b": 0.7,
                                                   "c": 0.1}}}}
        ans = TE.parse_answer(doc, "q", OPTIONS)
        self.assertEqual(ans["choice"], "b")
        self.assertAlmostEqual(ans["probs"]["b"], 0.7)
        self.assertFalse(ans["off_menu"])

    def test_distribution_only_still_yields_a_choice(self):
        doc = {"answers": {"q": {"distribution": {"a": 0.1, "c": 0.9}}}}
        self.assertEqual(TE.parse_answer(doc, "q", OPTIONS)["choice"], "c")

    def test_absent_distribution_is_none_not_faked(self):
        ans = TE.parse_answer({"answers": {"q": {"choice": "a"}}}, "q",
                              OPTIONS)
        self.assertIsNone(ans["probs"])

    def test_off_menu_answer_is_flagged_never_snapped(self):
        ans = TE.parse_answer({"answers": {"q": {"choice": "zzz"}}}, "q",
                              OPTIONS)
        self.assertTrue(ans["off_menu"])
        self.assertEqual(ans["choice"], "zzz")

    def test_unparsable_answer_is_reported(self):
        ans = TE.parse_answer({"nothing": 1}, "q", OPTIONS)
        self.assertTrue(ans["unparsed"])
        self.assertIsNone(ans["choice"])

    def test_usage_aliases(self):
        self.assertEqual(
            TE.usage_of({"usage": {"prompt_tokens": 9}})["input_tokens"], 9)


class TestBudget(unittest.TestCase):
    def test_call_cap_stops_before_the_call(self):
        b = TE.Budget(max_calls=2, max_usd=99)
        b.charge(10)
        b.charge(10)
        with self.assertRaises(TE.BudgetExceeded):
            b.check()

    def test_dollar_cap_stops_the_run(self):
        b = TE.Budget(max_calls=10_000, max_usd=0.000001)
        b.charge(1_000_000)
        with self.assertRaises(TE.BudgetExceeded):
            b.check()

    def test_card_reports_cost_per_call(self):
        b = TE.Budget()
        b.charge(1_000_000)
        self.assertGreater(b.card()["usd"], 0)
        self.assertEqual(b.card()["calls"], 1)


class TestTransport(ServerCase):
    def test_payload_is_the_documented_shape(self):
        self.client().decide("state text", "which?", OPTIONS, "dept")
        headers, body = self.srv.seen[0]
        self.assertEqual(headers["Authorization"], f"Bearer {KEY}")
        self.assertEqual(body["model"], "jev-test")
        self.assertEqual(body["state"], "state text")
        q = body["questions"]["dept"]
        self.assertEqual(q["type"], "choice")
        self.assertEqual(q["instructions"], "which?")
        self.assertEqual(q["criteria"], {"a": "Alpha", "b": "Beta",
                                         "c": "Gamma"})

    def test_second_identical_call_is_served_from_disk(self):
        c = self.client()
        first = c.decide("s", "q", OPTIONS)
        second = c.decide("s", "q", OPTIONS)
        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(len(self.srv.seen), 1)
        self.assertEqual(c.budget.calls, 1)
        self.assertEqual(c.budget.cached, 1)

    def test_cache_key_separates_different_option_sets(self):
        c = self.client()
        c.decide("s", "q", OPTIONS)
        c.decide("s", "q", OPTIONS[:2])
        self.assertEqual(len(self.srv.seen), 2)

    def test_retry_then_success(self):
        def script(body, n):
            if n == 1:
                return 503, {"error": "busy"}
            return ok()(body, n)
        self.srv.script = script
        TE.BACKOFF_BASE, base = 1.0001, TE.BACKOFF_BASE
        try:
            out = self.client().decide("s", "q", OPTIONS)
        finally:
            TE.BACKOFF_BASE = base
        self.assertEqual(out["choice"], "a")
        self.assertEqual(len(self.srv.seen), 2)

    def test_auth_failure_is_not_retried_and_is_redacted(self):
        self.srv.script = lambda body, n: (401, {"detail": f"bad {KEY}"})
        with self.assertRaises(TE.TeacherError) as ctx:
            self.client().decide("s", "q", OPTIONS)
        self.assertNotIn(KEY, str(ctx.exception))
        self.assertEqual(len(self.srv.seen), 1)

    def test_missing_usage_falls_back_to_an_estimate_and_says_so(self):
        self.srv.script = lambda body, n: (
            200, {"answers": {"q": {"choice": "a"}}})
        c = self.client()
        out = c.decide("s", "q", OPTIONS)
        self.assertTrue(out["tokens_estimated"])
        self.assertGreater(c.budget.input_tokens, 0)

    def test_budget_stops_the_client_mid_run(self):
        c = self.client(budget={"max_calls": 1})
        c.decide("s1", "q", OPTIONS)
        with self.assertRaises(TE.BudgetExceeded):
            c.decide("s2", "q", OPTIONS)


class _S:
    """The two fields of `data.optset.Sample` the probe actually reads."""

    def __init__(self, dataset, row, gold, k=4):
        self.dataset = dataset
        self.row_id = f"{dataset}-{row}"
        self.question_id = "q0"
        self.state = f"state {row}"
        self.question = "which?"
        self.options = [{"id": f"o{i}", "text": f"Option {i}"}
                        for i in range(k)]
        self.answer = gold

    def option_ids(self):
        return [o["id"] for o in self.options]


class TestProbe(unittest.TestCase):
    def test_stratify_is_proportional_seeded_and_repeatable(self):
        pool = {"big": [_S("big", i, "o0") for i in range(800)],
                "small": [_S("small", i, "o0") for i in range(200)]}
        a = TP.stratify(pool, 100)
        b = TP.stratify(pool, 100)
        self.assertEqual([s.row_id for _, s in a], [s.row_id for _, s in b])
        per = {}
        for cut, _ in a:
            per[cut] = per.get(cut, 0) + 1
        self.assertGreater(per["big"], per["small"])
        self.assertGreaterEqual(per["small"], 25)  # floor, not proportion

    def test_stratify_never_asks_for_more_rows_than_a_cut_has(self):
        pool = {"tiny": [_S("tiny", i, "o0") for i in range(8)]}
        self.assertEqual(len(TP.stratify(pool, 400)), 8)

    def test_subsample_card_is_enough_to_rebuild_the_cut(self):
        picked = TP.stratify({"d": [_S("d", i, "o1") for i in range(30)]}, 30)
        card = TP.subsample_card(picked, 30)
        self.assertEqual(card["seed"], TP.SUBSAMPLE_SEED)
        self.assertEqual(card["n"], len(picked))
        self.assertEqual(card["rows"][0]["gold"], "o1")
        self.assertEqual(len(card["rows"][0]["options"]), 4)

    def test_accuracy_reports_wilson_ci_and_mean_chance(self):
        rows = [{"pred": "a", "gold": "a", "k": 4}] * 3 + [
            {"pred": "b", "gold": "a", "k": 4}]
        rep = TP.accuracy(rows)
        self.assertEqual(rep["accuracy"], 0.75)
        self.assertEqual(rep["chance"], 0.25)
        self.assertLess(rep["accuracy_ci95"][0], 0.75)
        self.assertTrue(rep["beats_chance"])

    def test_distance_is_teacher_minus_our_forced_choice(self):
        ours = {"rows": {
            "r1": {"cut": "d", "pred": "unknown", "pred_options_only": "a",
                   "gold": "a", "k": 4},
            "r2": {"cut": "d", "pred": "b", "pred_options_only": "b",
                   "gold": "a", "k": 4}}}
        theirs = {"rows": {
            "r1": {"cut": "d", "pred": "a", "gold": "a", "k": 4},
            "r2": {"cut": "d", "pred": "a", "gold": "a", "k": 4}}}
        t = TP.distance_table(ours, theirs)
        self.assertEqual(t["ALL"]["teacher"]["accuracy"], 1.0)
        self.assertEqual(t["ALL"]["ours_forced_choice"]["accuracy"], 0.5)
        # the abstention counts as wrong in the headline number
        self.assertEqual(t["ALL"]["ours_with_unknown"]["accuracy"], 0.0)
        self.assertEqual(t["ALL"]["distance"], 0.5)

    def test_distance_uses_only_rows_both_sides_answered(self):
        ours = {"rows": {f"r{i}": {"cut": "d", "pred": "a",
                                   "pred_options_only": "a", "gold": "a",
                                   "k": 4} for i in range(10)}}
        theirs = {"rows": {"r0": {"cut": "d", "pred": "b", "gold": "a",
                                  "k": 4}}}
        t = TP.distance_table(ours, theirs)
        self.assertEqual(t["ALL"]["n"], 1)

    def test_worst_cut_is_named(self):
        def side(cut, rid, pred, gold):
            return {rid: {"cut": cut, "pred": pred,
                          "pred_options_only": pred, "gold": gold, "k": 4}}
        ours = {"rows": {**side("easy", "e", "a", "a"),
                         **side("hard", "h", "b", "a")}}
        theirs = {"rows": {**side("easy", "e", "a", "a"),
                           **side("hard", "h", "a", "a")}}
        t = TP.distance_table(ours, theirs)
        self.assertEqual(t["ALL"]["worst_cut"]["cut"], "hard")

    def test_verdict_states_the_league(self):
        far = {"ALL": {"n": 400, "distance": 0.45,
                       "teacher": {"accuracy": 0.74},
                       "ours_forced_choice": {"accuracy": 0.29,
                                              "beats_chance": True}}}
        self.assertIn("different league", TP.verdict(far))
        close = {"ALL": {"n": 400, "distance": 0.02,
                         "teacher": {"accuracy": 0.74},
                         "ours_forced_choice": {"accuracy": 0.72,
                                                "beats_chance": True}}}
        self.assertIn("within noise", TP.verdict(close))
        self.assertIn("did not measure", TP.verdict({}))


class TestGateShape(unittest.TestCase):
    def test_gate_publishes_cost_model_version_and_no_key(self):
        os.environ[TE.KEY_ENV] = KEY
        try:
            ours = {"model_version": "v9", "run_id": "r1", "rows": {
                "r0": {"cut": "d", "pred": "a", "pred_options_only": "a",
                       "gold": "a", "k": 4}}}
            theirs = {"rows": {"r0": {"cut": "d", "pred": "a", "gold": "a",
                                      "k": 4}},
                      "errors": 0, "budget": TE.Budget().card()}
            table = TP.distance_table(ours, theirs)
            gate = TP.compose_gate("ckpt/x", ours, theirs, {"seed": 1,
                                                            "rows": []},
                                   table)
            blob = json.dumps(gate)
            self.assertNotIn(KEY, blob)
            self.assertEqual(gate["model_version"], "v9")
            self.assertIn("usd", gate["cost"])
            self.assertIn("distance", gate["table"]["ALL"])
        finally:
            os.environ.pop(TE.KEY_ENV, None)


if __name__ == "__main__":
    unittest.main()
