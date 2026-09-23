"""Tests for #T-pointer-head (stdlib unittest).

Two layers, same rule as `test_torch_stack.py`:

* structure — runs without weights: the head must contain no parameter
  living in a label space, and must refuse malformed input;
* behaviour — needs torch + the sha256-verified backbone, and is skipped
  (never faked) when they are absent.

The four contract tests are the ones the gate runs, called through the
same functions, so `gate.json` cannot say something the suite doesn't.

Run:
    .venv-train/bin/python -m unittest model.test_decision_head -v
"""
from __future__ import annotations

import math
import os
import unittest

from . import weights

try:  # the real stack is optional at test time, never stubbed
    import torch

    from .decision_head import (UNKNOWN_ID, DecisionEngine,
                                PointerDecisionHead, check_order_invariance,
                                check_unseen_label, check_variable_k,
                                full_distribution, toy_unknown_fit)
    HAVE_TORCH = True
except ImportError:  # pragma: no cover - env without the training venv
    HAVE_TORCH = False

HAVE_WEIGHTS = os.path.exists(weights.manifest_path("ettin-68m"))
NEEDS_STACK = unittest.skipUnless(
    HAVE_TORCH and HAVE_WEIGHTS,
    "needs .venv-train (torch) and verified ettin-68m weights")

STATE = ("shift log, berth 4: the harbor gate is open, the night ledger "
         "is balanced, and the signal lamp reads green.")
QUESTION = "what is the state of the harbor gate?"
POOL = [{"id": f"o{i}", "text": t} for i, t in enumerate([
    "the harbor gate is open",
    "the harbor gate is closed",
    "the harbor gate is jammed",
    "the night ledger is short",
    "the signal lamp reads red",
    "the orbit path is drifting",
    "the meadow path is quiet",
    "the engine runs loud",
    "the harbor gate was never inspected",
])]
# A string that appears in no training corpus of this repo.
UNSEEN = "the bathyscaphe manifest was filed under seal in Reykjavik"

_ENGINE = None


def engine():
    """One CPU engine for the whole module — loading is the slow part."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = DecisionEngine(device="cpu")
    return _ENGINE


class TestStructure(unittest.TestCase):
    """No weights needed: the label-space ban is structural."""

    @unittest.skipUnless(HAVE_TORCH, "needs torch")
    def test_no_parameter_lives_in_a_label_space(self):
        head = PointerDecisionHead(d_state=512, d_model=256, n_layers=2)
        report = head.label_free_report()
        self.assertTrue(report["label_free"], report["offenders"])
        self.assertEqual(report["offenders"], {})
        # Belt and braces: nothing named like the old sklearn label space.
        names = " ".join(n for n, _ in head.named_parameters())
        for banned in ("classes_", "num_labels", "label_emb", "vocab"):
            self.assertNotIn(banned, names)

    @unittest.skipUnless(HAVE_TORCH, "needs torch")
    def test_parameter_count_is_independent_of_k(self):
        head = PointerDecisionHead(d_state=512, d_model=256, n_layers=2)
        before = head.n_params()
        memory = torch.randn(1, 12, 512)
        mask = torch.ones(1, 12, dtype=torch.bool)
        q = torch.randn(512)
        for k in (1, 2, 7, 31):
            out = head(memory, mask, q, torch.randn(k, 512))
            self.assertEqual(out.shape, (k + 1,))
        self.assertEqual(head.n_params(), before)

    @unittest.skipUnless(HAVE_TORCH, "needs torch")
    def test_layer_budget_and_empty_options_are_refused(self):
        with self.assertRaises(ValueError):
            PointerDecisionHead(d_state=512, n_layers=4)
        head = PointerDecisionHead(d_state=512, d_model=64, n_layers=1)
        with self.assertRaises(ValueError):
            head(torch.randn(1, 4, 512), torch.ones(1, 4, dtype=torch.bool),
                 torch.randn(512), torch.zeros(0, 512))


@NEEDS_STACK
class TestOrderInvariance(unittest.TestCase):
    """Contract test 1: 20 permutations, same argmax, KL < 1e-4."""

    def test_twenty_permutations(self):
        report = check_order_invariance(engine(), STATE, QUESTION,
                                        POOL[:4], n_perms=20)
        self.assertEqual(report["n_perms"], 20)
        self.assertTrue(report["argmax_stable"],
                        f"argmax moved: {report}")
        self.assertLess(report["max_kl"], 1e-4,
                        f"KL {report['max_kl']} over tolerance")

    def test_permutation_moves_probabilities_with_the_options(self):
        eng = engine()
        mem = eng.encode_state(STATE)
        base = eng.score(mem, QUESTION, POOL[:4])
        flipped = eng.score(mem, QUESTION, list(reversed(POOL[:4])))
        for oid, p in zip(base["option_ids"], base["probs"]):
            q = flipped["probs"][flipped["option_ids"].index(oid)]
            self.assertAlmostEqual(p, q, places=5)


@NEEDS_STACK
class TestVariableK(unittest.TestCase):
    """Contract test 2: K=2, 4, 9 on the same row, no fixed maximum."""

    def test_k_2_4_9(self):
        report = check_variable_k(engine(), STATE, QUESTION, POOL,
                                  ks=(2, 4, 9))
        self.assertTrue(report["ok"], report["per_k"])
        for k in (2, 4, 9):
            detail = report["per_k"][k]
            self.assertEqual(detail["k"], k)
            self.assertEqual(detail["n_logits"], k + 1)
            self.assertTrue(detail["sums_to_one"])
            self.assertTrue(detail["finite"])

    def test_state_is_encoded_once_per_row(self):
        eng = engine()
        eng._states.clear()
        row = {"state": STATE,
               "questions": [{"id": "q1", "text": QUESTION,
                              "options": POOL[:3]},
                             {"id": "q2", "text": "and the lamp?",
                              "options": POOL[:5]}]}
        results = eng.predict_row(row)
        self.assertEqual(len(results), 2)
        self.assertEqual([r["k"] for r in results], [3, 5])
        self.assertEqual(len(eng._states), 1)  # one forward for two Qs


@NEEDS_STACK
class TestUnseenLabel(unittest.TestCase):
    """Contract test 3: an option text never seen is finite and orderable."""

    def test_unseen_option_is_scored_like_any_other(self):
        report = check_unseen_label(engine(), STATE, QUESTION, POOL[:4],
                                    UNSEEN)
        self.assertTrue(report["ok"], report)
        self.assertTrue(math.isfinite(report["unseen_logit"]))
        self.assertTrue(report["orderable"])
        self.assertEqual(report["k"], 5)

    def test_an_entirely_unseen_option_set_still_produces_a_distribution(self):
        eng = engine()
        options = [{"id": "x1", "text": UNSEEN},
                   {"id": "x2", "text": "the funicular timetable is "
                                        "posted in Ladin"},
                   {"id": "x3", "text": "zarzuela rehearsal moved to "
                                        "the annex"}]
        res = eng.score(STATE, "which memo is this?", options)
        dist = full_distribution(res)
        self.assertEqual(len(dist), 4)
        self.assertAlmostEqual(sum(dist), 1.0, places=5)
        self.assertTrue(all(math.isfinite(x) for x in dist))
        self.assertEqual(sorted(res["ranking"]), ["x1", "x2", "x3"])


@NEEDS_STACK
class TestUnknownLogit(unittest.TestCase):
    """Contract test 4: `unknown` is a LEARNED logit, not a threshold.

    Scope, stated rather than papered over: this fits the head alone on
    eight toy rows. It proves the mechanism — `unknown` sits inside the
    softmax, gradient reaches its parameters, and it can win a row whose
    state does not contain the answer. It does NOT measure abstention
    quality on real data; that is `#T-train-real`.
    """

    def test_unknown_is_inside_the_softmax(self):
        res = engine().score(STATE, QUESTION, POOL[:4])
        dist = full_distribution(res)
        self.assertEqual(len(dist), 5)
        self.assertAlmostEqual(sum(dist), 1.0, places=5)
        self.assertGreater(res["unknown_prob"], 0.0)
        self.assertTrue(math.isfinite(res["unknown_logit"]))

    def test_unknown_is_learned_and_can_win_a_row(self):
        eng = DecisionEngine(backbone=engine().backbone)  # fresh head
        report = toy_unknown_fit(eng, steps=150)
        self.assertGreater(report["unknown_grad_step0"], 0.0,
                           "no gradient reached the unknown head")
        self.assertLess(report["last_loss"], report["first_loss"])
        self.assertTrue(report["unknown_wins_unanswerable"],
                        report["verdicts"])
        self.assertTrue(report["answerable_not_abstained"],
                        report["verdicts"])
        for verdict in report["verdicts"]:
            if verdict["gold"] == UNKNOWN_ID:
                self.assertEqual(verdict["argmax"], UNKNOWN_ID)


@NEEDS_STACK
class TestGate(unittest.TestCase):
    def test_gate_json_matches_the_suite(self):
        from .decision_head import GATE_DIR
        path = os.path.join(GATE_DIR, "gate.json")
        if not os.path.exists(path):
            self.skipTest("run `python -m model.decision_head gate` first")
        import json
        with open(path) as fh:
            gate = json.load(fh)
        self.assertEqual(gate["task"], "T-pointer-head")
        self.assertTrue(gate["pass"])
        self.assertGreater(gate["head_params"], 0)
        self.assertGreater(gate["latency_k4"]["p50_ms"], 0.0)
        self.assertEqual(gate["latency_k4"]["k"], 4)
        for name in ("order_invariance", "variable_k", "unseen_label",
                     "unknown_logit", "label_free"):
            self.assertTrue(gate["checks"][name]["pass"], name)


@NEEDS_STACK
class TestBatchedForward(unittest.TestCase):
    """`forward_batch` must be `forward`, row by row (#T-metal-throughput)."""

    def _head(self):
        import torch
        from .decision_head import PointerDecisionHead
        torch.manual_seed(11)
        return PointerDecisionHead(d_state=64, d_model=32, n_layers=2,
                                   n_heads=4).eval()

    def test_matches_the_row_path_with_ragged_option_sets(self):
        import torch
        head = self._head()
        ks = [1, 3, 7, 4]
        t = 9
        mem = torch.randn(len(ks), t, 64)
        mmask = torch.ones(len(ks), t, dtype=torch.bool)
        mmask[1, 5:] = False           # a short state, padded
        q = torch.randn(len(ks), 64)
        kmax = max(ks)
        opts = torch.zeros(len(ks), kmax, 64)
        omask = torch.zeros(len(ks), kmax, dtype=torch.bool)
        per_row = []
        for i, k in enumerate(ks):
            o = torch.randn(k, 64)
            opts[i, :k] = o
            omask[i, :k] = True
            per_row.append(head(mem[i:i + 1], mmask[i:i + 1], q[i], o))
        got = head.forward_batch(mem, mmask, q, opts, omask)
        self.assertEqual(list(got.shape), [len(ks), kmax + 1])
        for i, k in enumerate(ks):
            want = per_row[i]
            mine = torch.cat([got[i, :k], got[i, kmax:kmax + 1]])
            self.assertTrue(torch.allclose(mine, want, atol=1e-5),
                            f"row {i}: {mine.tolist()} != {want.tolist()}")
            if k < kmax:   # the padding columns carry no probability mass
                self.assertTrue(bool(torch.isinf(got[i, k:kmax]).all()))
        probs = torch.softmax(got, dim=-1)
        self.assertTrue(torch.allclose(probs.sum(-1), torch.ones(len(ks)),
                                       atol=1e-5))
        # what lands in a padding slot must not matter: `pack_options`
        # gathers rather than zero-fills, so the masks are the only thing
        # keeping the pad out of a real row's answer.
        noisy = opts.clone()
        for i, k in enumerate(ks):
            noisy[i, k:] = torch.randn(kmax - k, 64) * 50
        again = head.forward_batch(mem, mmask, q, noisy, omask)
        for i, k in enumerate(ks):
            self.assertTrue(torch.allclose(again[i, :k], got[i, :k],
                                           atol=1e-5))
            self.assertTrue(torch.allclose(again[i, kmax], got[i, kmax],
                                           atol=1e-5))

    def test_rejects_a_row_with_no_options(self):
        import torch
        head = self._head()
        mem = torch.randn(2, 4, 64)
        mmask = torch.ones(2, 4, dtype=torch.bool)
        omask = torch.tensor([[True, False], [False, False]])
        with self.assertRaises(ValueError):
            head.forward_batch(mem, mmask, torch.randn(2, 64),
                               torch.zeros(2, 2, 64), omask)


if __name__ == "__main__":
    unittest.main()
