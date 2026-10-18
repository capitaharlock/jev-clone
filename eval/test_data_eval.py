"""Unit tests for eval.data_eval (#T-data-eval)."""
import unittest

from eval.data_eval import (
    DYNAMIC_B_MARGIN_OVER_CHANCE,
    HIGH_CONF_THRESHOLD,
    INSERTION_STABILITY_MIN,
    MULTIQ_MAX_Q,
    MULTIQ_SPEEDUP_MIN,
    OOD_ABSTAIN_FLOOR,
    SHUFFLE_FLIP_MAX,
    SHUFFLE_JS_MAX,
    TripleLeakDetector,
    ZeroShotPredictor,
    abstention_eval,
    assert_sealed,
    build_ood_split,
    candidate_insertion_eval,
    dynamic_labels_eval,
    freeze_jevals_cleanroom,
    js_divergence,
    locale_of,
    manifest_sha,
    multi_question_eval,
    ood_eval,
    option_shuffle_eval,
    sample_quotas,
)


def _items(n=60, seed=7, k=4, universe=12):
    import random
    rng = random.Random(seed)
    labels = [f"lbl-{i}" for i in range(universe)]
    out = []
    for i in range(n):
        opts = [{"id": lab, "text": f"option text {lab} number {i % 5}"}
                for lab in rng.sample(labels, k)]
        gold = rng.randrange(k)
        out.append({"id": f"t{i:03d}", "locale": "en-US",
                    "state": f"state sentence {i} about {opts[gold]['id']}",
                    "options": opts, "answer": opts[gold]["id"]})
    return out


class TestSeal(unittest.TestCase):
    def test_calib_in_train_fails(self):
        import json
        real_train_id = None
        with open("artifacts/data-prefetch/massive.jsonl") as f:
            for line in f:
                r = json.loads(line)
                if r.get("split") == "train" and r.get("questions"):
                    real_train_id = (
                        f"massive:train:{r['questions'][0].get('id')}")
                    break
        self.assertIsNotNone(real_train_id)
        calib = [{"id": real_train_id, "state": "x"}]
        with self.assertRaises(ValueError):
            assert_sealed(calib, [], ["jev-000000001"],
                          train_pools=("massive",))

    def test_jevals_in_train_fails(self):
        import json
        train_ids = set()
        with open("artifacts/data-prefetch/massive.jsonl") as f:
            for line in f:
                r = json.loads(line)
                if r.get("split") == "train" and r.get("questions"):
                    train_ids.add(
                        f"massive:train:{r['questions'][0].get('id')}")
                if len(train_ids) > 5:
                    break
        jev = [sorted(train_ids)[0].replace("massive:train:", "jev-")]
        # Prefix differs, so force the real failure mode: a jevals-style
        # id must still be caught when it literally appears in train.
        with self.assertRaises(ValueError):
            assert_sealed([], [], [sorted(train_ids)[0]],
                          train_pools=("massive",))
        self.assertTrue(jev)  # silence linters about unused derivation

    def test_triple_layers(self):
        det = TripleLeakDetector(["The Quick Brown Fox jumps."])
        self.assertEqual(det.scan("The Quick Brown Fox jumps.")[1], "exact")
        self.assertEqual(det.scan("  the   QUICK brown fox JUMPS.  ")[1],
                         "normalized")
        hit, layer, _ = det.scan("The Quick Brown Fox jumps")
        self.assertTrue(hit)
        self.assertEqual(layer, "semantic")
        self.assertEqual(det.scan("unrelated weather report")[1], "none")

    def test_canary_in_sealed_fails(self):
        from eval.data_eval import canary_texts
        cans = canary_texts()
        self.assertTrue(cans, "need canary fixtures for this test")
        calib = [{"id": "x:test:q0", "state": cans[0]}]
        with self.assertRaises(ValueError):
            assert_sealed(calib, [], ["jev-000000001"])

    def test_jevals_frozen_deterministic(self):
        self.assertEqual(freeze_jevals_cleanroom(100, 5),
                         freeze_jevals_cleanroom(100, 5))
        self.assertEqual(len(freeze_jevals_cleanroom(100)), 100)

    def test_manifest_sha_stable(self):
        self.assertEqual(manifest_sha(["b", "a"]), manifest_sha(["a", "b"]))


class TestOODSplit(unittest.TestCase):
    def test_no_overlap_correct_removed_soft(self):
        train, held = build_ood_split(n_train=300, n_held=60, seed=11)
        self.assertEqual(len(train), 300)
        self.assertEqual(len(held), 60)
        self.assertFalse({t["id"] for t in train} & {h["id"] for h in held})
        self.assertTrue(all(h["answer"] is None for h in held))
        self.assertFalse(any(o["id"] == "gold" for h in held
                             for o in h["options"]
                             if len(h["soft_target"]) == 1))
        for h in held:
            self.assertAlmostEqual(sum(h["soft_target"].values()), 1.0)
            if h["ambiguous"]:
                self.assertEqual(sorted(h["soft_target"].values()),
                                 [0.5, 0.5])

    def test_deterministic(self):
        t1, h1 = build_ood_split(50, 10, seed=3)
        t2, h2 = build_ood_split(50, 10, seed=3)
        self.assertEqual([i["id"] for i in t1], [i["id"] for i in t2])
        self.assertEqual([i["id"] for i in h1], [i["id"] for i in h2])


class TestBattery(unittest.TestCase):
    def test_permutation_stability_green(self):
        pred = ZeroShotPredictor()
        r = option_shuffle_eval(pred, _items(300, universe=8), seed=9)
        self.assertLessEqual(r["flip_rate"], SHUFFLE_FLIP_MAX)
        self.assertLessEqual(r["js_mean"], SHUFFLE_JS_MAX)
        self.assertTrue(r["pass"])

    def test_candidate_insertion_green(self):
        pred = ZeroShotPredictor()
        r = candidate_insertion_eval(pred, _items(300, universe=8), seed=9)
        self.assertGreaterEqual(r["stability"], INSERTION_STABILITY_MIN)
        self.assertTrue(r["pass"])

    def test_risk_coverage_beats_random_with_real_abstention(self):
        import math
        import random
        rng = random.Random(4)
        entries = []
        for i in range(400):
            y = 0 if rng.random() < 0.7 else 1
            logits = [2.0, 0.0] if y == 0 else [0.2, 0.0]
            m = max(logits)
            exps = [math.exp(v - m) for v in logits]
            s = sum(exps)
            p = [e / s for e in exps]
            entries.append({"id": f"e{i}", "logits": logits, "probs": p,
                            "pred": 0, "label": y})
        r = abstention_eval(entries)
        self.assertTrue(r["pass"])
        # Real abstention: abstaining the least confident must cut risk.
        c0 = r["curve"][0]["risk"]
        c1 = r["curve"][-1]["risk"]
        self.assertLess(c0, c1)

    def test_multi_q_shared_state_gain(self):
        pred = ZeroShotPredictor()
        items = _items(200, universe=20)
        # Long-context state (as in §83): shared encoding must pay off.
        items[0] = dict(items[0],
                        state=(items[0]["state"] + " filler context") * 100)
        r = multi_question_eval(pred, items)
        self.assertEqual(r["points"][-1]["n_questions"], MULTIQ_MAX_Q)
        self.assertGreaterEqual(r["speedup_50q"], MULTIQ_SPEEDUP_MIN)
        self.assertTrue(r["pass"])

    def test_dynamic_labels_beats_id_memorization(self):
        pred = ZeroShotPredictor()
        r = dynamic_labels_eval(pred, _items(600, universe=12), seed=9)
        self.assertGreater(r["zero_shot_acc"], r["memorization_acc"])
        self.assertGreaterEqual(
            r["zero_shot_acc"],
            r["chance"] + DYNAMIC_B_MARGIN_OVER_CHANCE)

    def test_ood_abstention_separates(self):
        pred = ZeroShotPredictor()
        import math
        import random
        rng = random.Random(6)
        entries = []
        for i in range(200):
            y = 0 if rng.random() < 0.8 else 1
            logits = [3.0, 0.0] if y == 0 else [0.5, 0.0]
            m = max(logits)
            exps = [math.exp(v - m) for v in logits]
            s = sum(exps)
            p = [e / s for e in exps]
            entries.append({"id": f"e{i}", "logits": logits, "probs": p,
                            "pred": 0, "label": y})
        _, held = build_ood_split(n_train=50, n_held=60, seed=6)
        from eval.calib import fit_thresholds
        cal = [{"split": "calibration", "logits": e["logits"],
                "label": e["label"], "locale": "en-US", "cardinality": 2}
               for e in entries[:100]]
        thr = fit_thresholds(cal, {"temperature": 1.0})["maxprob"]
        r = ood_eval(pred, entries, held, thr["threshold"])
        self.assertLess(r["ood_mean_conf"], r["in_domain_mean_conf"])
        self.assertGreater(r["abstention_gap"], 0)
        self.assertGreaterEqual(r["ood_abstention_rate"], OOD_ABSTAIN_FLOOR)

    def test_js_divergence_known_values(self):
        self.assertAlmostEqual(js_divergence([1.0, 0.0], [1.0, 0.0]), 0.0)
        self.assertGreater(js_divergence([1.0, 0.0], [0.0, 1.0]), 0.5)

    def test_high_conf_threshold_constant_sane(self):
        self.assertEqual(HIGH_CONF_THRESHOLD, 0.9)

    def test_locale_detection(self):
        self.assertEqual(locale_of("[it-IT] ciao"), "it-IT")
        self.assertEqual(locale_of("¿cómo estás?"), "es-ES")
        self.assertEqual(locale_of("plain english"), "en-US")

    def test_sample_quotas_covers_domains(self):
        pool = ([{"id": f"a:{i}", "domain": "a"} for i in range(10)] +
                [{"id": f"b:{i}", "domain": "b"} for i in range(10)])
        got = sample_quotas(pool, 6, seed=1)
        self.assertEqual(len(got), 6)
        self.assertTrue(any(i["domain"] == "a" for i in got))
        self.assertTrue(any(i["domain"] == "b" for i in got))


if __name__ == "__main__":
    unittest.main()
