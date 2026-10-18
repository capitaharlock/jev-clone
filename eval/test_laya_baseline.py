"""Tests de `eval.laya_baseline` — #T-laya-baseline.

Todo corre en segundos y **nunca descarga un peso ni importa `laya`**: la
columna recibe su predictor por inyección, así que un scorer determinista
sustituye a los dos checkpoints. Lo que la task pide:

* el formato forzado es UNO: cada candidato como criterio `choice` con su
  id opaco de etiqueta y su texto de descripción;
* la columna es K ancha (sin `unknown`), el runner FALLA si a una fila le
  falta un peso o si la respuesta no trae un id ofrecido;
* la columna Laya mide las MISMAS 400 filas que las columnas del
  preflight, comprobado por `same_rows()` y por el sha que firmó su gate;
* el control de permutación distingue un predictor invariante de uno que
  lee la posición; se mide sobre todas las filas;
* cada informe sale de `eval.metrics_suite.report()` y pasa `require()`;
* la regla del veredicto separa gana / empata / pierde / falla igual;
* `eval.fullspace` acepta un scorer externo por `.entries()` sin cambiar
  la aritmética de `tally()`, y no reescribe el artefacto de paridad;
* el gate no escribe ninguna cifra que no se haya medido.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from data.optset import Sample
from eval import fullspace as F
from eval import laya_baseline as L
from eval import metrics_suite as MS
from eval import preflight_refs as P
from eval.test_preflight_refs import FIXTURE


def _h(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 10_000) / 1_000.0


def _softmax(z: list) -> list:
    m = max(z)
    e = [pow(2.718281828, v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def text_only(state, question, candidates, lang):
    """Lee sólo el texto del candidato: invariante al orden, ciego al estado."""
    z = [_h(c["text"]) for c in candidates]
    return {"probs": _softmax(z), "model": L.ROUTE[lang], "ms": 1.0}


def state_reading(state, question, candidates, lang):
    """Acierta: el candidato cuyo texto comparte más palabras con el estado."""
    words = set(state.lower().replace(".", "").split())
    z = [len(words & set(c["text"].lower().split())) * 3.0
         for c in candidates]
    return {"probs": _softmax(z), "model": L.ROUTE[lang], "ms": 1.0}


def position_reading(state, question, candidates, lang):
    """Prefiere la PRIMERA opción ofrecida: al revés, cambia de id."""
    z = [4.0 if i == 0 else 0.0 for i in range(len(candidates))]
    return {"probs": _softmax(z), "model": L.ROUTE[lang], "ms": 1.0}


class FormatTest(unittest.TestCase):
    def test_one_choice_with_the_id_as_label_and_the_text_as_description(self):
        q = L.question_of("¿Dónde?", [{"id": "c1", "text": "verde"},
                                      {"id": "c2", "text": "azul"}])
        self.assertEqual(list(q), ["q"])
        self.assertEqual(q["q"]["type"], "choice")
        self.assertEqual(q["q"]["instructions"], "¿Dónde?")
        self.assertEqual(q["q"]["criteria"], {"c1": "verde", "c2": "azul"})

    def test_a_label_that_is_its_own_text_carries_no_description(self):
        q = L.question_of("intent?", [{"id": "card_arrival",
                                       "text": "card_arrival"}])
        self.assertEqual(q["q"]["criteria"], {"card_arrival": None})

    def test_the_weights_are_realigned_by_id_to_the_offered_order(self):
        got = L.align({"c2": 0.7, "c1": 0.3}, ["c1", "c2"])
        self.assertEqual(got, [0.3, 0.7])

    def test_a_missing_id_fails_the_runner(self):
        with self.assertRaises(L.ProtocolMismatch):
            L.align({"c1": 1.0}, ["c1", "c2"])

    def test_the_protocol_routes_by_declared_lang(self):
        proto = L.protocol()
        self.assertEqual(proto["routing"]["es"], "laya-multilingual")
        self.assertEqual(proto["routing"]["en"], "laya")
        self.assertEqual(proto["device"], "cpu")


class ColumnTest(unittest.TestCase):
    def test_the_column_is_k_wide_with_no_unknown(self):
        rows, meta = L.laya_column(FIXTURE, text_only)
        self.assertEqual(meta["n"], len(FIXTURE))
        for row in rows:
            self.assertEqual(len(row["probs"]), row["k"])
            self.assertAlmostEqual(sum(row["probs"]), 1.0, places=6)
            self.assertLess(row["pred"], row["k"])
        self.assertEqual(meta["rows_by_model"],
                         {"laya-multilingual": 2, "laya": 2})

    def test_a_predictor_short_of_a_weight_fails(self):
        def short(state, question, candidates, lang):
            return {"probs": [1.0], "model": "laya", "ms": 1.0}
        with self.assertRaises(L.ProtocolMismatch):
            L.laya_column(FIXTURE, short)

    def test_the_column_resumes_from_its_row_log(self):
        calls = []

        def counting(*args):
            calls.append(args)
            return text_only(*args)

        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "picks.jsonl"
            rows1, meta1 = L.laya_column(FIXTURE, counting, log=log)
            rows2, meta2 = L.laya_column(FIXTURE, counting, log=log)
        self.assertEqual(len(calls), len(FIXTURE))
        self.assertEqual(meta1["resumed_rows"], 0)
        self.assertEqual(meta2["resumed_rows"], len(FIXTURE))
        self.assertEqual(rows1, rows2)

    def test_the_real_400_rows_are_the_preflight_rows(self):
        episodes = P.load_cut()
        rows, _ = L.laya_column(episodes, text_only)
        self.assertEqual(len(rows), 400)
        self.assertEqual(P.rows_sha(episodes), P.rows_sha(episodes))
        nli, _ = P.nli_column(episodes, score=lambda pairs: [
            _h(hypothesis) for _premise, hypothesis in pairs])
        got = P.same_rows({L.COLUMN: rows, P.REF_NLI: nli})
        self.assertTrue(got["pass"])
        self.assertEqual(got["n"], 400)
        # the sha the preflight gate signed, if it is on disk
        signed = L._preflight_gate_sha()
        if signed:
            self.assertEqual(got["rows_sha256"], signed)


class PermutationTest(unittest.TestCase):
    def test_an_order_invariant_predictor_passes(self):
        rows, _ = L.laya_column(FIXTURE, text_only)
        perm = L.permutation_control(FIXTURE, rows, text_only)
        self.assertTrue(perm["pass"])
        self.assertEqual(perm["flipped"], 0)
        self.assertEqual(perm["n_perms"], len(FIXTURE))
        self.assertLessEqual(perm["max_abs_prob_delta"], perm["tolerance"])

    def test_a_position_reading_predictor_flips(self):
        rows, _ = L.laya_column(FIXTURE, position_reading)
        perm = L.permutation_control(FIXTURE, rows, position_reading)
        self.assertFalse(perm["pass"])
        self.assertEqual(perm["flipped"], len(FIXTURE))
        self.assertEqual(perm["flip_rate"], 1.0)
        self.assertGreater(perm["max_abs_prob_delta"], perm["tolerance"])

    def test_the_control_covers_every_row_and_says_so(self):
        rows, _ = L.laya_column(FIXTURE, text_only)
        perm = L.permutation_control(FIXTURE, rows, text_only)
        self.assertEqual(perm["n_rows"], len(FIXTURE))
        self.assertIn("all 4 rows", perm["sample"])
        self.assertIn("rounding", perm["caveat"])


class SuiteTest(unittest.TestCase):
    def test_the_report_is_the_suite_and_passes_require(self):
        episodes = P.load_cut()
        rows, _ = L.laya_column(episodes, text_only)
        perm = L.permutation_control(episodes, rows, text_only)
        doc = L.report_for(rows, episodes=episodes,
                           model_version="fake@0", permutation=perm)
        self.assertEqual(doc["format"], MS.FORMAT)
        self.assertEqual(MS.check(doc), [])
        self.assertEqual(doc["counterfactual"]["n"], 140)
        self.assertEqual(doc["ranking"]["n"], 400)
        self.assertEqual(len(doc["by_family"]), 5)

    def test_by_lang_reports_split_the_cut_and_pass_require(self):
        episodes = P.load_cut()
        rows, _ = L.laya_column(episodes, text_only)
        perm = L.permutation_control(episodes, rows, text_only)
        langs = L.by_lang_reports(episodes, rows, model_version="fake@0",
                                 permutation=perm)
        self.assertEqual(sorted(langs), ["en", "es"])
        self.assertEqual(sum(d["ranking"]["n"] for d in langs.values()),
                         400)
        for doc in langs.values():
            self.assertEqual(MS.check(doc), [])

    def test_a_text_only_predictor_never_wins_a_group_whole(self):
        """El fallo documentado del pointer, reproducido por el fake."""
        episodes = P.load_cut()
        rows, _ = L.laya_column(episodes, text_only)
        tracking = P.tracking_control(episodes, rows, source="test")
        self.assertEqual(tracking["followed"], 0)


class VerdictTest(unittest.TestCase):
    def fig(self, lo, hi, chance=0.3375):
        return {"accuracy": (lo + hi) / 2, "accuracy_ci95": [lo, hi],
                "chance": chance}

    def test_wins_ties_loses_and_fails_alike(self):
        self.assertEqual(L.outcome(self.fig(0.7, 0.8), self.fig(0.5, 0.6)),
                         "wins")
        self.assertEqual(L.outcome(self.fig(0.5, 0.6), self.fig(0.7, 0.8)),
                         "loses")
        self.assertEqual(L.outcome(self.fig(0.5, 0.7), self.fig(0.6, 0.8)),
                         "ties")
        self.assertEqual(L.outcome(self.fig(0.2, 0.4), self.fig(0.25, 0.35)),
                         "fails_alike")
        self.assertEqual(L.outcome({}, self.fig(0.2, 0.4)), "unmeasured")

    def test_paired_counts_are_counted_not_estimated(self):
        a, _ = L.laya_column(FIXTURE, state_reading)
        b, _ = L.laya_column(FIXTURE, text_only)
        got = L.paired_counts(a, b)
        total = sum(got[k] for k in ("both_right", "laya_only",
                                     "ours_only", "neither"))
        self.assertEqual(total, len(FIXTURE))

    def test_the_mix_declaration_cites_its_files(self):
        for name, node in L.MIX_DECLARATION.items():
            self.assertIn("in_laya_training_mix", node, name)
            self.assertTrue(node["source"], name)
        self.assertFalse(L.MIX_DECLARATION["banking77"]["in_laya_training_mix"])
        self.assertTrue(L.MIX_DECLARATION["ag_news"]["in_laya_training_mix"])
        self.assertTrue(L.MIX_DECLARATION["boolq"]["in_laya_training_mix"])
        self.assertTrue(L.MIX_DECLARATION["phishing"]["in_laya_training_mix"])
        for node in L.PUBLISHED.values():
            self.assertTrue(node["citation"])
            self.assertIn("source", node)


class FakeEngine:
    """Un scorer externo: `.entries()` y `.device`, nada más."""
    device = "cpu"

    def __init__(self, right: bool):
        self.right = right
        self.calls = 0

    def entries(self, samples, split, cut, dataset, batch_size=16):
        self.calls += 1
        out = []
        for s in samples:
            k = len(s.options)
            probs = [0.0] * (k + 1)
            probs[s.gold_index if self.right else (s.gold_index + 1) % k] = 1.0
            pred = max(range(k), key=lambda j: probs[j])
            out.append({"id": s.row_id, "split": split, "cut": cut,
                        "dataset": dataset, "cardinality": k,
                        "logits": probs, "probs": probs,
                        "label": s.gold_index, "pred": pred})
        return out


def _samples(n: int, k: int) -> list:
    options = [{"id": f"l{i}", "text": f"l{i}"} for i in range(k)]
    return [Sample(dataset="banking77", row_id=f"r{i}", question_id="q",
                   state="state", question="question", options=list(options),
                   answer="l0", gold_index=0) for i in range(n)]


class FullspaceSeamTest(unittest.TestCase):
    def test_an_external_scorer_goes_through_the_same_tally(self):
        rows = _samples(50, 77)
        with mock.patch("eval.calib.entries_from_samples") as never:
            entries, rep = F.scored(FakeEngine(right=True), rows)
        never.assert_not_called()
        self.assertEqual(rep["n"], 50)
        self.assertEqual(rep["hits"], 50)
        self.assertEqual(rep["cardinality"], 77)
        self.assertAlmostEqual(rep["chance"], 1 / 77, places=6)
        self.assertTrue(rep["beats_chance"])

    def test_a_wrong_external_scorer_gets_no_credit(self):
        rep = F.score(FakeEngine(right=False), _samples(50, 77))
        self.assertEqual(rep["hits"], 0)
        self.assertFalse(rep["beats_chance"])

    def test_the_pointer_path_is_untouched_without_entries(self):
        with mock.patch("eval.calib.entries_from_samples",
                        return_value=FakeEngine(True).entries(
                            _samples(3, 5), "test", "full-space",
                            "banking77")) as pointer:
            rep = F.score(object(), _samples(3, 5))
        pointer.assert_called_once()
        self.assertEqual(rep["hits"], 3)

    def test_run_with_an_engine_writes_its_own_gate_and_no_parity(self):
        rows = _samples(20, 77)
        with tempfile.TemporaryDirectory() as tmp:
            gate_path = Path(tmp) / "fullspace.json"
            with mock.patch.object(F, "full_samples", return_value=rows), \
                    mock.patch.object(F.U.T, "load_checkpoint") as never, \
                    mock.patch("eval.cuts.record_query") as ledger:
                doc = F.run("laya:fake@0", ("banking77",), 20, "cpu",
                            write=True, sweep=False, reason="unit test",
                            engine=FakeEngine(True),
                            manifest={"model_version": "laya:fake@0"},
                            gate_path=gate_path, parity=False, log=lambda *a: None)
            never.assert_not_called()
            ledger.assert_called_once()
            self.assertEqual(ledger.call_args.args[0], "laya:fake@0")
            self.assertTrue(gate_path.exists())
            written = json.loads(gate_path.read_text())
        self.assertEqual(doc["checkpoint"], "laya:fake@0")
        self.assertFalse(doc["parity"]["emitted"])
        self.assertIn("external scorer", doc["parity"]["why"])
        self.assertEqual(written["table"]["banking77"]["ours"]["hits"], 20)
        self.assertEqual(doc["model_version"], "laya:fake@0")

    def test_the_laya_engine_shapes_entries_like_the_pointer(self):
        class FakePredictor:
            device = "cpu"

            def predict_with(self, name, state, question, candidates):
                k = len(candidates)
                return {"probs": [1.0 / k] * k, "model": name, "ms": 1.0}

        engine = L.LayaEngine(FakePredictor(), "laya", log=lambda *a: None)
        got = engine.entries(_samples(2, 4), "test", "full-space",
                             "banking77")
        self.assertEqual(len(got), 2)
        for e in got:
            self.assertEqual(len(e["probs"]), 5)
            self.assertEqual(e["probs"][-1], 0.0)
            self.assertLess(e["pred"], 4)
            self.assertEqual(e["label"], 0)


class GateTest(unittest.TestCase):
    def test_nothing_unmeasured_is_written_as_a_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(L, "GATE_DIR", Path(tmp)), \
                    mock.patch.object(L, "GATE_PATH", Path(tmp) / "g.json"), \
                    mock.patch.object(L, "COLUMN_DIR", Path(tmp) / "c"), \
                    mock.patch.object(L, "REPORT_PATH", Path(tmp) / "r.json"), \
                    mock.patch.object(L, "FULLSPACE_PATH",
                                      Path(tmp) / "f.json"):
                doc = L.gate(write=True)
                self.assertTrue((Path(tmp) / "g.json").exists())
        self.assertEqual(doc["verdict"], "AWAITING-COMPUTE")
        self.assertIsNone(doc["pass"])
        self.assertIsNone(doc["verdict_line"])
        for name in ("counterfactual_pairs", "battery_dev_forced",
                     "banking77_k77", "same_rows_as_preflight"):
            self.assertIsNone(doc["checks"][name]["pass"], name)
            self.assertNotIn("accuracy", doc["checks"][name], name)
        self.assertTrue(doc["checks"]["mix_declared_per_dataset"]["pass"])

    def test_a_measured_column_signs_the_battery_and_the_pairs(self):
        episodes = P.load_cut()
        rows, trace = L.laya_column(episodes, state_reading)
        perm = L.permutation_control(episodes, rows, state_reading)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(L, "GATE_DIR", Path(tmp)), \
                    mock.patch.object(L, "GATE_PATH", Path(tmp) / "g.json"), \
                    mock.patch.object(L, "COLUMN_DIR", Path(tmp) / "c"), \
                    mock.patch.object(L, "REPORT_PATH", Path(tmp) / "r.json"), \
                    mock.patch.object(L, "FULLSPACE_PATH",
                                      Path(tmp) / "f.json"):
                column = L.save_measured_column(
                    episodes, rows, trace, model_version="fake@0",
                    permutation=perm, checkpoints={}, package={"v": 0},
                    command="unit")
                self.assertIsNotNone(L.load_measured_column(episodes))
                built = L.build_reports(episodes, column, write=True)
                doc = L.gate(write=True, episodes=episodes)
                names = sorted(os.listdir(tmp))
        self.assertIn("refs-laya-lang-es.json", names)
        self.assertIn("refs-laya-lang-en.json", names)
        pairs = doc["checks"]["counterfactual_pairs"]
        self.assertTrue(pairs["pass"])
        self.assertEqual(pairs["n"], 140)
        self.assertEqual(pairs["accuracy"],
                         built["whole"]["counterfactual"]["accuracy"])
        forced = doc["checks"]["battery_dev_forced"]
        self.assertEqual(forced["accuracy"],
                         built["whole"]["ranking"]["accuracy"])
        self.assertEqual(sorted(forced["by_lang"]), ["en", "es"])
        self.assertEqual(len(forced["by_family"]), 5)
        self.assertTrue(doc["checks"]["same_rows_as_preflight"]["pass"])
        self.assertIsNone(doc["checks"]["banking77_k77"]["pass"])
        self.assertEqual(doc["verdict"], "AWAITING-COMPUTE")
        self.assertIn("joint counterfactual", doc["verdict_line"])

    def test_a_column_measured_on_other_rows_is_not_reused(self):
        rows, trace = L.laya_column(FIXTURE, text_only)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(L, "COLUMN_DIR", Path(tmp)):
                L.save_measured_column(
                    FIXTURE, rows, trace, model_version="fake@0",
                    permutation={}, checkpoints={}, package={}, command="")
                self.assertIsNotNone(L.load_measured_column(FIXTURE))
                self.assertIsNone(L.load_measured_column(P.load_cut()))


class NoLayaImportTest(unittest.TestCase):
    def test_the_module_imports_no_laya_and_no_torch(self):
        import subprocess
        import sys
        code = ("import sys; import eval.laya_baseline; "
                "print(sorted(m for m in ('laya', 'torch', 'transformers') "
                "if m in sys.modules))")
        got = subprocess.run([sys.executable, "-c", code], cwd=str(L.ROOT),
                             env={**os.environ, "PYTHONPATH": str(L.ROOT)},
                             capture_output=True, text=True, check=True)
        self.assertEqual(got.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
