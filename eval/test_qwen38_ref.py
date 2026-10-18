"""Tests de `eval.qwen38_ref` — #T-qwen38-ref.

Todo lo de aquí corre en milisegundos y **nunca levanta un modelo**: el
chooser de la columna es inyectable, así que un profesor determinista de
seis líneas sustituye a Qwen sobre las MISMAS 400 filas de desarrollo. Lo
que los tests fijan:

* la columna nueva se firma con el modelo ACTUAL y a nombre de esta task,
  y el runner se NIEGA a re-medir el modelo que ya está medido;
* dos columnas de cortes distintos no son una comparación (falla por
  `rows_sha256` antes de publicar nada);
* la regla del gate es «no peor»: elecciones idénticas dan GO, y un
  profesor medible-peor da NO-GO y manda al productor de vuelta al 3.6;
* el control de permutación se mide a la temperatura del generador (la
  comparable con el 3.6) y se repite en greedy, y el de greedy dice que
  lo es;
* la temperatura del profesor viaja de verdad hasta el cuerpo de la
  petición: es lo que hace medible el control a 0.
"""
from __future__ import annotations

import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from data import episode_gen as EG
from eval import metrics_suite as MS
from eval import preflight_refs as P
from eval import qwen38_ref as Q

#: El modelo que YA está medido en `#T-preflight-refs`.
MEASURED_ALREADY = "qwen3.6:27b-mlx"
BASELINE_MV = f"qwen-local:{MEASURED_ALREADY}"


def gold_map(episodes: list) -> dict:
    """(estado, pregunta) → id correcta. Único por fila del corte."""
    return {(ep["state"], ep["question"]): ep["answer"] for ep in episodes}


def perfect_teacher(episodes: list):
    """Un profesor que acierta siempre. Determinista y sin pesos."""
    gold = gold_map(episodes)

    def choose(state, question, candidates):
        return {"choice": gold[(state, question)], "evidence": state[:12]}

    return choose


def lazy_teacher(episodes: list):
    """Un profesor que siempre elige la primera opción OFRECIDA.

    Sobre un corte de K mixto eso es el nivel del azar con prior de
    posición: medible-peor que el perfecto, y sensible al orden.
    """
    def choose(state, question, candidates):
        return {"choice": candidates[0]["id"], "evidence": state[:12]}

    return choose


def column_of(episodes: list, choose, model: str,
              rows_sha: str | None = None) -> dict:
    """Una columna medida con un chooser inyectado, en la forma del disco."""
    rows, trace = P.qwen_column(episodes, choose=choose)
    picks = {t["row_id"]: t.get("choice") for t in trace["traces"]}
    perm = P.qwen_permutation(episodes, picks, choose=choose,
                              temperature=EG.TEMPERATURE)
    return {"column": "test", "task": Q.TASK, "cut": P.CUT_NAME,
            "rows_sha256": rows_sha or P.rows_sha(episodes),
            "measured_utc": P.utcnow(),
            "model_version": f"qwen-local:{model}",
            "permutation": perm, "trace": trace, "rows": rows}


class MeasureTest(unittest.TestCase):
    """La columna nueva: firmada por quien mide, sobre las filas de siempre."""

    @classmethod
    def setUpClass(cls):
        cls.episodes = P.load_cut()

    def measure_in_tmp(self, model: str = "qwen3.8:27b-mlx"):
        """`measure` + `publish` con las rutas dentro de un temporal."""
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        patches = {
            "GATE_DIR": tmp, "COLUMN_DIR": tmp / "columns",
            "REPORT_PATH": tmp / "refs-qwen-local-3.8.json",
            "GATE_PATH": tmp / "gate.json",
        }
        for name, value in patches.items():
            self.enterContext(unittest.mock.patch.object(Q, name, value))
        self.enterContext(unittest.mock.patch.object(
            Q, "baseline_report", lambda: {"format": MS.FORMAT,
                                           "model_version": BASELINE_MV}))
        self.enterContext(unittest.mock.patch.object(
            Q, "qwen_model_id", lambda: model))
        column = Q.measure(self.episodes,
                           choose=perfect_teacher(self.episodes),
                           perm_choose=perfect_teacher(self.episodes))
        return column, tmp

    def test_the_column_is_signed_with_the_current_model_and_this_task(self):
        column, tmp = self.measure_in_tmp()
        self.assertEqual(column["model_version"], "qwen-local:qwen3.8:27b-mlx")
        self.assertEqual(column["task"], Q.TASK)
        self.assertEqual(column["rows_sha256"], P.rows_sha(self.episodes))
        self.assertEqual(len(column["rows"]), 400)
        self.assertTrue((tmp / "columns" / f"{Q.COLUMN_KEY}.json").exists())
        again = Q.load_column(self.episodes)
        self.assertEqual(again["measured_utc"], column["measured_utc"])

    def test_the_report_is_the_suite_and_nothing_computed_apart(self):
        column, tmp = self.measure_in_tmp()
        doc = Q.publish(self.episodes, column)
        self.assertEqual(MS.check(doc), [])
        self.assertEqual(doc["task"], Q.TASK)
        self.assertEqual(doc["column"], Q.COLUMN_KEY)
        self.assertEqual(doc["cut"]["n"], 400)
        self.assertEqual(doc["ranking"]["accuracy"], 1.0)
        self.assertEqual(json.loads(
            (tmp / "refs-qwen-local-3.8.json").read_text())["task"], Q.TASK)

    def test_it_refuses_to_re_measure_the_model_already_measured(self):
        with self.assertRaises(P.ProtocolMismatch) as got:
            self.measure_in_tmp(model=MEASURED_ALREADY)
        self.assertIn(MEASURED_ALREADY, str(got.exception))

    def test_the_greedy_control_is_kept_apart_and_says_it_is_greedy(self):
        column, _ = self.measure_in_tmp()
        self.assertEqual(column["permutation"]["temperature"],
                         EG.TEMPERATURE)
        zero = column["permutation_temperature_0"]
        self.assertEqual(zero["temperature"], Q.PERM_TEMPERATURE)
        self.assertIn("greedily", zero["caveat"])
        self.assertIn("sampling noise", column["permutation"]["caveat"])
        doc = Q.publish(self.episodes, column, write=False)
        self.assertEqual(doc["permutation_invariance"]["temperature"],
                         EG.TEMPERATURE)


class CompareTest(unittest.TestCase):
    """La regla: «no peor», medida pareada sobre las mismas filas."""

    @classmethod
    def setUpClass(cls):
        cls.episodes = P.load_cut()
        cls.perfect = column_of(cls.episodes, perfect_teacher(cls.episodes),
                                "qwen3.8:27b-mlx")
        cls.lazy = column_of(cls.episodes, lazy_teacher(cls.episodes),
                             "qwen3.8:27b-mlx")
        cls.baseline = column_of(cls.episodes,
                                 perfect_teacher(cls.episodes),
                                 MEASURED_ALREADY)

    def test_two_columns_of_different_rows_are_not_a_comparison(self):
        other = dict(self.perfect, rows_sha256="0" * 64)
        with self.assertRaises(P.ProtocolMismatch):
            Q.compare(self.baseline, other, self.episodes, reps=50)

    def test_identical_picks_are_not_worse_and_give_go(self):
        got = Q.compare(self.baseline, self.perfect, self.episodes, reps=200)
        self.assertTrue(got["identical_picks"])
        self.assertEqual(got["verdict"], "GO")
        self.assertTrue(got["pass"])
        self.assertFalse(got["better"])
        self.assertEqual(got["forced"]["ci95"], [0.0, 0.0])

    def test_a_measurably_worse_teacher_is_a_no_go(self):
        got = Q.compare(self.baseline, self.lazy, self.episodes, reps=500)
        self.assertEqual(got["verdict"], "NO-GO")
        self.assertFalse(got["pass"])
        self.assertLess(got["forced"]["ci95"][1], 0)
        self.assertLess(got["forced"]["candidate"], got["forced"]["baseline"])
        self.assertTrue(got["families_below_baseline"])

    def test_a_no_go_sends_the_producer_back_to_the_measured_teacher(self):
        worse = Q.compare(self.baseline, self.lazy, self.episodes, reps=500)
        decision = Q.producer_decision(worse)
        self.assertEqual(decision["qwen_model"], Q.FALLBACK_MODEL)
        self.assertTrue(decision["operator_decides"])
        fine = Q.compare(self.baseline, self.perfect, self.episodes, reps=200)
        ok = Q.producer_decision(fine)
        self.assertEqual(ok["qwen_model"], "qwen3.8:27b-mlx")
        self.assertFalse(ok["operator_decides"])

    def test_the_difference_is_paired_over_the_same_units(self):
        got = Q.compare(self.baseline, self.lazy, self.episodes, reps=300)
        self.assertEqual(got["forced"]["bootstrap"]["unit"], "paired")
        self.assertEqual(got["forced"]["n"], 400)
        self.assertEqual(got["n"], 400)
        self.assertEqual(sorted(got["by_lang"]), ["en", "es"])
        self.assertEqual(len(got["by_family"]), 5)


class GateTest(unittest.TestCase):
    """El gate: las dos columnas, el veredicto y la decisión, en un fichero."""

    def gate_in_tmp(self, candidate_choose) -> dict:
        episodes = P.load_cut()
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        column = column_of(episodes, candidate_choose, "qwen3.8:27b-mlx")
        column["column"] = Q.COLUMN_KEY
        (tmp / "columns").mkdir()
        (tmp / "columns" / f"{Q.COLUMN_KEY}.json").write_text(
            json.dumps(column))
        base_col = column_of(episodes, perfect_teacher(episodes),
                             MEASURED_ALREADY)
        base_rep = P.report_for(P.REF_QWEN, base_col["rows"],
                                episodes=episodes,
                                model_version=BASELINE_MV,
                                permutation=base_col["permutation"])
        for name, value in {"GATE_DIR": tmp, "COLUMN_DIR": tmp / "columns",
                            "REPORT_PATH": tmp / "refs.json",
                            "GATE_PATH": tmp / "gate.json"}.items():
            self.enterContext(unittest.mock.patch.object(Q, name, value))
        self.enterContext(unittest.mock.patch.object(
            Q, "baseline_column", lambda: base_col))
        self.enterContext(unittest.mock.patch.object(
            Q, "baseline_report", lambda: base_rep))
        self.enterContext(unittest.mock.patch.object(
            Q, "qwen_model_id", lambda: "qwen3.8:27b-mlx"))
        return Q.gate(reps=200), tmp

    def test_the_gate_publishes_both_columns_and_the_decision(self):
        episodes = P.load_cut()
        doc, tmp = self.gate_in_tmp(perfect_teacher(episodes))
        self.assertEqual(doc["verdict"], "GO")
        self.assertEqual(doc["cut"]["rows_sha256"], P.rows_sha(episodes))
        self.assertFalse(doc["cut"]["sealed_cut_read"])
        self.assertEqual(sorted(doc["columns"]), ["qwen3.6", "qwen3.8"])
        self.assertEqual(doc["columns"]["qwen3.6"]["model_version"],
                         BASELINE_MV)
        self.assertEqual(doc["producer"]["qwen_model"], "qwen3.8:27b-mlx")
        self.assertIn("go_if", doc["rule"])
        self.assertEqual(json.loads(
            (tmp / "gate.json").read_text())["verdict"], "GO")

    def test_a_worse_candidate_writes_a_no_go_gate(self):
        episodes = P.load_cut()
        doc, _ = self.gate_in_tmp(lazy_teacher(episodes))
        self.assertEqual(doc["verdict"], "NO-GO")
        self.assertFalse(doc["pass"])
        self.assertEqual(doc["producer"]["qwen_model"], Q.FALLBACK_MODEL)

    def test_the_gate_refuses_to_run_without_the_measured_column(self):
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(unittest.mock.patch.object(
            Q, "COLUMN_DIR", tmp / "columns"))
        with self.assertRaises(Q.Missing):
            Q.gate(write=False)


class TeacherTemperatureTest(unittest.TestCase):
    """La temperatura llega al cuerpo de la petición, o el control a 0 miente."""

    def test_the_chooser_passes_its_temperature_to_the_teacher(self):
        seen = {}

        def capture(messages, num_predict, timeout, temperature=None):
            seen["temperature"] = temperature
            return json.dumps({"choice": "c1", "evidence": "x"})

        with unittest.mock.patch.object(EG, "_ollama_chat", capture):
            Q.chooser(0.0)("x y z", "q?", [{"id": "c1", "text": "a"},
                                           {"id": "c2", "text": "b"}])
        self.assertEqual(seen["temperature"], 0.0)

    def test_the_body_of_the_request_carries_the_temperature(self):
        sent = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return json.dumps({"message": {"content": "ok"}}).encode()

        def fake_urlopen(req, timeout=None):
            sent["body"] = json.loads(req.data)
            return FakeResponse()

        with unittest.mock.patch("urllib.request.urlopen", fake_urlopen):
            EG._ollama_chat([{"role": "user", "content": "hi"}],
                            num_predict=8, timeout=5, temperature=0.0)
            self.assertEqual(sent["body"]["options"]["temperature"], 0.0)
            EG._ollama_chat([{"role": "user", "content": "hi"}],
                            num_predict=8, timeout=5)
            self.assertEqual(sent["body"]["options"]["temperature"],
                             EG.TEMPERATURE)


if __name__ == "__main__":
    unittest.main()
