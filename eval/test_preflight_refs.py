"""Tests de `eval.preflight_refs` — #T-preflight-refs.

Todo lo de aquí corre en segundos y **nunca carga un modelo**: las tres
columnas del runner reciben su predictor por inyección, así que un scorer
determinista de seis líneas sustituye a Qwen, al cabezal NLI y al pointer
head. Los cuatro que la task pide explícitamente:

* el runner FALLA si los n de las columnas no coinciden — y también si
  coinciden los n pero no las filas;
* el control pointer reproduce sobre la batería el fallo ya documentado en
  `.meshkore/docs/evidence/probe-mechanism-2026-09-24.json`, y el
  comprobador SEPARA ese caso del contrario (un predictor que sí lee el
  estado no lo reproduce);
* ninguna referencia consulta el corte sellado: cada ruta abierta durante
  una pasada completa se registra y se compara con las declaradas;
* cada columna se publica por `eval.metrics_suite.report()` y pasa
  `require()`; el gate escrito pasa C1-C7 de `eval.gate_rules`.

«No carga un modelo» se comprueba en un intérprete LIMPIO
(`NoModelTest`): en el mismo proceso otros tests de `eval/` sí importan
torch, así que afirmarlo sobre `sys.modules` del proceso compartido sería
una comprobación que depende del orden de los tests.
"""
from __future__ import annotations

import builtins
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from eval import gate_rules as GR
from eval import metrics_suite as MS
from eval import preflight_refs as P

ROOT = P.ROOT

#: Un grupo contrafactual hecho a mano: los dos miembros comparten
#: pregunta y textos de candidato y sólo cambia el hecho decisivo, que es
#: exactamente la forma de `#T-battery-dev`.
FIXTURE = [
    {"id": "dev-fix-es-2k-t01a", "variant_group": "dev-fix-es-2k-t01",
     "family": "extraction_paraphrase", "lang": "es", "answer": "c1",
     "state": "La llave está en el cajón verde. El azul guarda pilas.",
     "question": "¿En qué cajón está la llave?",
     "candidates": [{"id": "c1", "text": "En el cajón verde"},
                    {"id": "c2", "text": "En el cajón azul"}]},
    {"id": "dev-fix-es-2k-t01b", "variant_group": "dev-fix-es-2k-t01",
     "family": "extraction_paraphrase", "lang": "es", "answer": "c2",
     "state": "La llave está en el cajón azul. El verde guarda pilas.",
     "question": "¿En qué cajón está la llave?",
     "candidates": [{"id": "c1", "text": "En el cajón verde"},
                    {"id": "c2", "text": "En el cajón azul"}]},
    {"id": "dev-fix-en-2k-t02a", "variant_group": "dev-fix-en-2k-t02",
     "family": "attribute_comparison", "lang": "en", "answer": "c1",
     "state": "The train costs 42 euros and takes 3 hours. The bus costs "
              "25 euros and takes 5 hours.",
     "question": "Which option is faster?",
     "candidates": [{"id": "c1", "text": "The train, at 42 euros"},
                    {"id": "c2", "text": "The bus, at 25 euros"}]},
    {"id": "dev-fix-en-2k-t02b", "variant_group": "dev-fix-en-2k-t02",
     "family": "attribute_comparison", "lang": "en", "answer": "c2",
     "state": "The train costs 42 euros and takes 6 hours. The bus costs "
              "25 euros and takes 5 hours.",
     "question": "Which option is faster?",
     "candidates": [{"id": "c1", "text": "The train, at 42 euros"},
                    {"id": "c2", "text": "The bus, at 25 euros"}]},
]

#: Los dos checkpoints de la evidencia, por su `model_version`.
LEVERSTACK = "jev-dec-ettin-68m-d512l2h8-s20260922-n1000000-be9f0d454360"
FULLSPACE = "jev-dec-ettin-68m-d512l2h8-s20260922-n62528-62ee2f57d7e6"


# -- predictores deterministas: ni pesos, ni descargas, ni torch ----------

def _h(text: str) -> float:
    """Un escalar estable por texto. Sin parámetros y sin estado."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 10_000) / 1_000.0


def text_only_logits(decisions):
    """EL MECANISMO DOCUMENTADO: puntúa la OPCIÓN y no lee el estado.

    Es la reconstrucción mínima del fallo que la sonda del 2026-09-24
    midió: la elección forzada depende sólo del texto del candidato, así
    que no se mueve cuando el hecho decisivo del estado se mueve.
    """
    out = []
    for dec in decisions:
        z = [_h(c.text) for c in dec.candidates]
        out.append(z + [min(z) - 4.0])   # `unknown` bajo: no abstiene
    return out


def state_reading_logits(decisions):
    """El contraste: un predictor que SÍ lee el estado y acierta."""
    out = []
    for dec in decisions:
        gold = [c.id for c in dec.candidates].index(dec.gold)
        z = [6.0 if i == gold else 0.0 for i in range(dec.k)]
        out.append(z + [-4.0])
    return out


def text_only_scores(pairs):
    """La misma invariancia, en la interfaz del scorer NLI (por pares)."""
    return [_h(hypothesis) for _premise, hypothesis in pairs]


def text_only_choice(state, question, candidates):
    """La misma invariancia, en la interfaz de elección estructurada."""
    best = max(candidates, key=lambda c: _h(c["text"]))
    return {"choice": best["id"], "evidence": "", "backend": "stub"}


def unparseable_choice(state, question, candidates):
    """Una respuesta que no es un id válido: se registra abstención."""
    return {"reject": "unparseable: stub", "backend": "stub"}


def perm(model_version=LEVERSTACK):
    return P.permutation_from_evidence(model_version)


class SameRowsTest(unittest.TestCase):
    """Mismo corte, mismas filas — o el runner levanta."""

    def columns(self):
        rows, _ = P.pointer_column(FIXTURE, logits=text_only_logits)
        return {"a": rows, "b": [dict(r) for r in rows],
                "c": [dict(r) for r in rows]}

    def test_three_identical_columns_agree(self):
        got = P.same_rows(self.columns())
        self.assertTrue(got["pass"])
        self.assertEqual(got["n"], len(FIXTURE))
        self.assertEqual(got["columns"], ["a", "b", "c"])
        self.assertEqual(len(got["rows_sha256"]), 64)

    def test_a_column_short_by_one_row_fails_the_runner(self):
        cols = self.columns()
        cols["b"] = cols["b"][:-1]
        with self.assertRaises(P.ProtocolMismatch) as ctx:
            P.same_rows(cols)
        self.assertIn("do not share their n", str(ctx.exception))

    def test_same_n_but_other_rows_fails_the_runner(self):
        cols = self.columns()
        cols["b"][0]["row_id"] = "dev-fix-es-2k-t99a"
        with self.assertRaises(P.ProtocolMismatch) as ctx:
            P.same_rows(cols)
        self.assertIn("does not measure the same rows", str(ctx.exception))

    def test_a_moved_gold_slot_fails_the_runner(self):
        cols = self.columns()
        cols["c"][1]["gold_index"] = 0
        with self.assertRaises(P.ProtocolMismatch):
            P.same_rows(cols)

    def test_one_column_is_not_a_table(self):
        with self.assertRaises(P.ProtocolMismatch):
            P.same_rows({"a": self.columns()["a"]})

    def test_the_three_real_columns_share_the_400_rows(self):
        """Sobre la batería REAL, con predictores deterministas."""
        episodes = P.load_cut()
        qwen, _ = P.qwen_column(episodes, choose=text_only_choice)
        nli, _ = P.nli_column(episodes, score=text_only_scores)
        ptr, _ = P.pointer_column(episodes, logits=text_only_logits)
        got = P.same_rows({P.REF_QWEN: qwen, P.REF_NLI: nli,
                           P.REF_POINTER: ptr})
        self.assertTrue(got["pass"])
        self.assertEqual(got["n"], 400)


class ColumnShapeTest(unittest.TestCase):
    def test_the_nli_column_has_no_unknown_column(self):
        rows, meta = P.nli_column(FIXTURE, score=text_only_scores)
        self.assertEqual(meta["n"], len(FIXTURE))
        for row in rows:
            self.assertEqual(len(row["probs"]), row["k"])
            self.assertAlmostEqual(sum(row["probs"]), 1.0, places=6)
            self.assertLess(row["pred"], row["k"])

    def test_the_pointer_column_is_k_plus_one(self):
        rows, meta = P.pointer_column(FIXTURE, logits=text_only_logits)
        for row in rows:
            self.assertEqual(len(row["probs"]), row["k"] + 1)
        self.assertEqual(meta["abstained"], 0)

    def test_a_pointer_row_with_the_wrong_width_fails(self):
        def short(decisions):
            return [[0.0] * d.k for d in decisions]
        with self.assertRaises(P.ProtocolMismatch):
            P.pointer_column(FIXTURE, logits=short)

    def test_a_scorer_that_returns_fewer_scores_fails(self):
        with self.assertRaises(P.ProtocolMismatch):
            P.nli_column(FIXTURE, score=lambda pairs: [0.0])

    def test_an_invalid_qwen_reply_is_an_abstention_not_a_guess(self):
        rows, meta = P.qwen_column(FIXTURE, choose=unparseable_choice)
        self.assertEqual(meta["abstained"], len(FIXTURE))
        self.assertEqual(meta["rejected_replies"], len(FIXTURE))
        for row in rows:
            self.assertEqual(row["pred"], row["k"])

    def test_the_qwen_column_publishes_an_indicator_and_says_so(self):
        rows, _ = P.qwen_column(FIXTURE, choose=text_only_choice)
        for row in rows:
            self.assertEqual(sorted(set(row["probs"])), [0.0, 1.0])
        self.assertIn("INDICATOR", P.REFERENCES[P.REF_QWEN]["weights"])


class MechanismReproductionTest(unittest.TestCase):
    """El control reproduce el fallo documentado, y el comprobador separa."""

    def test_the_evidence_itself_shows_the_invariance(self):
        sig = P.documented_signature()
        self.assertEqual(sorted(sig["runs"]), sorted([FULLSPACE, LEVERSTACK]))
        for run in sig["runs"].values():
            self.assertEqual(run["rate"], 1.0)
            self.assertEqual(run["invariant"], run["n_pairs"])
            self.assertEqual(run["shapes"],
                             ["question_moved", "state_moved"])

    def test_a_text_only_predictor_never_moves_its_pick(self):
        rows, _ = P.pointer_column(FIXTURE, logits=text_only_logits)
        track = P.tracking_control(FIXTURE, rows, source="fixture")
        self.assertEqual(track["n"], 2)
        self.assertEqual(track["followed"], 0)
        self.assertEqual(track["same_slot_rate"], 1.0)
        self.assertFalse(track["pass"])

    def test_a_state_reading_predictor_moves_with_the_fact(self):
        rows, _ = P.pointer_column(FIXTURE, logits=state_reading_logits)
        track = P.tracking_control(FIXTURE, rows, source="fixture")
        self.assertEqual(track["followed"], 2)
        self.assertEqual(track["same_slot_rate"], 0.0)
        self.assertTrue(track["pass"])

    def report(self, logits):
        episodes = P.load_cut()
        rows, _ = P.pointer_column(episodes, logits=logits)
        doc = P.report_for(P.REF_POINTER, rows, episodes=episodes,
                           model_version=LEVERSTACK, permutation=perm())
        track = P.tracking_control(episodes, rows, source="test")
        return doc, track

    def test_the_battery_reproduces_the_documented_failure(self):
        doc, track = self.report(text_only_logits)
        got = P.reproduces(doc, track, P.documented_signature(), LEVERSTACK)
        self.assertTrue(got["reproduced"], json.dumps(got["checks"],
                                                      indent=1))
        self.assertEqual(doc["counterfactual"]["hits"], 0)
        self.assertEqual(track["same_slot_rate"], 1.0)
        self.assertLessEqual(doc["counterfactual"]["accuracy_ci95"][1],
                             doc["counterfactual"]["chance"])

    def test_the_checker_says_no_when_the_failure_is_absent(self):
        doc, track = self.report(state_reading_logits)
        got = P.reproduces(doc, track, P.documented_signature(), LEVERSTACK)
        self.assertFalse(got["reproduced"])
        self.assertFalse(got["checks"]["battery_invariance"]["pass"])
        self.assertFalse(got["checks"]["joint_at_or_below_chance"]["pass"])
        self.assertIn("not representative", got["reading"])

    def test_the_rule_was_written_before_the_measurement(self):
        got = P.reproduces({}, {"same_slot_rate": None},
                           P.documented_signature(), LEVERSTACK)
        self.assertEqual(got["rule"], P.REPRO_RULE)
        self.assertFalse(got["reproduced"])


class SealedCutTest(unittest.TestCase):
    """Ninguna referencia consulta el test sellado."""

    def test_the_guard_refuses_every_sealed_path(self):
        for path in ("data/battery_sealed.jsonl",
                     "artifacts/gates/T-battery-sealed/cut.jsonl",
                     "data/battery_diag.jsonl",
                     "/abs/path/to/battery-sealed.jsonl"):
            with self.assertRaises(P.SealedCutTouched):
                P.assert_not_sealed(path)

    def test_the_loader_refuses_a_cut_that_is_not_development(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "rows.jsonl"
            rows = [dict(FIXTURE[0], variant_group="sealed-fix-es-2k-t01")]
            bad.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            with self.assertRaises(P.SealedCutTouched):
                P.load_cut(bad)

    def test_a_full_pass_opens_only_the_declared_sources(self):
        opened = []
        real_open = builtins.open

        def recording(file, *args, **kwargs):
            opened.append(str(file))
            return real_open(file, *args, **kwargs)

        builtins.open = recording
        try:
            episodes = P.load_cut()
            qwen, _ = P.qwen_column(episodes, choose=text_only_choice)
            nli, _ = P.nli_column(episodes, score=text_only_scores)
            ptr, _ = P.pointer_column(episodes, logits=text_only_logits)
            P.same_rows({P.REF_QWEN: qwen, P.REF_NLI: nli,
                         P.REF_POINTER: ptr})
            P.report_for(P.REF_POINTER, ptr, episodes=episodes,
                         model_version=LEVERSTACK, permutation=perm())
            P.documented_signature()
        finally:
            builtins.open = real_open
        inside = sorted({P._rel(p) for p in opened
                         if str(p).startswith(str(ROOT))})
        self.assertTrue(inside, "the run opened nothing: the recorder is "
                                "not wired to the code under test")
        self.assertEqual(inside, sorted(set(P.SOURCES)))
        for path in opened:
            for marker in P.SEALED_MARKERS:
                self.assertNotIn(marker, str(path))

    def test_the_declared_sources_carry_no_sealed_marker(self):
        for source in P.SOURCES:
            P.assert_not_sealed(source)


class SuiteIsTheOnlyArithmeticTest(unittest.TestCase):
    """Toda cifra sale de `metrics_suite.report()` y pasa `require()`."""

    def columns(self):
        episodes = P.load_cut()
        return episodes, {
            P.REF_QWEN: P.qwen_column(episodes, choose=text_only_choice)[0],
            P.REF_NLI: P.nli_column(episodes, score=text_only_scores)[0],
            P.REF_POINTER: P.pointer_column(
                episodes, logits=text_only_logits)[0],
        }

    def test_every_column_publishes_a_complete_suite_report(self):
        episodes, cols = self.columns()
        for name, rows in sorted(cols.items()):
            doc = P.report_for(name, rows, episodes=episodes,
                               model_version=LEVERSTACK, permutation=perm())
            self.assertEqual(MS.check(doc), [], name)
            self.assertEqual(doc["format"], MS.FORMAT)
            self.assertEqual(doc["task"], P.TASK)
            self.assertEqual(doc["cut"]["n"], 400)
            for where in (doc["ranking"], doc["abstention"],
                          doc["counterfactual"]):
                for key in MS.FIGURE_KEYS:
                    self.assertIn(key, where, f"{name}/{key}")

    def test_the_report_carries_the_five_families_and_the_k_table(self):
        episodes, cols = self.columns()
        doc = P.report_for(P.REF_NLI, cols[P.REF_NLI], episodes=episodes,
                           model_version=LEVERSTACK, permutation=perm())
        self.assertEqual(len(doc["by_family"]), 5)
        self.assertEqual(sorted(doc["chance"]["by_k"]), ["2", "3", "8"])

    def test_the_temperature_is_fitted_and_verified_on_disjoint_halves(self):
        episodes, cols = self.columns()
        cal = P.dev_temperature(cols[P.REF_POINTER])
        self.assertNotEqual(cal["fitted_on"], cal["verified_on"])
        self.assertGreater(cal["temperature"], 0.0)
        fit = {r["variant_group"] for r in cols[P.REF_POINTER]
               if P._half(r["variant_group"]) == "fit"}
        ver = {r["variant_group"] for r in cols[P.REF_POINTER]
               if P._half(r["variant_group"]) == "verify"}
        self.assertEqual(fit & ver, set())
        self.assertTrue(fit and ver)

    def test_the_permutation_control_is_carried_in_never_invented(self):
        got = P.permutation_from_evidence(FULLSPACE)
        self.assertTrue(got["measured"])
        self.assertTrue(got["pass"])
        self.assertIn("probe-mechanism-2026-09-24.json", got["source"])
        with self.assertRaises(P.ProtocolMismatch):
            P.permutation_from_evidence("no-such-checkpoint")


class GateTest(unittest.TestCase):
    """El gate: lo medido con `pass`, lo pendiente con `null`."""

    def written(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = Path(tmp.name) / P.TASK
        d.mkdir(parents=True)
        old_dir, old_path = P.GATE_DIR, P.GATE_PATH
        P.GATE_DIR, P.GATE_PATH = d, d / "gate.json"
        try:
            doc = P.gate(write=True)
        finally:
            P.GATE_DIR, P.GATE_PATH = old_dir, old_path
        return d, doc

    def test_nothing_unmeasured_is_written_as_a_number(self):
        _, doc = self.written()
        awaiting = {k: v for k, v in doc["checks"].items()
                    if v.get("pass") is None}
        self.assertTrue(awaiting)
        for name, check in awaiting.items():
            self.assertEqual(check["reason"], "awaiting-operator-compute",
                             name)
            self.assertTrue(check["needs_compute"], name)
            self.assertIn("signed_by", check, name)
            self.assertEqual(check["job"], "preflight-refs", name)

    def test_the_measured_checks_say_how_they_were_measured(self):
        _, doc = self.written()
        measured = {k: v for k, v in doc["checks"].items()
                    if v.get("pass") is not None}
        self.assertTrue(measured)
        for name, check in measured.items():
            self.assertIn("how_measured", check, name)
            self.assertFalse(check["needs_compute"], name)
            self.assertTrue(check["pass"], name)

    def test_the_gate_claims_nothing_green(self):
        _, doc = self.written()
        self.assertIsNone(doc["pass"])
        self.assertFalse(GR.green_claim(doc)["green"])

    def test_the_gate_directory_passes_c1_to_c7(self):
        d, _ = self.written()
        got = GR.check_gate_dir(d)
        self.assertEqual(got["errors"], [], json.dumps(got["errors"],
                                                       indent=1))
        self.assertTrue(got["pass"])
        self.assertTrue(got["metrics_suite"]["pilot"])
        self.assertEqual(got["metrics_suite"]["hand_computed"], 0)
        self.assertEqual(got["publishes"]["quality_metrics"], 0)

    def test_the_gate_records_the_job_the_operator_presses(self):
        _, doc = self.written()
        self.assertEqual(doc["job"]["id"], "preflight-refs")
        self.assertIn("eval.preflight_refs refs", doc["job"]["command"])
        self.assertIn("STOPPED", doc["job"]["state"])


class ProtocolTest(unittest.TestCase):
    def test_the_three_columns_are_the_ones_the_task_names(self):
        self.assertEqual(sorted(P.REFERENCES),
                         sorted([P.REF_NLI, P.REF_POINTER, P.REF_QWEN]))
        self.assertEqual(sorted(P.POINTER_CHECKPOINTS),
                         ["fullspace-62528", "leverstack-1m"])
        for spec in P.POINTER_CHECKPOINTS.values():
            self.assertTrue((ROOT / spec["dir"]).is_dir(), spec["dir"])

    def test_the_format_is_the_frozen_one(self):
        proto = P.protocol()
        self.assertEqual(proto["hypothesis_format_fingerprint"],
                         proto["hypothesis_format_id"])
        self.assertEqual(len(proto["comparative_context_by_family"]), 5)
        self.assertEqual(sorted(proto["per_model"]), sorted(P.REFERENCES))

    def test_the_teacher_is_a_fourth_column_and_blocks_nothing(self):
        self.assertFalse(P.TEACHER_COLUMN["present"])
        self.assertIn("backlog", P.TEACHER_COLUMN["why"])

    def test_the_cut_is_the_development_one_and_carries_its_seal(self):
        cut = P.cut_descriptor(P.load_cut())
        self.assertFalse(cut["reserved"])
        self.assertEqual(cut["split"], "development")
        self.assertEqual(len(cut["split_sha256"]), 64)
        self.assertFalse(cut["sealed_cut_read"])


#: Una pasada COMPLETA del camino caliente, en un intérprete limpio: el
#: único sitio donde «no carga un modelo» se puede afirmar, porque en el
#: mismo proceso otros tests del directorio sí importan torch.
COLD_PASS = """
import json, sys
from eval import preflight_refs as P
from eval.test_preflight_refs import (LEVERSTACK, text_only_choice,
                                      text_only_logits, text_only_scores)
episodes = P.load_cut()
cols = {P.REF_QWEN: P.qwen_column(episodes, choose=text_only_choice)[0],
        P.REF_NLI: P.nli_column(episodes, score=text_only_scores)[0],
        P.REF_POINTER: P.pointer_column(episodes,
                                        logits=text_only_logits)[0]}
P.same_rows(cols)
for name, rows in sorted(cols.items()):
    P.report_for(name, rows, episodes=episodes, model_version=LEVERSTACK,
                 permutation=P.permutation_from_evidence(LEVERSTACK))
P.documented_signature()
P.gate(write=False)
print(json.dumps(sorted(m for m in sys.modules
                        if m.split(".")[0] in ("torch", "transformers",
                                               "safetensors"))))
"""


class NoModelTest(unittest.TestCase):
    """Ni torch, ni transformers, ni una descarga — comprobado en frío."""

    def test_a_full_pass_in_a_clean_interpreter_loads_no_model(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        got = subprocess.run([sys.executable, "-c", COLD_PASS], cwd=ROOT,
                             env=env, capture_output=True, text=True)
        self.assertEqual(got.returncode, 0, got.stderr[-2000:])
        self.assertEqual(json.loads(got.stdout.strip().splitlines()[-1]), [],
                         "the hot path pulled in a model stack")

    def test_the_module_itself_imports_no_model_stack(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        got = subprocess.run(
            [sys.executable, "-c",
             "import sys, json; import eval.preflight_refs; "
             "print(json.dumps([m for m in sys.modules "
             "if m.split('.')[0] in ('torch', 'transformers')]))"],
            cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(got.returncode, 0, got.stderr[-2000:])
        self.assertEqual(json.loads(got.stdout.strip()), [])


if __name__ == "__main__":
    unittest.main()
