"""Tests del verificador separado (#T-episode-verify).

El gate, ejecutable sin Qwen (el auditor se sustituye por un doble que
conoce la verdad ORIGINAL de cada episodio, no el gold que se le pasa):

* un gold manipulado a mano cae en cuarentena — por la regla en las
  familias numéricas, por el verificador en las semánticas — y nunca pasa;
* el embudo generado → aceptado → publicado → verificado → consumido se
  reconstruye desde los manifests, incluido el `train_manifest.json` que
  cite el run;
* la muestra humana tiene el tamaño fijado antes y `human_error: null`.

Run:
    PYTHONPATH=. pytest data/test_episode_verify.py -q
"""

from __future__ import annotations

import json
import os
import re
import shutil
import unittest
import unittest.mock

from . import episode_contract as EC
from . import episode_gen as EG
from . import episode_verify as V

SEED = 20260926
TMP = "/tmp/episode-verify-test"


def _fresh_run(name: str, n: int = 40) -> str:
    out = os.path.join(TMP, name)
    shutil.rmtree(out, ignore_errors=True)
    got = EG.run(n=n, seed=SEED, prose="local", teacher="stub", out_dir=out)
    assert got["n_episodes"] == n, got
    return out


def _truth_oracle(episodes: list[dict]):
    """Un auditor de mentira que sabe la verdad ORIGINAL por texto de
    contexto+pregunta, no por el gold que viaja en el episodio."""
    truth = {(ep["state"], ep["question"]): ep["answer"] for ep in episodes}

    def fake(prompt: str, num_predict: int, timeout: int) -> str:
        lines = []
        for i, block in enumerate(prompt.split("## Item ")[1:], 1):
            state = block.split("Context: ", 1)[1].split("\nQuestion:", 1)[0]
            q = block.split("Question: ", 1)[1].split("\nOptions:", 1)[0]
            lines.append(f"{i}: {truth[(state, q)]}")
        return "\n".join(lines)
    return fake


def _rewrite(out: str, episodes: list[dict]) -> None:
    with open(os.path.join(out, "episodes.jsonl"), "w", encoding="utf-8") as fh:
        for ep in episodes:
            fh.write(json.dumps(ep, ensure_ascii=False) + "\n")


class RuleVerifyTest(unittest.TestCase):
    def test_rule_recomputes_the_published_gold_in_every_family(self):
        out = _fresh_run("rule")
        seen = set()
        for ep in EG.load_episodes(out):
            rule = V.rule_verify(ep)
            self.assertTrue(rule["applies"], ep["id"])
            self.assertTrue(rule["agree"], (ep["id"], rule))
            seen.add(ep["family"])
        self.assertEqual(seen, set(EC.FAMILIES))

    def test_a_manipulated_numeric_gold_is_caught_by_the_rule_alone(self):
        out = _fresh_run("rule-flip")
        for ep in EG.load_episodes(out):
            if ep["family"] not in (EC.COMPARISON, EC.PRIORITY):
                continue
            bad = dict(ep)
            bad["answer"] = "c2" if ep["answer"] == "c1" else "c1"
            rule = V.rule_verify(bad)
            self.assertFalse(rule["agree"])
            self.assertTrue(any(r.startswith("rule:") for r in rule["reasons"]))
            kind, reasons = V.decide(bad, rule, None)
            self.assertEqual(kind, "quarantine")

    def test_prompt_marker_in_the_state_is_a_leak_not_an_episode(self):
        out = _fresh_run("leak", n=10)
        ep = EG.load_episodes(out)[0]
        ep["state"] += " KEEP: " + ep["evidence"]
        rule = V.rule_verify(ep)
        self.assertTrue(any(r.startswith("prompt-leak:") for r in rule["reasons"]))

    def test_no_trace_means_the_rule_does_not_apply_not_that_it_approves(self):
        rule = V.rule_verify({"family": EC.EXTRACTION, "lang": "es",
                              "state": "x y z", "question": "¿Dónde está x?",
                              "candidates": [{"id": "c1", "text": "a"},
                                             {"id": "c2", "text": "b"}],
                              "answer": "c1", "evidence": "x", "origin": "t",
                              "generator_seed": 1, "generator_version": "t",
                              "variant_group": "g"})
        self.assertFalse(rule["applies"])
        self.assertIsNone(rule["agree"])


class VerifierPromptTest(unittest.TestCase):
    def _case(self):
        return {"id": "ep-x-01", "state": "El tren cuesta 40 euros.",
                "question": "¿Cuál es más barato?",
                "candidates": [{"id": "c1", "text": "tren"},
                               {"id": "c2", "text": "bus"}],
                "answer": "c1", "evidence": "cuesta 40 euros"}

    def test_the_auditor_never_sees_gold_or_evidence(self):
        prompt = V.verifier_prompt([self._case()])
        self.assertIsNone(re.search(r"^\s*(answer|gold|evidence)\s*:", prompt,
                                    re.IGNORECASE | re.MULTILINE), prompt)
        self.assertNotIn("evidence", prompt.lower())
        self.assertNotIn("gold", prompt.lower())

    def test_the_auditor_prompt_is_not_the_generator_teacher_prompt(self):
        prompt = V.verifier_prompt([self._case()] * 2)
        # Lo que el profesor del generador usa y el auditor no.
        for marker in ("### Item", "Facts:", "JSON", "\"choice\""):
            self.assertNotIn(marker, prompt)
        self.assertIn("independent auditor", prompt)
        self.assertIn("Context:", prompt)

    def test_options_are_shuffled_per_episode_but_stably(self):
        a = V.verifier_prompt([self._case()])
        b = V.verifier_prompt([self._case()])
        self.assertEqual(a, b)
        many = [dict(self._case(), id=f"ep-x-{i:02d}") for i in range(12)]
        orders = {V.verifier_prompt([c]).split("Options:\n", 1)[1]
                  for c in many}
        self.assertGreater(len(orders), 1)

    def test_parse_a_missing_line_is_none_not_a_shift(self):
        self.assertEqual(V.parse_verdicts("1: c1\n3: c2", 3), ["c1", None, "c2"])
        self.assertEqual(V.parse_verdicts("Item 1: (c2)\n2) none", 2),
                         ["c2", "none"])

    def test_every_non_verdict_is_a_status_never_a_choice(self):
        cases = [self._case()] * 4
        reply = "1: c1\n2: none\n3: c9"
        with unittest.mock.patch.object(V, "_chat", lambda *a, **k: reply):
            got = V.verify_batch(cases)
        self.assertEqual([t["status"] for t in got],
                         ["verdict", "abstain", "invalid", "missing"])
        self.assertEqual(got[0]["choice"], "c1")
        self.assertTrue(all(t["choice"] is None for t in got[1:]))
        with unittest.mock.patch.object(V, "_chat", lambda *a, **k: None):
            got = V.verify_batch(cases[:1])
        self.assertEqual(got[0]["status"], "no-reply")

    def test_decide_quarantines_everything_but_an_agreeing_verdict(self):
        ep = self._case()
        rule = {"applies": False, "expected": None, "agree": None, "reasons": []}
        for status in ("abstain", "invalid", "missing", "no-reply"):
            kind, reasons = V.decide(ep, rule, {"status": status, "choice": None})
            self.assertEqual(kind, "quarantine", status)
            self.assertIn(f"verifier: {status}", reasons)
        kind, _ = V.decide(ep, rule, {"status": "verdict", "choice": "c2"})
        self.assertEqual(kind, "quarantine")
        kind, _ = V.decide(ep, rule, {"status": "verdict", "choice": "c1"})
        self.assertEqual(kind, "verified")
        kind, _ = V.decide(ep, rule, None)
        self.assertEqual(kind, "quarantine")


class ManipulatedGoldTest(unittest.TestCase):
    """Gate 1: un gold manipulado a mano cae en cuarentena, no pasa."""

    @classmethod
    def setUpClass(cls):
        cls.out = _fresh_run("manipulated", n=40)
        eps = EG.load_episodes(cls.out)
        cls.oracle = _truth_oracle(eps)
        cls.tampered = {}
        for ep in eps:
            fam = ep["family"]
            if fam not in cls.tampered:  # uno por familia, las cinco
                ep["answer"] = "c2" if ep["answer"] == "c1" else "c1"
                cls.tampered[fam] = ep["id"]
        _rewrite(cls.out, eps)
        with unittest.mock.patch.object(V, "_chat", cls.oracle):
            cls.manifest = V.run(cls.out, batch=7, concurrency=2,
                                 checkpoints_root=os.path.join(cls.out, "ck"))
        cls.quarantine = V._read_jsonl(os.path.join(cls.out, "quarantine.jsonl"))
        cls.verified = V._read_jsonl(os.path.join(cls.out, "verified.jsonl"))

    def test_every_tampered_episode_is_quarantined_with_a_reason(self):
        q_ids = {q["episode"]["id"]: q["reasons"] for q in self.quarantine}
        self.assertEqual(len(self.tampered), 5)
        for fam, eid in self.tampered.items():
            self.assertIn(eid, q_ids, fam)
            self.assertTrue(any(r.startswith("verifier:") for r in q_ids[eid]),
                            q_ids[eid])
            if fam in (EC.COMPARISON, EC.PRIORITY):
                self.assertTrue(any(r.startswith("rule:") for r in q_ids[eid]))

    def test_nothing_tampered_passes_and_nothing_honest_is_lost(self):
        v_ids = {ep["id"] for ep in self.verified}
        self.assertTrue(v_ids.isdisjoint(set(self.tampered.values())))
        self.assertEqual(len(self.verified) + len(self.quarantine), 40)
        self.assertEqual(len(self.quarantine), 5)

    def test_quarantine_is_an_artifact_and_the_manifest_counts_it(self):
        self.assertEqual(self.manifest["n_quarantine"], 5)
        self.assertEqual(self.manifest["quarantine_reasons"]["verifier:disagree"], 5)
        # La regla aplica en las cinco familias del generador: las cinco caen
        # también por regla, no sólo las dos numéricas.
        self.assertEqual(self.manifest["quarantine_reasons"]["rule:disagree"], 5)
        self.assertEqual(self.manifest["status"], "published")
        self.assertEqual(len(self.manifest["prompt_sha"]), 64)

    def test_gate_check_flips_every_verified_gold_and_catches_all(self):
        check = V.manipulated_gold_check(self.verified)
        self.assertTrue(check["pass"])
        self.assertEqual(check["n_manipulated"], 35)
        self.assertEqual(check["n_quarantined"], 35)

    def test_gate_agreement_has_n_and_ci_per_cell_and_human_error_null(self):
        with unittest.mock.patch.object(V, "GATE_PATH",
                                        os.path.join(self.out, "gate.json")):
            gate = V.measure_gate(self.out, os.path.join(self.out, "ck"))
        agr = gate["checks"]["agreement_published"]
        self.assertTrue(agr["pass"])
        self.assertEqual(agr["overall"]["n"], 40)
        self.assertEqual(agr["overall"]["agree"], 35)
        for cell, row in agr["by_cell"].items():
            self.assertEqual(len(row["ci95"]), 2)
            self.assertLessEqual(row["ci95"][0], row["rate"])
        self.assertEqual(len(agr["by_family"]), 5)
        human = gate["checks"]["human_error_published"]
        self.assertIsNone(human["pass"])
        self.assertIsNone(human["human_error"])
        self.assertEqual(human["status"], "awaiting-operator")
        self.assertEqual(human["per_cell_declared"], V.HUMAN_PER_CELL)
        self.assertFalse(gate["resolved"])
        self.assertTrue(gate["checks"]["manipulated_gold_quarantined"]["pass"])
        self.assertTrue(gate["checks"]["funnel_from_manifests"]["pass"])


class HumanSampleTest(unittest.TestCase):
    def test_size_is_declared_before_and_rows_are_unreviewed(self):
        out = _fresh_run("human", n=60)
        eps = EG.load_episodes(out)
        with unittest.mock.patch.object(V, "_chat", _truth_oracle(eps)):
            V.run(out, batch=10, checkpoints_root=os.path.join(out, "ck"))
        rows = V._read_jsonl(os.path.join(out, "human_sample.jsonl"))
        per_cell: dict = {}
        for r in rows:
            self.assertIsNone(r["human_error"])
            self.assertEqual(r["status"], "awaiting-operator")
            self.assertIn(r["variant_type"], V.VARIANT_TYPES)
            k = (r["family"], r["lang"], r["variant_type"])
            per_cell[k] = per_cell.get(k, 0) + 1
        self.assertTrue(per_cell)
        self.assertTrue(all(v <= V.HUMAN_PER_CELL for v in per_cell.values()))
        again = V.human_sample(V._read_jsonl(os.path.join(out, "verified.jsonl")))
        self.assertEqual([r["id"] for r in again], [r["id"] for r in rows])

    def test_variant_type_from_the_generator_plan(self):
        out = _fresh_run("variant", n=8)
        eps = EG.load_episodes(out)
        types = [V.variant_type(ep) for ep in eps]
        self.assertEqual(types[:2], ["base", "counterfactual"])
        self.assertEqual(V.variant_type({"id": "ep-g01a"}), "unknown")


class FunnelTest(unittest.TestCase):
    """Gate 2: el embudo se reconstruye desde los manifests, por familia."""

    def test_funnel_from_manifests_with_and_without_a_training_run(self):
        out = _fresh_run("funnel", n=40)
        eps = EG.load_episodes(out)
        eps[3]["answer"] = "c2" if eps[3]["answer"] == "c1" else "c1"
        _rewrite(out, eps)
        ck = os.path.join(out, "ck")
        with unittest.mock.patch.object(V, "_chat", _truth_oracle(eps)):
            V.run(out, batch=10, checkpoints_root=ck)
        funnel = json.load(open(os.path.join(out, "funnel.json")))
        st = funnel["stages"]
        self.assertEqual(st["planned"]["total"], 40)
        self.assertEqual(st["generated"]["total"], 40)
        self.assertEqual(st["published"]["total"], 40)
        self.assertEqual(st["verified"]["total"], 39)
        self.assertEqual(st["verified"]["quarantined"], 1)
        self.assertEqual(st["consumed"]["total"], 0)
        self.assertEqual(st["consumed"]["train_manifests"], [])

        # Un entreno que cita este run y dice qué ids consumió.
        verified = V._read_jsonl(os.path.join(out, "verified.jsonl"))
        used = [ep["id"] for ep in verified[:12]]
        os.makedirs(os.path.join(ck, "ce", "run-1"), exist_ok=True)
        with open(os.path.join(ck, "ce", "run-1", "train_manifest.json"), "w") as fh:
            json.dump({"episodes": {
                "source": os.path.relpath(out, V.ROOT),
                "consumed_ids": used}}, fh)
        # Y otro que NO cita este run: no cuenta.
        os.makedirs(os.path.join(ck, "ce", "other"), exist_ok=True)
        with open(os.path.join(ck, "ce", "other", "train_manifest.json"), "w") as fh:
            json.dump({"episodes": {"source": "elsewhere",
                                    "consumed_ids": used}}, fh)
        rebuilt = V.build_funnel(out, ck)
        self.assertEqual(rebuilt["stages"]["consumed"]["total"], 12)
        expect: dict = {}
        for ep in verified[:12]:
            expect[V.cell_of(ep)] = expect.get(V.cell_of(ep), 0) + 1
        self.assertEqual(rebuilt["stages"]["consumed"]["by_cell"], expect)
        self.assertEqual(len(rebuilt["stages"]["consumed"]["train_manifests"]), 1)
        self.assertEqual(rebuilt["stages"]["verified"]["by_cell"],
                         V._count_cells(verified))

    def test_wilson_interval_is_the_honest_one(self):
        lo, hi = V.wilson_interval(0, 0)
        self.assertEqual((lo, hi), (0.0, 1.0))
        lo, hi = V.wilson_interval(95, 100)
        self.assertLess(lo, 0.95)
        self.assertGreater(hi, 0.95)
        self.assertLess(hi, 1.0)


if __name__ == "__main__":
    unittest.main()
