"""Tests del generador de episodios (#T-episode-gen).

El gate, ejecutable:

* el 100 % de lo publicado pasa el validador de `#T-episode-contract`
  y los rechazos quedan con motivo;
* en comparación, perturbar un atributo del estado cambia el gold de
  la regla (el generador no puede publicar un gold contradicho);
* ningún episodio lleva el marcador de plantilla de `data/labelgen.py`
  ni una pregunta igual a un ID de dataset.

Run (UN directorio cada vez — colisión de basenames en este repo):
    PYTHONPATH=. pytest data/test_episode_gen.py -q
"""

from __future__ import annotations

import json
import random
import re
import unittest
import unittest.mock

from . import episode_contract as EC
from . import episode_gen as EG

SEED = 20260926


def _smoke_episodes(n: int = 20) -> list[dict]:
    out = EG.run(n=n, seed=SEED, prose="local", teacher="stub",
                 out_dir="/tmp/episode-gen-test")
    assert out["n_episodes"] == n, out
    return EG.load_episodes(out["out"])


class SplitDeclaredTest(unittest.TestCase):
    def test_plan_covers_five_families_two_langs(self):
        plan = EG.declared_split(2000, SEED)
        self.assertEqual(len(plan), 10)
        for f in EC.FAMILIES:
            for lang in EC.LANGS:
                self.assertGreater(plan[(f, lang)], 0)
        self.assertEqual(sum(plan.values()), 2000)

    def test_plan_deterministic(self):
        self.assertEqual(EG.declared_split(25, SEED),
                         EG.declared_split(25, SEED))

    def test_manifest_declared_before_generating(self):
        import json
        out = EG.run(n=10, seed=SEED, prose="local", teacher="stub",
                     out_dir="/tmp/episode-gen-test-plan")
        with open(out["out"] + "/manifest.json") as fh:
            manifest = json.load(fh)
        self.assertIn("planned_split", manifest)
        self.assertEqual(sum(manifest["planned_split"].values()), 10)
        self.assertEqual(manifest["schema_sha"], EC.schema_sha())


class RuleGoldTest(unittest.TestCase):
    def test_comparison_deterministic(self):
        attrs = [{"price_eur": 42, "duration_h": 3},
                 {"price_eur": 25, "duration_h": 5}]
        self.assertEqual(EG.comparison_gold(attrs, "price"), 1)
        self.assertEqual(EG.comparison_gold(attrs, "duration"), 0)

    def test_comparison_perturb_flips_gold(self):
        rng = random.Random(0)
        for _ in range(20):
            p1, p2 = rng.randint(15, 60), rng.randint(15, 60)
            if p1 == p2:
                continue
            attrs = [{"price_eur": p1, "duration_h": 2},
                     {"price_eur": p2, "duration_h": 2}]
            before = EG.comparison_gold(attrs, "price")
            attrs[1 - before]["price_eur"] = min(p1, p2) - 3
            self.assertNotEqual(EG.comparison_gold(attrs, "price"), before)

    def test_priority_tie_goes_to_price(self):
        attrs = [{"price_eur": 180, "duration_h": 2},
                 {"price_eur": 150, "duration_h": 2}]
        self.assertEqual(EG.priority_gold(attrs), 1)

    def test_smoke_rule_golds_match_recomputation(self):
        for ep in _smoke_episodes():
            rt = ep["rule_trace"]
            cands = [c["id"] for c in ep["candidates"]]
            if ep["family"] == EC.COMPARISON:
                gold = EG.comparison_gold(rt["attrs"], rt["criterion"])
            elif ep["family"] == EC.PRIORITY:
                gold = EG.priority_gold(rt["attrs"])
            else:
                continue
            self.assertEqual(ep["answer"], cands[gold])


class GateOnSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episodes = _smoke_episodes(20)

    def test_validator_pass_100pc(self):
        per = EC.batch_validate(self.episodes)
        self.assertEqual(per["n_invalid"], 0, per["per_episode"])

    def test_no_labelgen_marker_no_dataset_id(self):
        for ep in self.episodes:
            for m in EG.LABELGEN_MARKERS:
                self.assertNotIn(m, ep["state"])
                self.assertNotIn(m, ep["question"])
            self.assertFalse(
                EC.is_dataset_id_question(ep["question"]), ep["question"])

    def test_teacher_trace_never_a_probability(self):
        for ep in self.episodes:
            trace = ep["teacher_trace"]
            self.assertIn(trace["choice"],
                          [c["id"] for c in ep["candidates"]])
            self.assertTrue(trace.get("confidence_discarded"))
            self.assertNotIn("probability", trace)
            self.assertNotIn("confidence", trace)

    def test_counterfactual_pairs_share_group(self):
        groups: dict[str, set] = {}
        for ep in self.episodes:
            groups.setdefault(ep["variant_group"], set()).add(ep["id"])
        self.assertTrue(any(len(v) >= 2 for v in groups.values()))

    def test_reproducible_by_seed(self):
        again = EG.run(n=20, seed=SEED, prose="local", teacher="stub",
                       out_dir="/tmp/episode-gen-test-repro")
        eps = EG.load_episodes(again["out"])
        self.assertEqual([e["state"] for e in eps],
                         [e["state"] for e in self.episodes])


if __name__ == "__main__":
    unittest.main()


# -- Qwen por lotes: un request lleva muchos episodios ----------------------
# Ollama cuesta ~19 s fijos por petición contra ~3 s de cómputo en
# qwen3.6:27b-mlx (medido 2026-09-27), así que el piloto sólo es viable
# agrupando. Lo que NO puede cambiar al agrupar es qué se acepta.
def _fake_ollama(messages, num_predict, timeout,
                 temperature=EG.TEMPERATURE):
    """Qwen de mentira, determinista: parafrasea copiando, elige el 1.er id."""
    prompt = messages[0]["content"]
    if "Rewrite EACH numbered fact-set" in prompt:
        body = prompt.split("nothing else.\n", 1)[1]
        out = []
        for ln in body.splitlines():
            if not ln.strip() or ln.lstrip().startswith("KEEP:"):
                continue  # el tramo a conservar ya viaja dentro de los hechos
            num, facts = ln.split(".", 1)
            out.append(f"{num}. {facts.strip()} (reescrito)")
        return "\n".join(out)
    if "Rewrite these facts" in prompt:
        facts = prompt.split("rewritten sentences.\n", 1)[1]
        return f"{facts.split(chr(10) + 'KEEP:')[0]} (reescrito)"
    if "For EACH of the" in prompt:
        out = []
        for i, block in enumerate(prompt.split("### Item ")[1:], 1):
            facts = block.split("Facts: ", 1)[1].split("\nQuestion:", 1)[0]
            cid = block.split("  - ", 1)[1].split(":", 1)[0]
            out.append(json.dumps({"item": i, "choice": cid,
                                   "evidence": facts[:18]}))
        return "\n".join(out)
    facts = prompt.split("Facts: ", 1)[1].split("\nQuestion:", 1)[0]
    cid = prompt.split("- ", 1)[1].split(":", 1)[0]
    return json.dumps({"choice": cid, "evidence": facts[:18]})


class NumberedRepliesTest(unittest.TestCase):
    def test_parses_one_slot_per_item(self):
        got = EG._numbered_lines("1. uno\n2. dos\n3. tres", 3)
        self.assertEqual(got, ["uno", "dos", "tres"])

    def test_a_skipped_item_is_none_not_a_shift(self):
        # Lo peligroso de agrupar: que falte el 2 y el 3 ocupe su hueco.
        got = EG._numbered_lines("1. uno\n3. tres", 3)
        self.assertEqual(got, ["uno", None, "tres"])

    def test_prose_around_the_list_is_ignored(self):
        got = EG._numbered_lines("Claro, aquí tienes:\n1. uno\ngracias", 1)
        self.assertEqual(got, ["uno"])


class BatchedTeacherTest(unittest.TestCase):
    def _cases(self, n=3):
        return [{"state": f"El tren cuesta {40 + i} euros.",
                 "question": "¿Cuál?",
                 "candidates": [{"id": "c1", "text": "tren"},
                                {"id": "c2", "text": "bus"}]}
                for i in range(n)]

    def test_batched_and_single_accept_the_same_thing(self):
        with unittest.mock.patch.object(EG, "_ollama_chat", _fake_ollama):
            batched = EG.teacher_structured_batch(self._cases(3))
            single = [EG.teacher_structured(c["state"], c["question"],
                                            c["candidates"])
                      for c in self._cases(3)]
        for b, s in zip(batched, single):
            self.assertEqual(b["choice"], s["choice"])
            self.assertEqual(b["evidence"], s["evidence"])

    def test_a_missing_item_rejects_with_reason_not_silently(self):
        reply = '{"item": 1, "choice": "c1", "evidence": "El tren cuesta 40"}'
        with unittest.mock.patch.object(
                EG, "_ollama_chat", lambda *a, **k: reply):
            got = EG.teacher_structured_batch(self._cases(3))
        self.assertIn("choice", got[0])
        for trace in got[1:]:
            self.assertNotIn("choice", trace)
            self.assertTrue(trace["reject"])

    def test_an_invented_id_is_rejected(self):
        reply = '{"item": 1, "choice": "c9", "evidence": "El tren cuesta 40"}'
        with unittest.mock.patch.object(
                EG, "_ollama_chat", lambda *a, **k: reply):
            got = EG.teacher_structured_batch(self._cases(1) * 1)
        self.assertNotIn("choice", got[0])

    def test_evidence_not_in_the_facts_is_rejected(self):
        reply = '{"item": 1, "choice": "c1", "evidence": "el avión despega"}'
        with unittest.mock.patch.object(
                EG, "_ollama_chat", lambda *a, **k: reply):
            got = EG.teacher_structured_batch(self._cases(2))
        self.assertNotIn("choice", got[0])

    def test_a_confidence_never_survives_the_batch(self):
        reply = ('{"item": 1, "choice": "c1", "confidence": 0.93, '
                 '"evidence": "El tren cuesta 40"}')
        with unittest.mock.patch.object(
                EG, "_ollama_chat", lambda *a, **k: reply):
            got = EG.teacher_structured_batch(self._cases(2))
        self.assertTrue(got[0]["confidence_discarded"])
        self.assertNotIn("confidence", got[0])
        self.assertNotIn("probability", got[0])


class BatchingChangesNothingTest(unittest.TestCase):
    def test_batch_12_publishes_exactly_what_batch_1_publishes(self):
        with unittest.mock.patch.object(EG, "_ollama_chat", _fake_ollama):
            a = EG.run(n=24, seed=SEED, prose="qwen", teacher="qwen",
                       out_dir="/tmp/episode-gen-test-b1", batch=1)
            b = EG.run(n=24, seed=SEED, prose="qwen", teacher="qwen",
                       out_dir="/tmp/episode-gen-test-b12", batch=12,
                       concurrency=4)
        self.assertEqual(a["n_episodes"], b["n_episodes"])
        self.assertEqual(EG.load_episodes(a["out"]),
                         EG.load_episodes(b["out"]))
        self.assertEqual(a["prose_origins"], b["prose_origins"])

    def test_lost_rule_token_keeps_the_local_prose(self):
        def drops_the_numbers(messages, num_predict, timeout):
            out = _fake_ollama(messages, num_predict, timeout)
            return re.sub(r"\d+", "N", out) \
                if "Rewrite EACH" in messages[0]["content"] else out

        with unittest.mock.patch.object(EG, "_ollama_chat",
                                        drops_the_numbers):
            out = EG.run(n=20, seed=SEED, prose="qwen", teacher="stub",
                         out_dir="/tmp/episode-gen-test-lost", batch=10)
        self.assertEqual(out["prose_origins"].get("qwen-local"), None)
        self.assertEqual(out["prose_origins"]["local-fallback"],
                         out["n_episodes"])


class VolumeCheckTest(unittest.TestCase):
    def test_unpublished_directory_is_not_signed(self):
        check = EG._volume_check("/tmp/episode-gen-test-nope", 5, "datagen")
        self.assertIsNone(check["pass"])
        self.assertEqual(check["status"], "awaiting-operator-compute")

    def test_a_short_pilot_fails_it_does_not_stay_null(self):
        out = EG.run(n=20, seed=SEED, prose="local", teacher="stub",
                     out_dir="/tmp/episode-gen-test-short")
        check = EG._volume_check(out["out"], 20, "datagen")
        self.assertIs(check["pass"], False)
        self.assertEqual(check["status"], "measured")
        self.assertEqual(check["n_published"], 20)
        self.assertEqual(len(check["manifest_sha"]), 64)
        self.assertEqual(check["seed"], SEED)


class EvidenceSurvivesTheParaphraseTest(unittest.TestCase):
    """El fallo que se comió el 46 % del primer tramo del piloto.

    `episode-v1` exige que `evidence` sea un fragmento literal de
    `state`. Cuando Qwen reescribía el estado, el tramo dejaba de estar
    y el episodio se iba a rechazos — no por culpa del profesor ni del
    gold, sino por prosa. Prosa que pierde el tramo es prosa que se
    descarta; el episodio se publica con la que escribió la regla.
    """

    def test_prose_that_drops_the_span_falls_back_instead_of_rejecting(self):
        def eats_the_span(messages, num_predict, timeout):
            out = _fake_ollama(messages, num_predict, timeout)
            if "Rewrite" not in messages[0]["content"]:
                return out
            return "\n".join(f"{ln.split('.', 1)[0]}. Reescrito del todo."
                             for ln in out.splitlines() if ln.strip()) \
                if "EACH" in messages[0]["content"] else "Reescrito del todo."

        with unittest.mock.patch.object(EG, "_ollama_chat", eats_the_span):
            out = EG.run(n=20, seed=SEED, prose="qwen", teacher="stub",
                         out_dir="/tmp/episode-gen-test-span", batch=10)
        self.assertEqual(out["n_episodes"], 20)
        self.assertEqual(out["n_rejects"], 0)
        self.assertEqual(out["prose_origins"], {"local-fallback": 20})

    def test_the_span_is_asked_for_in_the_prompt(self):
        seen = {}

        def capture(messages, num_predict, timeout):
            seen.setdefault("prompt", messages[0]["content"])
            return _fake_ollama(messages, num_predict, timeout)

        with unittest.mock.patch.object(EG, "_ollama_chat", capture):
            EG.run(n=10, seed=SEED, prose="qwen", teacher="stub",
                   out_dir="/tmp/episode-gen-test-keep", batch=10)
        self.assertIn("KEEP:", seen["prompt"])
        self.assertIn("word for word", seen["prompt"])

    def test_accepted_prose_is_prose_the_validator_accepts(self):
        with unittest.mock.patch.object(EG, "_ollama_chat", _fake_ollama):
            out = EG.run(n=20, seed=SEED, prose="qwen", teacher="stub",
                         out_dir="/tmp/episode-gen-test-ok", batch=10)
        eps = EG.load_episodes(out["out"])
        self.assertEqual(out["prose_origins"], {"qwen-local": 20})
        for ep in eps:
            self.assertTrue(
                EC.evidence_is_fragment(ep["evidence"], ep["state"]), ep["id"])

    def test_the_capitalised_first_word_does_not_kill_the_paraphrase(self):
        # Los builders ponen en mayúscula la primera palabra de la prosa;
        # comparando con mayúsculas exactas, extracción caía SIEMPRE a
        # prosa local y Qwen no escribía esa familia nunca.
        self.assertTrue(EG._kept("The archive key is at drawer 2.",
                                 ["the archive key", "drawer 2"]))
        self.assertFalse(EG._kept("The archive key is at drawer 9.",
                                  ["the archive key", "drawer 2"]))


class ResumeTest(unittest.TestCase):
    """A relaunch must continue the dead run, not overwrite it (2026-09-27)."""

    def _truncate(self, out: str, keep: int) -> None:
        import os
        path = os.path.join(out, "episodes.jsonl")
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()
        with open(path, "w", encoding="utf-8") as fh:
            fh.writelines(lines[:keep])

    def test_resume_finishes_a_dead_run_byte_identical_to_one_go(self):
        import os
        import shutil
        out = "/tmp/episode-gen-resume-test"
        shutil.rmtree(out, ignore_errors=True)
        full = EG.run(n=10, seed=SEED, prose="local", teacher="stub",
                      out_dir=out)
        self.assertEqual(full["n_episodes"], 10)
        expected = EG.load_episodes(out)
        self._truncate(out, 6)                      # the run "dies" here
        self.assertEqual(len(EG.load_episodes(out)), 6)
        again = EG.run(n=10, seed=SEED, prose="local", teacher="stub",
                       out_dir=out, resume=True)
        got = EG.load_episodes(out)
        self.assertEqual(again["n_episodes"], 10)
        self.assertEqual([e["id"] for e in got], [e["id"] for e in expected])
        self.assertEqual(len({e["id"] for e in got}), 10)
        with open(os.path.join(out, "manifest.json")) as fh:
            manifest = json.load(fh)
        self.assertEqual(manifest["resumed_from"], 6)
        self.assertEqual(manifest["n_episodes"], 10)
        self.assertEqual(manifest["n"], 10)

    def test_resume_refuses_another_plan(self):
        import shutil
        out = "/tmp/episode-gen-resume-test-3"
        shutil.rmtree(out, ignore_errors=True)
        EG.run(n=10, seed=SEED, prose="local", teacher="stub", out_dir=out)
        self._truncate(out, 4)
        with self.assertRaises(ValueError):
            EG.run(n=12, seed=SEED, prose="local", teacher="stub",
                   out_dir=out, resume=True)

    def test_without_resume_a_relaunch_starts_over(self):
        import shutil
        out = "/tmp/episode-gen-resume-test-2"
        shutil.rmtree(out, ignore_errors=True)
        EG.run(n=6, seed=SEED, prose="local", teacher="stub", out_dir=out)
        again = EG.run(n=4, seed=SEED, prose="local", teacher="stub",
                       out_dir=out)
        self.assertEqual(again["n_episodes"], 4)
        self.assertEqual(len(EG.load_episodes(out)), 4)
