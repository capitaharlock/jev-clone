"""Tests del generador por regla con variedad (#T-numeric-gen).

Sin GPU, sin red, sin Qwen: todo lo que este fichero comprueba es
aritmética entera y prosa de plantilla. Lo que exige, uno por uno, es lo
que la task pide antes de publicar un lote:

* el gold sale de la regla y la regla se recomputa desde `rule_trace`;
* los contrafactuales lo son: un número o el criterio (o el ORDEN de los
  criterios) mueven el gold, una paráfrasis no;
* la posición del gold es uniforme por la χ² escrita antes de medir;
* el vocabulario de entidades no aparece ni en dev ni en el sellado;
* la variedad realizada llega a los 12 atributos, las 7 formas de
  pregunta, K en {2, 3, 8} y las dos lenguas;
* el lote se publica con semilla = número de lote y es reproducible.

Run (UN directorio cada vez — colisión de basenames en este repo):
    PYTHONPATH=. pytest data/test_rule_variety.py -q
"""

from __future__ import annotations

import json
import os
import random
import shutil
import unittest

from . import episode_contract as EC
from . import episode_gen as EG
from . import rule_variety as RV

TMP = "/tmp/rule-variety-test"
BATCH = 7


def _batch(n: int = 4000, variety: int = 16, out: str = "",
           seed: int = 1) -> list[dict]:
    """Un lote por regla, en local: el mismo camino que el lote de 50 000."""
    where = out or f"{TMP}/n{n}-v{variety}-s{seed}"
    shutil.rmtree(where, ignore_errors=True)
    EG.run(n=n, seed=seed, prose="local", teacher="stub", out_dir=where,
           families=list(RV.FAMILIES), variety=variety,
           group_prefix=f"rule{seed:04d}-")
    return EG.load_episodes(where)


class _Batch(unittest.TestCase):
    """Un lote compartido: generarlo cuesta menos de un segundo, pero no
    hace falta uno por test."""

    episodes: list[dict]

    @classmethod
    def setUpClass(cls):
        cls.episodes = _batch()


# -- la regla del gold ----------------------------------------------------
class RuleGoldTest(unittest.TestCase):
    def test_single_criterion_min_and_max(self):
        attrs = [{"price_eur": 4200, "rating_pt": 70},
                 {"price_eur": 2500, "rating_pt": 95}]
        self.assertEqual(RV.rule_gold(attrs, [("price_eur", "min")]), 1)
        self.assertEqual(RV.rule_gold(attrs, [("price_eur", "max")]), 0)
        self.assertEqual(RV.rule_gold(attrs, [("rating_pt", "max")]), 1)

    def test_the_chain_only_breaks_ties(self):
        attrs = [{"duration_min": 120, "price_eur": 9000},
                 {"duration_min": 120, "price_eur": 5000},
                 {"duration_min": 180, "price_eur": 100}]
        chain = [("duration_min", "min"), ("price_eur", "min")]
        self.assertEqual(RV.rule_gold(attrs, chain), 1)
        # Sin el desempate, el orden lo decide el índice; con él, el precio.
        self.assertEqual(RV.rule_gold(attrs, [("duration_min", "min")]), 0)

    def test_three_criteria_chain(self):
        attrs = [{"a": 1, "b": 1, "c": 3}, {"a": 1, "b": 1, "c": 2},
                 {"a": 1, "b": 2, "c": 1}]
        RV.ATTR_BY_KEY.setdefault("a", {"step": 1})
        chain = [("a", "min"), ("b", "min"), ("c", "min")]
        self.assertEqual(RV.rule_gold(attrs, chain), 1)

    def test_the_threshold_discards_before_ordering(self):
        attrs = [{"duration_min": 30, "price_eur": 40000},
                 {"duration_min": 90, "price_eur": 1000}]
        crit = [("duration_min", "min")]
        self.assertEqual(RV.rule_gold(attrs, crit), 0)
        self.assertEqual(
            RV.rule_gold(attrs, crit, [("price_eur", "le", 20000)]), 1)

    def test_no_candidate_passes_is_an_error_not_a_guess(self):
        attrs = [{"price_eur": 40000}, {"price_eur": 50000}]
        with self.assertRaises(ValueError):
            RV.rule_gold(attrs, [("price_eur", "min")],
                         [("price_eur", "le", 100)])

    def test_rank_is_the_nth_best(self):
        attrs = [{"p": 5}, {"p": 1}, {"p": 3}]
        crit = [("p", "min")]
        self.assertEqual(RV.rule_gold(attrs, crit, rank=1), 1)
        self.assertEqual(RV.rule_gold(attrs, crit, rank=2), 2)
        self.assertEqual(RV.rule_gold(attrs, crit, rank=3), 0)

    def test_an_unknown_direction_is_refused(self):
        with self.assertRaises(ValueError):
            RV.rule_gold([{"p": 1}, {"p": 2}], [("p", "lowest")])


# -- formatos: cosméticos, nunca aritméticos ------------------------------
class NumberFormatTest(unittest.TestCase):
    def test_the_same_integer_in_four_money_formats(self):
        attr = RV.ATTR_BY_KEY["price_eur"]
        got = {style: RV.render_value(attr, 120000, "es", style)
               for style in RV.STYLES["money"]}
        self.assertEqual(got["plain"], "1200 €")
        self.assertEqual(got["group"], "1.200 €")
        self.assertEqual(got["thousand"], "1,2 mil euros")
        self.assertEqual(len(set(got.values())), 3)  # decimal == plain aquí

    def test_english_groups_with_a_comma_spanish_with_a_dot(self):
        attr = RV.ATTR_BY_KEY["price_eur"]
        self.assertEqual(RV.render_value(attr, 120000, "en", "group"),
                         "1,200 euros")
        self.assertEqual(RV.render_value(attr, 120050, "es", "decimal"),
                         "1.200,50 €")

    def test_words_for_the_times_that_have_words(self):
        attr = RV.ATTR_BY_KEY["duration_min"]
        self.assertEqual(RV.render_value(attr, 135, "es", "words"),
                         "dos horas y cuarto")
        self.assertEqual(RV.render_value(attr, 90, "en", "words"),
                         "an hour and a half")
        self.assertEqual(RV.render_value(attr, 180, "es", "plain_h"),
                         "3 horas")
        self.assertEqual(RV.render_value(attr, 185, "es", "h_min"),
                         "3 h 5 min")

    def test_a_style_that_does_not_apply_falls_back_it_never_invents(self):
        attr = RV.ATTR_BY_KEY["duration_min"]
        # 137 no está en la tabla de palabras ni es múltiplo de 60.
        self.assertEqual(RV.render_value(attr, 137, "es", "words"),
                         "137 minutos")
        self.assertEqual(RV.render_value(attr, 137, "es", "plain_h"),
                         "137 minutos")

    def test_the_format_cannot_change_the_gold(self):
        rng = random.Random(4)
        spec = RV.base_case(EC.COMPARISON, "es", 3, rng, 16)
        before = spec["gold"]
        for key in spec["styles"]:
            kind = RV.ATTR_BY_KEY[key]["kind"]
            spec["styles"][key] = RV.STYLES[kind][-1]
        self.assertEqual(RV.rule_gold(spec["attrs"], spec["criteria"],
                                      spec["filters"], spec["rank"]), before)


# -- el caso base ---------------------------------------------------------
class BaseCaseTest(unittest.TestCase):
    def test_values_are_distinct_per_attribute_except_a_declared_tie(self):
        for seed in range(60):
            rng = random.Random(seed)
            for k in RV.KS:
                spec = RV.base_case(EC.COMPARISON, "en", k, rng, 16)
                for key in spec["keys"]:
                    col = [a[key] for a in spec["attrs"]]
                    dupes = len(col) - len(set(col))
                    tie = spec["form"] in ("tie",) \
                        and key == spec["criteria"][0][0]
                    self.assertLessEqual(dupes, 1 if tie else 0,
                                         f"{spec['form']}/{key}: {col}")

    def test_the_state_always_carries_at_least_two_attributes(self):
        # La variante `rule` de comparación cambia el criterio: sin un
        # segundo atributo en el estado no habría a qué cambiarlo.
        for seed in range(40):
            for family in RV.FAMILIES:
                spec = RV.base_case(family, "es", 3, random.Random(seed), 16)
                self.assertGreaterEqual(len(spec["keys"]), 2)

    def test_deterministic_in_the_seeded_rng(self):
        a = RV.base_case(EC.PRIORITY, "es", 8, random.Random(11), 16)
        b = RV.base_case(EC.PRIORITY, "es", 8, random.Random(11), 16)
        self.assertEqual(a, b)

    def test_variety_is_a_dial_not_a_yes_or_no(self):
        low = {RV.base_case(EC.COMPARISON, "en", 2,
                            random.Random(s), 2)["criteria"][0][0]
               for s in range(40)}
        high = {RV.base_case(EC.COMPARISON, "en", 2,
                             random.Random(s), 16)["criteria"][0][0]
                for s in range(40)}
        self.assertLessEqual(len(low), 2)
        self.assertGreater(len(high), len(low))

    def test_no_extreme_of_the_grid_is_ever_sampled(self):
        # Sin margen, el contrafactual `number` se queda sin sitio: 10 de
        # 192 000 casos fallaban (medido el 2026-09-27).
        for seed in range(40):
            for k in RV.KS:
                spec = RV.base_case(EC.PRIORITY, "en", k,
                                    random.Random(seed), 16)
                for key in spec["keys"]:
                    attr = RV.ATTR_BY_KEY[key]
                    grid = list(range(attr["lo"], attr["hi"] + 1,
                                      attr["step"]))
                    if len(grid) < k + 2 * RV.GRID_MARGIN:
                        continue
                    for a in spec["attrs"]:
                        self.assertGreaterEqual(a[key],
                                                grid[RV.GRID_MARGIN], key)
                        self.assertLessEqual(
                            a[key], grid[-1 - RV.GRID_MARGIN], key)

    def test_every_case_of_every_cell_has_its_four_variants(self):
        # La barrida que encontró los tres agujeros de rejilla: todas las
        # celdas, todos los K, todas las variantes, sin una excepción.
        built = 0
        for family in RV.FAMILIES:
            for lang in EC.LANGS:
                for k in RV.KS:
                    for case in range(30):
                        group = f"rule0001-{family[:4]}-{lang}-c{case:04d}"
                        for kind in RV.VARIANTS:
                            base_rng, var_rng = RV.case_rngs(1, group, kind)
                            base = RV.base_case(family, lang, k, base_rng, 16)
                            RV.apply_variant(base, kind, var_rng)
                            built += 1
        self.assertEqual(built, 2 * 2 * 3 * 30 * 4)

    def test_a_variety_too_small_for_a_form_is_an_error_not_a_bad_episode(self):
        # `--variety 2` no tiene tres atributos que enseñar, así que la
        # forma de umbral no existe; lo que NO pasa es publicarla mal.
        forms = {RV.base_case(EC.COMPARISON, "es", 3,
                              random.Random(s), 2)["form"]
                 for s in range(40)}
        self.assertNotIn("threshold", forms)
        self.assertTrue(forms <= {"superlative", "ordinal", "tie"}, forms)

    def test_the_five_non_rule_families_are_refused(self):
        with self.assertRaises(ValueError):
            RV.base_case(EC.EXTRACTION, "es", 2, random.Random(0), 16)


# -- las variantes --------------------------------------------------------
class VariantTest(unittest.TestCase):
    def _specs(self, family: str, kind: str, k: int = 3, n: int = 40):
        out = []
        for seed in range(n):
            base = RV.base_case(family, "es", k, random.Random(seed), 16)
            out.append((base, RV.apply_variant(base, kind,
                                               random.Random(seed + 500))))
        return out

    def test_one_number_moves_the_gold(self):
        for family in RV.FAMILIES:
            for k in RV.KS:
                for base, var in self._specs(family, "number", k, 25):
                    self.assertNotEqual(var["gold"], base["gold"])
                    changed = [(i, key)
                               for i, (a, b) in enumerate(
                                   zip(base["attrs"], var["attrs"]))
                               for key in a if a[key] != b[key]]
                    self.assertEqual(len(changed), 1, changed)

    def test_the_criterion_moves_the_gold_in_comparison(self):
        for k in RV.KS:
            for base, var in self._specs(EC.COMPARISON, "rule", k, 25):
                self.assertNotEqual(var["gold"], base["gold"])

    def test_the_order_of_the_criteria_moves_the_gold_in_priority(self):
        for k in RV.KS:
            for base, var in self._specs(EC.PRIORITY, "rule", k, 25):
                self.assertNotEqual(var["gold"], base["gold"])
                self.assertEqual(sorted(map(tuple, var["criteria"])),
                                 sorted(map(tuple, base["criteria"])))

    def test_a_paraphrase_changes_the_prose_and_not_the_gold(self):
        for family in RV.FAMILIES:
            for base, var in self._specs(family, "paraphrase", 3, 25):
                self.assertEqual(var["gold"], base["gold"])
                self.assertEqual(var["attrs"], base["attrs"])
                self.assertEqual(list(var["criteria"]), list(base["criteria"]))
                self.assertNotEqual(RV.state_of(var), RV.state_of(base))

    def test_an_unknown_variant_is_refused(self):
        base = RV.base_case(EC.COMPARISON, "es", 2, random.Random(0), 16)
        with self.assertRaises(ValueError):
            RV.apply_variant(base, "shuffle", random.Random(0))

    def test_the_four_variants_of_a_group_share_the_base_case(self):
        rngs = [RV.case_rngs(1, "rule0001-attr-es-c0003", v)
                for v in RV.VARIANTS]
        bases = [RV.base_case(EC.COMPARISON, "es", 3, r, 16) for r, _ in rngs]
        for spec in bases[1:]:
            self.assertEqual(spec["attrs"], bases[0]["attrs"])
            self.assertEqual(spec["names"], bases[0]["names"])


# -- el episodio publicado ------------------------------------------------
class PublishedEpisodeTest(_Batch):
    def test_every_episode_passes_the_contract(self):
        per = EC.batch_validate(self.episodes)
        self.assertEqual(per["n_invalid"], 0,
                         [r for r in per["per_episode"] if r][:3])
        self.assertEqual(per["duplicate_ids"], [])

    def test_the_gold_recomputes_from_the_published_rule_trace(self):
        for ep in self.episodes:
            rt = ep["rule_trace"]
            ids = [c["id"] for c in ep["candidates"]]
            self.assertEqual(ids[RV.trace_gold(rt)], ep["answer"], ep["id"])

    def test_the_rule_trace_survives_a_json_round_trip(self):
        for ep in self.episodes[:200]:
            again = json.loads(json.dumps(ep, ensure_ascii=False))
            self.assertEqual(RV.trace_gold(again["rule_trace"]),
                             RV.trace_gold(ep["rule_trace"]))

    def test_no_labelgen_marker_no_dataset_id_question(self):
        for ep in self.episodes:
            for marker in EG.LABELGEN_MARKERS:
                self.assertNotIn(marker, ep["state"])
            self.assertFalse(EC.is_dataset_id_question(ep["question"]),
                             ep["question"])

    def test_the_evidence_is_the_gold_clause_and_lives_in_the_state(self):
        for ep in self.episodes:
            self.assertTrue(
                EC.evidence_is_fragment(ep["evidence"], ep["state"]), ep["id"])
            gold = RV.gold_position(ep)
            self.assertIn(ep["rule_trace"]["entities"][gold], ep["evidence"])

    def test_the_state_fits_the_scorer_window(self):
        # El scorer tiene 512 tokens y se NIEGA a recortar el estado. Con
        # K = 8 y tres atributos, 419 de 17 328 pares del primer lote
        # pasaban de 512 (máximo 570, medido el 2026-09-27); con la cota
        # de `MAX_STATE_ATTRS` el máximo bajó a 485 y ninguno se recorta.
        self.assertEqual(RV.MAX_STATE_ATTRS[8], 2)
        for ep in self.episodes:
            self.assertLessEqual(len(ep["state"]), RV.MAX_STATE_CHARS,
                                 ep["id"])
            shown = len(ep["rule_trace"]["keys"])
            budget = RV.MAX_STATE_ATTRS[len(ep["candidates"])]
            self.assertLessEqual(shown, budget, ep["id"])

    def test_every_form_still_appears_somewhere(self):
        forms = {f"{ep['family']}/{ep['rule_trace']['form']}"
                 for ep in self.episodes}
        self.assertEqual(len(forms), 7, sorted(forms))

    def test_reproducible_by_seed(self):
        again = _batch(n=400, out=f"{TMP}/repro-a")
        twice = _batch(n=400, out=f"{TMP}/repro-b")
        self.assertEqual(again, twice)

    def test_a_different_batch_number_is_a_different_corpus(self):
        one = _batch(n=400, out=f"{TMP}/b1", seed=1)
        two = _batch(n=400, out=f"{TMP}/b2", seed=2)
        self.assertNotEqual([e["state"] for e in one],
                            [e["state"] for e in two])
        self.assertEqual({e["variant_group"][:9] for e in one}, {"rule0001-"})
        self.assertEqual({e["variant_group"][:9] for e in two}, {"rule0002-"})


# -- los cuatro informes del gate -----------------------------------------
class PositionChi2Test(_Batch):
    def test_the_rule_is_written_before_the_measurement(self):
        self.assertEqual(RV.POSITION_CHI2["written_utc"], "2026-09-27")
        self.assertEqual(RV.POSITION_CHI2["alpha"], 0.01)
        self.assertEqual(RV.POSITION_CHI2["critical_values"],
                         {"1": 6.635, "2": 9.210, "7": 18.475})

    def test_the_gold_position_is_uniform_at_every_k(self):
        got = RV.position_chi2(self.episodes)
        self.assertTrue(got["pass"], got["by_k"])
        self.assertEqual(sorted(int(k) for k in got["by_k"]), list(RV.KS))
        for k, node in got["by_k"].items():
            self.assertLessEqual(node["statistic"], node["critical_0.01"],
                                 f"K={k}: {node}")

    def test_a_cell_with_too_few_rows_declares_itself_underpowered(self):
        got = RV.position_chi2(self.episodes[:20])
        self.assertFalse(got["pass"])
        self.assertTrue(all(node["status"] == "underpowered"
                            for node in got["by_k"].values()))

    def test_a_skewed_corpus_fails_the_test(self):
        skewed = [dict(ep, answer=ep["candidates"][0]["id"])
                  for ep in self.episodes]
        self.assertFalse(RV.position_chi2(skewed)["pass"])


class CounterfactualReportTest(_Batch):
    def test_every_group_has_its_four_variants_and_they_behave(self):
        got = RV.counterfactual_report(self.episodes)
        self.assertTrue(got["pass"], got["failures"])
        self.assertTrue(got["gold_recomputes_from_rule"]["pass"])
        for kind in ("number", "rule", "paraphrase"):
            node = got["by_variant"][kind]
            self.assertEqual(node["n"], node["ok"])
            self.assertGreater(node["n"], 0)

    def test_a_tampered_gold_is_caught_not_averaged_away(self):
        bad = [dict(ep) for ep in self.episodes]
        victim = next(i for i, ep in enumerate(bad)
                      if ep["rule_trace"]["cf_kind"] == "paraphrase")
        ids = [c["id"] for c in bad[victim]["candidates"]]
        other = next(i for i in ids if i != bad[victim]["answer"])
        bad[victim] = dict(bad[victim], answer=other)
        got = RV.counterfactual_report(bad)
        self.assertFalse(got["pass"])
        self.assertIn(bad[victim]["id"],
                      got["gold_recomputes_from_rule"]["mismatched"])

    def test_perturbing_the_criterion_number_flips_every_published_gold(self):
        rows = [ep["rule_trace"] for ep in self.episodes]
        flipped = sum(RV.perturb_flips(rt) for rt in rows)
        self.assertEqual(flipped, len(rows))


class LeakageTest(_Batch):
    def test_the_entity_vocabulary_is_absent_from_both_batteries(self):
        got = RV.leakage_report(self.episodes, sample_n=40)
        self.assertTrue(got["pass"], got["layers"])
        for name in ("battery_dev", "battery_sealed"):
            layer = got["layers"][name]
            self.assertTrue(layer["present"], name)
            self.assertEqual(layer["vocabulary_hits"], [])
            self.assertEqual(layer["n_paraphrase_hits"], 0)

    def test_the_detector_would_catch_a_copy_of_a_dev_row(self):
        stolen = RV._battery_texts(RV.DEV_BATTERY)[0]
        planted = [dict(self.episodes[0], state=stolen, question=stolen)]
        got = RV.leakage_report(planted, sample_n=4)
        self.assertFalse(got["pass"])
        self.assertTrue(got["layers"]["battery_dev"]["paraphrase_hits"])

    def test_the_vocabulary_is_the_one_the_episodes_use(self):
        vocab = set(RV.vocabulary())
        for ep in self.episodes[:100]:
            noun, name = ep["rule_trace"]["entities"][0].rsplit(" ", 1)
            self.assertIn(noun.lower(), vocab)
            self.assertIn(name.lower(), vocab)


class VarietyReportTest(_Batch):
    def test_the_realised_variety_meets_what_the_task_asked(self):
        got = RV.variety_report(self.episodes)
        self.assertTrue(got["pass"], got)
        self.assertGreaterEqual(got["n_attributes"], RV.MIN_VARIETY)
        self.assertEqual(got["ks"], list(RV.KS))
        self.assertEqual(got["langs"], sorted(EC.LANGS))
        self.assertEqual(got["n_question_forms"], 7)

    def test_twelve_is_enough_for_twelve_attributes(self):
        got = RV.variety_report(_batch(n=4000, variety=12,
                                       out=f"{TMP}/v12"))
        self.assertGreaterEqual(got["n_attributes"], 12)
        self.assertTrue(got["pass"], got)

    def test_number_formats_and_entity_nouns_are_really_mixed(self):
        got = RV.variety_report(self.episodes)
        self.assertGreaterEqual(got["n_number_styles"], 20)
        self.assertGreaterEqual(got["n_entity_nouns"], 12)


# -- el plan declarado ----------------------------------------------------
class DeclaredPlanTest(unittest.TestCase):
    def test_families_filters_the_plan_and_keeps_the_es_en_proportion(self):
        plan = EG.declared_split(4000, 1, list(RV.FAMILIES))
        self.assertEqual(len(plan), 4)
        self.assertEqual(sum(plan.values()), 4000)
        for lang in EC.LANGS:
            share = sum(v for (f, ln), v in plan.items() if ln == lang)
            self.assertEqual(share, 2000)
        self.assertEqual({f for f, _ in plan}, set(RV.FAMILIES))

    def test_an_unknown_family_is_refused_not_silently_dropped(self):
        with self.assertRaises(ValueError):
            EG.declared_split(10, 1, ["attribute_comparison", "vibes"])

    def test_k_is_declared_per_case_and_balanced(self):
        ks = EG.declared_ks(600, 1)
        self.assertEqual(len(ks), 600)
        for k in RV.KS:
            self.assertEqual(ks.count(k), 200)
        self.assertEqual(ks, EG.declared_ks(600, 1))
        self.assertNotEqual(ks, EG.declared_ks(600, 2))

    def test_every_variant_of_a_group_shares_its_k(self):
        items = EG._work_items([(EC.COMPARISON, "es")] * 12, 12,
                               RV.VARIANTS, EG.declared_ks(4, 1), "rule0001-")
        by_group: dict = {}
        for _idx, _f, _ln, group, _v, k in items:
            by_group.setdefault(group, set()).add(k)
        self.assertTrue(all(len(v) == 1 for v in by_group.values()), by_group)
        self.assertEqual(len(by_group), 3)


# -- publicar el lote ----------------------------------------------------
class PublishTest(unittest.TestCase):
    def test_the_batch_seed_is_the_batch_number_and_the_manifest_rebuilds(self):
        out = f"{TMP}/publish"
        shutil.rmtree(out, ignore_errors=True)
        got = RV.publish(BATCH, 800, variety=16, out_dir=out, sample_n=20)
        with open(os.path.join(out, "manifest.json"), encoding="utf-8") as fh:
            manifest = json.load(fh)
        self.assertEqual(manifest["seed"], BATCH)
        self.assertEqual(manifest["batch"], BATCH)
        self.assertIs(manifest["seed_is_batch_number"], True)
        self.assertIs(manifest["episodes_versioned"], False)
        self.assertEqual(len(manifest["episodes_sha256"]), 64)
        self.assertIn("--batch 7", manifest["rebuild"])
        self.assertEqual(manifest["families"], list(RV.FAMILIES))
        self.assertEqual(manifest["status"], "published")
        self.assertEqual(got["n_episodes"], 800)
        self.assertEqual(got["n_rejects"], 0)

    def test_the_manifest_carries_the_four_measured_reports(self):
        out = f"{TMP}/publish2"
        shutil.rmtree(out, ignore_errors=True)
        RV.publish(BATCH, 1200, variety=16, out_dir=out, sample_n=20)
        with open(os.path.join(out, "manifest.json"), encoding="utf-8") as fh:
            manifest = json.load(fh)
        for key in ("variety_realised", "gold_position_chi2",
                    "counterfactuals", "leakage"):
            self.assertIn("pass", manifest[key], key)
        # 1 200 episodios / 4 variantes = 300 casos + 1 de holgura, así
        # que un K se lleva uno más: el reparto es balanceado, no exacto.
        self.assertEqual(sorted(manifest["planned_ks"]), ["2", "3", "8"])
        self.assertEqual(sum(manifest["planned_ks"].values()), 301)
        self.assertEqual(max(manifest["planned_ks"].values())
                         - min(manifest["planned_ks"].values()), 1)

    def test_batch_zero_is_refused_the_seed_would_be_zero(self):
        with self.assertRaises(ValueError):
            RV.publish(0, 10, out_dir=f"{TMP}/nope")

    def test_the_batch_directory_is_named_after_its_number(self):
        self.assertTrue(RV.batch_dir(1).endswith("episodes-rule/batch-0001"))
        self.assertTrue(RV.batch_dir(42).endswith("episodes-rule/batch-0042"))


if __name__ == "__main__":
    unittest.main()
