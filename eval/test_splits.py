"""Unit tests for eval.splits (#T-split-domain)."""
import json
import tempfile
import unittest
from pathlib import Path

from eval.splits import (
    DEFAULT_SEED,
    ROOT,
    GroupKey,
    audit_file,
    build_split,
    detect_language,
    group_digest,
    group_split_of,
    invalidate_artifacts,
    iter_rows,
    normalize_skeleton,
    scan_index_modulo,
    seal_split,
    split_of_digest,
    verify_split,
)

TPL = [
    ("archivar", "Asunto: Factura {emp} nº {num}\n{name}, te adjuntamos la "
                 "factura de {mes} por {imp} €. No requiere acción."),
    ("urgente", "Asunto: Fuga de agua - actúa ya\n{name}, el vecino de abajo "
                "reporta una fuga que viene de tu piso. Ve hoy mismo."),
    ("spam", "Asunto: Herencia de un familiar: {imp}€\nEstimado {name}, un "
             "familiar lejano te dejó {imp}€. Envía tus claves."),
    ("responder", "Asunto: Duda sobre tu reserva {num}\nHola {name}, ¿prefieres "
                  "la mañana o la tarde del {dia}? Confírmanos."),
    ("archivar", "Asunto: Resumen semanal del equipo\nHola {name}, aquí tienes "
                 "el resumen de la semana. Léelo cuando puedas."),
]
NAMES = ["Marta", "Jordi", "Anna", "Pau", "Laia"]
EMPS = ["Endesa", "Iberia", "Renfe"]
MESES = ["enero", "febrero", "marzo"]
DIAS = ["lunes", "martes", "viernes"]


def _corpus(n=400, index_split=True):
    """Template-generated rows, shipped with the old `i % 10` split."""
    rows = []
    for i in range(n):
        state, text = TPL[i % len(TPL)]
        body = text.format(emp=EMPS[i % 3], num=1000 + i, name=NAMES[i % 5],
                           mes=MESES[i % 3], imp=100 + 7 * i, dia=DIAS[i % 3])
        if index_split:
            m = i % 10  # index-split-fixture: reproduces the legacy leak
            split = "train" if m < 8 else ("calibration" if m == 8 else "test")
        else:
            split = group_split_of(body, "fixture")
        rows.append({"state": body, "split": split,
                     "questions": [{"id": f"fixture-{i}", "kind": "choice",
                                    "options": [{"id": state, "text": state}],
                                    "answer": state}]})
    return rows


def _write(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return Path(path)


class TestSkeleton(unittest.TestCase):
    def test_digits_amounts_months_and_names_collapse(self):
        a = ("Asunto: Recibo del parking junio\nAdjuntamos el recibo de junio "
             "por 1631 €. Solo archivo.")
        b = ("Asunto: Recibo del parking abril\nAdjuntamos el recibo de abril "
             "por 25,50 €. Solo archivo.")
        self.assertEqual(normalize_skeleton(a), normalize_skeleton(b))
        self.assertIn("<month>", normalize_skeleton(a))
        self.assertIn("<amount>", normalize_skeleton(a))

    def test_vocative_and_midsentence_names_are_masked(self):
        a = "Jordi, el vecino de abajo reporta una fuga en tu piso."
        b = "Júlia, el vecino de abajo reporta una fuga en tu piso."
        self.assertEqual(normalize_skeleton(a), normalize_skeleton(b))
        self.assertEqual(normalize_skeleton("Viajo con Endesa mañana"),
                         normalize_skeleton("Viajo con Iberia mañana"))

    def test_title_cased_headlines_keep_their_words(self):
        h = "There Were 2 Mass Shootings In Texas Last Week But Only 1 On TV"
        sk = normalize_skeleton(h)
        self.assertIn("texas", sk)
        self.assertNotIn("<name>", sk)
        self.assertIn("<num>", sk)

    def test_urls_emails_dates_and_times(self):
        sk = normalize_skeleton(
            "escribe a foo.bar@mail.com o entra en https://x.example/a el "
            "03/04/2026 a las 23:59")
        for ph in ("<email>", "<url>", "<date>", "<time>"):
            self.assertIn(ph, sk)

    def test_different_templates_do_not_collapse(self):
        self.assertNotEqual(normalize_skeleton(TPL[0][1]),
                            normalize_skeleton(TPL[1][1]))


class TestLanguage(unittest.TestCase):
    def test_locale_tag_wins(self):
        self.assertEqual(detect_language("[it-IT] svegliami alle cinque"), "it")

    def test_stopword_heuristic(self):
        self.assertEqual(detect_language("el informe de la reunión y los datos"),
                         "es")
        self.assertEqual(detect_language("the report of the meeting and it is"),
                         "en")

    def test_unknown_when_no_evidence(self):
        self.assertEqual(detect_language("xyz qqq"), "und")


class TestGroupKey(unittest.TestCase):
    def test_domain_and_language_are_part_of_the_key(self):
        text = "el informe de la reunión y los datos"
        self.assertNotEqual(group_digest(text, "a"), group_digest(text, "b"))
        self.assertNotEqual(group_digest(text, "a", "es"),
                            group_digest(text, "a", "en"))

    def test_digest_is_stable(self):
        k = GroupKey("skeleton", "dom", "es")
        self.assertEqual(k.digest(), GroupKey("skeleton", "dom", "es").digest())
        self.assertEqual(len(k.digest()), 16)


class TestSplitAssignment(unittest.TestCase):
    def test_deterministic_and_index_free(self):
        d = group_digest("Hola Marta, ¿vienes el lunes?", "fixture")
        self.assertEqual(split_of_digest(d), split_of_digest(d))
        self.assertIn(split_of_digest(d), ("train", "test"))

    def test_seed_changes_the_assignment_of_some_groups(self):
        # distinct WORDS, not digits: digits collapse into one skeleton
        digests = [group_digest(f"grupo {chr(97 + i % 26)}{'x' * (i // 26)} "
                                "de la prueba", "fixture") for i in range(80)]
        a = [split_of_digest(d, seed=1) for d in digests]
        b = [split_of_digest(d, seed=99) for d in digests]
        self.assertNotEqual(a, b)
        self.assertIn("test", a)
        self.assertIn("train", a)

    def test_groups_are_never_split_across_sides(self):
        rows = _corpus(400)
        res = build_split(
            ((r["questions"][0]["id"], r["state"], r["split"]) for r in rows),
            "fixture")
        self.assertEqual(res.overlap()["shared_groups"], 0)
        self.assertEqual(res.overlap()["jaccard"], 0.0)
        self.assertEqual(sum(len(v) for v in res.ids.values()), len(rows))

    def test_calibration_frac_is_honoured_at_group_level(self):
        rows = _corpus(400)
        res = build_split(
            ((r["questions"][0]["id"], r["state"], r["split"]) for r in rows),
            "fixture", test_frac=0.2, calib_frac=0.2)
        sides = set(res.ids)
        self.assertTrue(sides <= {"train", "test", "calibration"})
        seen = {}
        for side, ids in res.ids.items():
            for i in ids:
                self.assertNotIn(i, seen)
                seen[i] = side


class TestSealing(unittest.TestCase):
    def _sealed(self, tmp):
        rows = _corpus(200)
        src = _write(rows, Path(tmp) / "fixture.jsonl")
        res = build_split(iter_rows(src, "fixture"), "fixture", name="fixture")
        manifest = seal_split(res, source=src, source_rows=len(rows),
                              root=Path(tmp) / "splits")
        return src, res, manifest

    def test_seal_writes_ids_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, res, manifest = self._sealed(tmp)
            base = Path(tmp) / "splits" / "fixture"
            self.assertTrue((base / "train.ids").exists())
            self.assertTrue((base / "test.ids").exists())
            self.assertEqual(manifest["seed"], DEFAULT_SEED)
            self.assertEqual(manifest["overlap"]["shared_groups"], 0)
            self.assertEqual(manifest["source"]["rows"], 200)
            self.assertEqual(len(manifest["source"]["sha256"]), 64)
            res = verify_split("fixture", root=Path(tmp) / "splits")
            self.assertTrue(res["ok"], res["errors"])

    def test_verify_catches_tampered_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._sealed(tmp)
            base = Path(tmp) / "splits" / "fixture"
            with open(base / "test.ids", "a", encoding="utf-8") as f:
                f.write("fixture-9999\n")
            out = verify_split("fixture", root=Path(tmp) / "splits", strict=False)
            self.assertFalse(out["ok"])
            with self.assertRaises(ValueError):
                verify_split("fixture", root=Path(tmp) / "splits")

    def test_verify_catches_edited_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._sealed(tmp)
            mpath = Path(tmp) / "splits" / "fixture" / "manifest.json"
            data = json.loads(mpath.read_text())
            data["seed"] = 1
            mpath.write_text(json.dumps(data))
            out = verify_split("fixture", root=Path(tmp) / "splits", strict=False)
            self.assertFalse(out["ok"])

    def test_a_sealed_split_is_not_silently_regenerated(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, res, _ = self._sealed(tmp)
            with self.assertRaises(FileExistsError):
                seal_split(res, source=src, root=Path(tmp) / "splits")
            again = seal_split(res, source=src, root=Path(tmp) / "splits",
                               force=True)
            self.assertEqual(again["counts"], res.counts())


class TestAudit(unittest.TestCase):
    def test_index_split_leaks_and_group_split_does_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = _write(_corpus(400, index_split=True),
                         Path(tmp) / "fixture.jsonl")
            rep = audit_file(src, "fixture", max_rows=None)
            self.assertEqual(rep["rows_scanned"], 400)
            self.assertGreater(rep["skeletons"], 1)
            self.assertGreater(rep["largest_group"]["rows"], 1)
            ship = rep["shipped_split"]
            self.assertGreater(ship["shared_groups_train_test"], 0)
            self.assertGreater(ship["jaccard_train_test"], 0.0)
            self.assertEqual(ship["test_row_leak_rate"], 1.0)
            self.assertEqual(rep["group_split"]["shared_groups_train_test"], 0)
            self.assertEqual(rep["group_split"]["jaccard_train_test"], 0.0)

    def test_group_split_corpus_audits_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = _write(_corpus(400, index_split=False),
                         Path(tmp) / "fixture.jsonl")
            rep = audit_file(src, "fixture", max_rows=None)
            self.assertEqual(rep["shipped_split"]["shared_groups_train_test"], 0)
            self.assertEqual(rep["shipped_split"]["jaccard_train_test"], 0.0)
            self.assertIn(rep["shipped_split"]["test_row_leak_rate"], (0.0, None))


class TestIndexModuloScanner(unittest.TestCase):
    def test_flags_an_index_modulo_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "bad.py").write_text(
                "def split_of(i):\n"
                "    m = i % 10\n"
                "    if m < 8:\n"
                "        return 'train'\n"
                "    return 'test'\n")
            found = scan_index_modulo(Path(tmp))
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0].function, "split_of")

    def test_flags_a_ternary_index_modulo_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "bad2.py").write_text(
                "def s(i):\n    return 'test' if i % 10 == 0 else 'train'\n")
            self.assertEqual(len(scan_index_modulo(Path(tmp))), 1)

    def test_hash_derived_group_split_is_not_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "ok.py").write_text(
                "import hashlib\n"
                "def split_of(gid):\n"
                "    h = int(hashlib.sha256(gid.encode()).hexdigest(), 16)\n"
                "    if h % 10 < 8:\n"
                "        return 'train'\n"
                "    return 'test'\n")
            self.assertEqual(scan_index_modulo(Path(tmp)), [])

    def test_pragma_exempts_a_fixture_and_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "fix.py").write_text(
                "def legacy(i):\n"
                "    m = i % 10  # index-split-fixture\n"
                "    return 'train' if m < 8 else 'test'\n")
            exempt = []
            self.assertEqual(scan_index_modulo(Path(tmp), exemptions=exempt), [])
            self.assertEqual(len(exempt), 1)

    def test_a_mention_in_a_comment_is_not_a_violation(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "doc.py").write_text(
                '"""The old split was i % 10 — train/test by row index."""\n'
                "# see also: i % 10\n")
            self.assertEqual(scan_index_modulo(Path(tmp)), [])

    def test_repo_has_no_index_modulo_split(self):
        found = scan_index_modulo(ROOT)
        self.assertEqual(
            found, [],
            "index-modulo split(s): " + "; ".join(
                f"{v.path}:{v.line} {v.function}() {v.snippet}" for v in found))


class TestInvalidation(unittest.TestCase):
    def test_stamps_only_artifacts_built_on_the_index_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.json"
            b = Path(tmp) / "b.json"
            a.write_text(json.dumps({"jobs": [{"task": "synth-loop",
                                               "accuracy": 1.0}]}))
            b.write_text(json.dumps({"jobs": [{"task": "boolq",
                                               "accuracy": 0.67}]}))
            touched = invalidate_artifacts([a, b])
            self.assertEqual(len(touched), 1)
            stamped = json.loads(a.read_text())
            self.assertEqual(stamped["invalidated"]["invalidated_on"],
                             "2026-09-21")
            self.assertIn("synth-loop", stamped["invalidated"]["affects"])
            self.assertNotIn("invalidated", json.loads(b.read_text()))
            self.assertEqual(invalidate_artifacts([a, b]), [])


class TestGeneratorsUseGroupSplits(unittest.TestCase):
    def test_email_triage_never_straddles_a_template(self):
        from data.intent import email_triage as et
        rows = et.generate(300, seed=5)
        sides = {}
        for ex in rows:
            d = group_digest(ex.state, "email-triage")
            sides.setdefault(d, set()).add(ex.split)
        self.assertTrue(all(len(s) == 1 for s in sides.values()))
        self.assertGreater(len(sides), 1)

    def test_huffpost_converter_splits_by_group(self):
        from data.convert_huffpost import split_of
        h = "There Were 2 Mass Shootings In Texas Last Week"
        self.assertEqual(split_of(h), split_of(h))
        self.assertIn(split_of(h), ("train", "test"))


if __name__ == "__main__":
    unittest.main()
