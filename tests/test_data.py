import csv
import gzip
import random
import re
import tempfile
import unittest
from pathlib import Path

from bench.rules import decide
from data import curriculum, off_products
from data.io import read_jsonl, valid_episode


def numbers(text: str) -> dict:
    found = {}
    for label, value, unit in re.findall(
            r"(energy|energía|fat|grasas|saturates|saturadas|sugars|azúcares|"
            r"fibre|fibra|protein|proteínas|salt|sal|sodium|sodio) "
            r"([0-9.]+) (kcal|g|mg)", text):
        found[label] = (float(value), unit)
    return found


class CurriculumTest(unittest.TestCase):

    def test_every_rung_yields_valid_episodes(self):
        for rung in curriculum.RUNGS:
            for row in curriculum.batch(rung, 200, seed=3):
                self.assertTrue(valid_episode(row), row)
                texts = [c["text"] for c in row["candidates"]]
                self.assertEqual(len(set(texts)), len(texts))

    def test_first_rung_only_asks_for_the_most(self):
        for row in curriculum.batch("r0-most", 200, seed=2):
            self.assertRegex(row["question"], r"most|highest|más")

    def test_pair_gold_is_the_extreme_value(self):
        for row in curriculum.batch("r1-pair", 300, seed=5):
            values = [float(re.search(r"([0-9.]+) (?:kcal|g)$",
                                      c["text"]).group(1))
                      for c in row["candidates"]]
            most = any(w in row["question"] for w in ("most", "highest", "más"))
            expected = values.index(max(values) if most else min(values))
            self.assertEqual(int(row["answer"][1:]), expected, row)

    def test_sodium_is_printed_as_salt_times_400(self):
        panel = {n: 1.0 for n in curriculum.NUTRIENTS}
        panel["salt"] = 1.3
        text = curriculum._panel(random.Random(1), "en", panel, sodium=True,
                                 name=None)
        self.assertIn("sodium 520 mg", text)
        self.assertNotIn("salt", text)

    def test_salt_gold_accounts_for_sodium_options(self):
        rows = [row for row in curriculum.batch("r3-units", 400, seed=9)
                if "salt" in row["question"] and "sodium" in " ".join(
                    c["text"] for c in row["candidates"])]
        self.assertTrue(rows)
        for row in rows:
            salts = []
            for candidate in row["candidates"]:
                found = numbers(candidate["text"])
                salts.append(found["salt"][0] if "salt" in found
                             else found["sodium"][0] / 400)
            most = "most" in row["question"] or "highest" in row["question"]
            expected = salts.index(max(salts) if most else min(salts))
            self.assertEqual(int(row["answer"][1:]), expected, row)

    def test_constraint_gold_respects_the_cap(self):
        asked = {v: k for k, v in curriculum.ASKED["en"].items()}
        printed = {k: v.split()[-1] for k, v in curriculum.LABELS["en"].items()}
        rows = [row for row in curriculum.batch("r4-constraints", 300, seed=4)
                if "at most" in row["question"]]
        self.assertTrue(rows)
        for row in rows:
            cap, capped, word, target = re.search(
                r"at most ([0-9.]+) g of (.+?) per 100 g, "
                r"which has the (most|least) (.+)\?$", row["question"]).groups()

            def value(text, nutrient):
                found = numbers(text)
                if nutrient == "salt" and "salt" not in found:
                    return found["sodium"][0] / 400
                return found[printed[nutrient]][0]

            texts = [c["text"] for c in row["candidates"]]
            allowed = [i for i, t in enumerate(texts)
                       if value(t, asked[capped]) <= float(cap)]
            scores = {i: value(texts[i], asked[target]) for i in allowed}
            pick = (max if word == "most" else min)(scores, key=scores.get)
            self.assertEqual(int(row["answer"][1:]), pick, row)

    def test_batches_are_reproducible(self):
        self.assertEqual(curriculum.batch("r4-constraints", 50, seed=8),
                         curriculum.batch("r4-constraints", 50, seed=8))


COLUMNS = ["code", "product_name", "brands", "quantity", "countries_en",
           "main_category_en", "categories_en", "sugars_100g", "salt_100g",
           "fat_100g", "saturated-fat_100g", "proteins_100g", "fiber_100g",
           "energy-kcal_100g", "sodium_100g"]


def fake_dump(path: Path, products: int = 400, seed: int = 0) -> None:
    rng = random.Random(seed)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(COLUMNS)
        for index in range(products):
            salt = round(rng.uniform(0, 3), 2)
            writer.writerow([
                f"{index:013d}", f"Product {index}", f"Brand {index % 7}",
                "200 g", rng.choice(["Spain", "United Kingdom"]),
                f"Category {index % 3}", f"Group {index % 3}",
                round(rng.uniform(0, 40), 1), salt, round(rng.uniform(0, 30), 1),
                round(rng.uniform(0, 10), 1), round(rng.uniform(0, 25), 1),
                round(rng.uniform(0, 8), 1), round(rng.uniform(20, 600)),
                round(salt / 2.5, 4)])


class OffProductsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        root = Path(cls.folder.name)
        fake_dump(root / "dump.csv.gz")
        cls.manifest = off_products.build(root / "dump.csv.gz", root / "out",
                                          per_category=20)
        cls.train = read_jsonl(root / "out" / "train.jsonl")
        cls.dev = read_jsonl(root / "out" / "dev-pool.jsonl")
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_build_writes_valid_decidable_rows(self):
        self.assertGreater(len(self.train), 50)
        for row in self.train + self.dev:
            self.assertTrue(valid_episode(row), row)
            self.assertTrue(row["decidable"])

    def test_gold_follows_from_the_printed_cards(self):
        for row in self.train + self.dev:
            self.assertEqual(decide(row), row["answer"], row)

    def test_no_product_is_on_both_sides(self):
        codes = lambda rows: {code for row in rows
                              for code in row["group_id"][4:].split("+")}
        self.assertFalse(codes(self.train) & codes(self.dev))

    def test_sealed_groups_are_never_written(self):
        self.assertFalse(any(off_products.sealed(row["group_id"])
                             for row in self.dev))

    def test_build_is_byte_reproducible(self):
        again = off_products.build(self.root / "dump.csv.gz",
                                   self.root / "again", per_category=20)
        self.assertEqual(again, self.manifest)
        self.assertEqual((self.root / "again" / "train.jsonl").read_bytes(),
                         (self.root / "out" / "train.jsonl").read_bytes())

    def test_sodium_cards_convert_to_salt(self):
        product = {"brand": "B", "name": "N", "quantity": "", "sodium": 0.4,
                   "values": {"energy": 100, "fat": 1, "saturated_fat": 0.5,
                              "sugars": 2, "fiber": 1, "proteins": 3,
                              "salt": 1.0}}
        self.assertIn("sodium 400 mg", off_products.card(product, "en", True))
        self.assertIn("sal 1 g", off_products.card(product, "es", False))
        self.assertAlmostEqual(off_products.printed(product, "salt", True), 1.0)

    def test_rounding_on_the_cap_marks_a_row_undecidable(self):
        product = {"values": {"fiber": 0.04}, "sodium": 0.0}
        self.assertEqual(off_products.printed(product, "fiber", False), 0.0)
        self.assertNotEqual(off_products.printed(product, "fiber", False) <= 0,
                            product["values"]["fiber"] <= 0)


class BenchmarkFileTest(unittest.TestCase):

    def test_benchmark_rows_are_valid_and_unique(self):
        rows = read_jsonl(Path(__file__).resolve().parents[1] / "benchmark"
                          / "product_specs.dev.jsonl")
        self.assertEqual(len(rows), 3000)
        self.assertEqual(len({row["id"] for row in rows}), 3000)
        self.assertTrue(all(valid_episode(row) for row in rows))
        for row in rows:
            self.assertEqual(decide(row) == row["answer"], row["decidable"])


if __name__ == "__main__":
    unittest.main()
