import io
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from asimovia.data import curriculum
from asimovia.data.io import write_jsonl
from asimovia.metrics import chance, wilson_lower
from asimovia.model.checkpoint import load_model
from asimovia.training import loop
from asimovia.training.train import parser, schedule, shuffled
from asimovia.training.train import train as run_training


def train(args):
    with redirect_stdout(io.StringIO()):
        return run_training(args)


class TrainTest(unittest.TestCase):
    def test_a_tiny_run_lowers_the_loss_and_writes_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_jsonl(root / "train.jsonl", curriculum.batch("r0-most", 64, seed=1))
            write_jsonl(root / "dev.jsonl", curriculum.batch("r0-most", 16, seed=2))
            args = parser().parse_args(
                [
                    "--episodes",
                    str(root / "train.jsonl"),
                    "--eval",
                    str(root / "dev.jsonl"),
                    "--out",
                    str(root / "run"),
                    "--steps",
                    "30",
                    "--batch-size",
                    "8",
                    "--device",
                    "cpu",
                    "--max-length",
                    "256",
                    "--d-model",
                    "32",
                    "--heads",
                    "4",
                    "--layers",
                    "1",
                    "--d-ff",
                    "64",
                    "--pointer-layers",
                    "1",
                    "--lr",
                    "1e-3",
                    "--warmup",
                    "5",
                    "--log-every",
                    "1000",
                ]
            )
            result = train(args)
            self.assertLess(result["last_loss"], result["first_loss"])
            self.assertEqual(result["initialization"], "random")
            self.assertFalse(result["external_checkpoint"])
            for name in ("model.safetensors", "config.json", "train.json"):
                self.assertTrue((root / "run" / name).is_file(), name)
            self.assertEqual(load_model(root / "run").config.d_model, 32)

    def test_a_continued_run_records_its_parent_without_a_local_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_jsonl(root / "train.jsonl", curriculum.batch("r0-most", 32, seed=1))
            write_jsonl(root / "dev.jsonl", curriculum.batch("r0-most", 8, seed=2))
            common = [
                "--episodes",
                str(root / "train.jsonl"),
                "--eval",
                str(root / "dev.jsonl"),
                "--steps",
                "2",
                "--batch-size",
                "8",
                "--device",
                "cpu",
                "--max-length",
                "256",
                "--d-model",
                "32",
                "--heads",
                "4",
                "--layers",
                "1",
                "--d-ff",
                "64",
                "--pointer-layers",
                "1",
                "--warmup",
                "1",
            ]
            train(parser().parse_args([*common, "--out", str(root / "first")]))
            result = train(
                parser().parse_args(
                    [
                        *common,
                        "--out",
                        str(root / "second"),
                        "--init",
                        str(root / "first" / "model.safetensors"),
                    ]
                )
            )
            self.assertEqual(result["initialization"], "continued")
            self.assertEqual(result["continued_from"], "first/model.safetensors")

    def test_shuffling_moves_options_but_keeps_the_answer_text(self):
        row = curriculum.batch("r2-panel", 1, seed=3)[0]
        gold = next(c["text"] for c in row["candidates"] if c["id"] == row["answer"])
        for view in shuffled([row] * 20, random.Random(0)):
            self.assertEqual(
                next(c["text"] for c in view["candidates"] if c["id"] == view["answer"]), gold
            )

    def test_schedule_warms_up_then_decays_to_the_floor(self):
        self.assertAlmostEqual(schedule(0, 1000, 100), 0.01)
        self.assertAlmostEqual(schedule(100, 1000, 100), 1.0)
        self.assertAlmostEqual(schedule(1000, 1000, 100), 0.1)

    def test_wilson_and_chance(self):
        self.assertAlmostEqual(wilson_lower(2861, 3000), 0.9455, places=3)
        rows = [{"candidates": [0] * 4}, {"candidates": [0] * 2}]
        self.assertAlmostEqual(chance(rows), 0.375)


class LoopTest(unittest.TestCase):
    def test_hard_rows_are_the_weak_families(self):
        base = {"candidates": [{"text": "salt 1 g"}] * 3}
        self.assertTrue(
            loop.hard({**base, "question": "Among the products with at most 2 g of fat"})
        )
        self.assertTrue(
            loop.hard({**base, "question": "El contenido de sal no puede pasar de 1 g"})
        )
        self.assertTrue(
            loop.hard(
                {"question": "Which has the most fat?", "candidates": [{"text": "sodium 4 mg"}] * 2}
            )
        )
        self.assertFalse(loop.hard({**base, "question": "¿Qué producto tiene más sal?"}))

    def test_a_rung_clears_at_ninety_percent(self):
        state = {"rung_index": 0}
        result = {
            "holdout_forced_accuracy": 0.91,
            "holdout_forced_chance": 0.5,
            "holdout_forced_wilson_lower": 0.89,
            "holdout_rows": 1000,
            "last_loss": 0.1,
        }
        line = loop.update(state, loop.RUNGS[0], result, 7, 60.0, 9600)
        self.assertTrue(line["cleared"])
        self.assertEqual(state["rung_index"], 1)
        self.assertEqual(state["cleared"]["r0-most"], 7)

    def test_the_real_rung_never_clears(self):
        state = {"rung_index": 5}
        result = {
            "holdout_forced_accuracy": 0.99,
            "holdout_forced_chance": 0.3,
            "holdout_forced_wilson_lower": 0.98,
            "holdout_rows": 3000,
            "last_loss": 0.01,
        }
        self.assertFalse(loop.update(state, loop.RUNGS[5], result, 9, 1.0, 1)["cleared"])
        self.assertEqual(state["rung_index"], 5)

    def test_generated_dev_rows_fit_the_context(self):
        for rung in curriculum.RUNGS:
            self.assertGreaterEqual(loop.fits(curriculum.batch(rung, 200, seed=11)), 0.97, rung)


if __name__ == "__main__":
    unittest.main()
