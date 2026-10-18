"""Tests for the continuous decision-data loop (#T-gen-loop).

What is under test is the thing the killed loop lacked: a stop criterion, a
per-batch gate whose rejections never reach the corpus, and a manifest that
survives a restart.
"""
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.gen_schemas import domain  # noqa: E402
from tools.gen_schemas.generator import build  # noqa: E402
from tools.gen_schemas.loop import (MIN_BATCH_NOVELTY,  # noqa: E402
                                    MIN_BATCH_ROWS, ContinuousLoop,
                                    CorpusIndex, LoopConfig, build_gate,
                                    judge_batch, measure_batch)

QUIET = lambda *a, **k: None  # noqa: E731


def cfg(out: Path, **kw) -> LoopConfig:
    base = dict(out=out, per_cell=2, interval=0.0, firewall=False,
                max_rounds=3, gate_dir=out / "gate", run_id="test-loop")
    base.update(kw)
    return LoopConfig(**base)


def sample_batch(n=120, doms=("it-incident", "logistics", "finance-approval"),
                 langs=("es", "en", "fr", "de"), seed=3):
    """Real schemas, spread across domains/languages/K."""
    rng = random.Random(seed)
    out = []
    i = 0
    while len(out) < n and i < n * 40:
        i += 1
        d = domain(doms[len(out) % len(doms)])
        s, _ = build(d, langs[len(out) % len(langs)], 3 + (len(out) % 5), rng,
                     unknown_rate=0.15)
        if s is not None:
            out.append(s)
    return out


class BatchGateTest(unittest.TestCase):
    def index(self, tmp):
        return CorpusIndex(Path(tmp) / "state")

    def test_balanced_batch_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            kept = sample_batch(180)
            m = measure_batch(kept, len(kept), 0, self.index(tmp))
            ok, checks = judge_batch(m)
            self.assertTrue(ok, checks)
            self.assertEqual(m["new_skeletons"], m["unique_skeletons"])

    def test_single_domain_batch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            kept = sample_batch(150, doms=("logistics",))
            m = measure_batch(kept, len(kept), 0, self.index(tmp))
            ok, checks = judge_batch(m)
            self.assertFalse(ok)
            self.assertFalse(checks["domain_balance"])

    def test_single_language_batch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            kept = sample_batch(150, langs=("es",))
            m = measure_batch(kept, len(kept), 0, self.index(tmp))
            ok, checks = judge_batch(m)
            self.assertFalse(ok)
            self.assertFalse(checks["language_balance"])

    def test_low_novelty_batch_is_rejected(self):
        """Most of the batch died against the corpus -> it re-treads old ground."""
        with tempfile.TemporaryDirectory() as tmp:
            kept = sample_batch(150)
            produced = int(len(kept) / (MIN_BATCH_NOVELTY / 2))
            m = measure_batch(kept, produced, produced - len(kept),
                              self.index(tmp))
            ok, checks = judge_batch(m)
            self.assertFalse(ok)
            self.assertFalse(checks["novelty"])
            self.assertAlmostEqual(m["novelty"], MIN_BATCH_NOVELTY / 2, places=2)

    def test_tiny_batch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            kept = sample_batch(MIN_BATCH_ROWS - 1)
            m = measure_batch(kept, len(kept), 0, self.index(tmp))
            ok, checks = judge_batch(m)
            self.assertFalse(ok)
            self.assertFalse(checks["rows"])

    def test_empty_batch_is_rejected_and_measurable(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = measure_batch([], 40, 40, self.index(tmp))
            ok, checks = judge_batch(m)
            self.assertFalse(ok)
            self.assertEqual(m["novelty"], 0.0)
            self.assertEqual(m["corpus_dedup_rejection_rate"], 1.0)


class RejectedBatchesStayOutTest(unittest.TestCase):
    def test_rejected_rows_never_reach_the_corpus_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            # A one-language, one-domain loop: every batch fails the balance
            # checks, so every batch must land in rejected/, never in corpus/.
            loop = ContinuousLoop(cfg(out, domains=("logistics",),
                                      languages=("es",), per_cell=8,
                                      max_rounds=2), log=QUIET)
            loop.run()
            self.assertEqual(loop.state.batches_accepted, 0)
            self.assertTrue(loop.state.batches_rejected >= 1)
            self.assertFalse(list((out / "corpus").glob("*.jsonl")))
            self.assertTrue(list((out / "rejected").glob("*.jsonl")))
            self.assertEqual(loop.state.rows_accepted, 0)
            self.assertTrue(loop.state.rows_rejected > 0)
            # Nothing a rejected batch produced is indexed as corpus pressure.
            self.assertFalse(loop.index.skeletons)


class StopCriterionTest(unittest.TestCase):
    def test_warmup_has_no_rate_and_a_low_floor_stops_the_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            loop = ContinuousLoop(cfg(out, max_rounds=0, rate_window=2,
                                      min_rate=1e18), log=QUIET)
            self.assertIsNone(loop.marginal_rate())
            loop.run()
            self.assertIn("diversity gate", loop.state.stopped)
            self.assertIn("marginal diversity", loop.state.stopped)
            # It STOPPED — it did not keep spinning past the window.
            self.assertEqual(len(loop.state.rounds), 2)

    def test_consecutive_rejections_stop_the_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            loop = ContinuousLoop(cfg(out, domains=("logistics",),
                                      languages=("es",), per_cell=8,
                                      max_rounds=0, min_rate=0.0), log=QUIET)
            loop.run()
            self.assertIn("consecutive", loop.state.stopped)
            self.assertEqual(loop.state.batches_accepted, 0)

    def test_loop_always_terminates_with_a_recorded_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop = ContinuousLoop(cfg(Path(tmp), max_rounds=2), log=QUIET)
            loop.run()
            self.assertTrue(loop.state.stopped)


class ResumeTest(unittest.TestCase):
    def test_restart_rebuilds_the_index_and_does_not_regenerate(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            first = ContinuousLoop(cfg(out, max_rounds=2), log=QUIET)
            first.run()
            rows1, skel1 = first.state.rows_accepted, len(first.index.skeletons)
            rounds1 = len(first.state.rounds)
            self.assertTrue(rows1 > 0)

            second = ContinuousLoop(cfg(out, max_rounds=4), log=QUIET)
            found = second.resume()
            self.assertTrue(found["manifest"])
            self.assertEqual(found["norm_rows"], rows1)
            self.assertEqual(len(second.index.skeletons), skel1)
            self.assertEqual(second.state.next_round, rounds1 + 1)

            second.run()
            self.assertEqual(second.state.rounds[rounds1]["round"], rounds1 + 1)
            self.assertTrue(second.state.rows_accepted > rows1)
            # Every skeleton in the corpus is still unique after the restart:
            # the rebuilt index kept the old rows as duplicate pressure.
            skeletons = []
            for shard in sorted((out / "corpus").glob("batch-*.jsonl")):
                for line in shard.read_text().splitlines():
                    if line.strip():
                        skeletons.append(json.loads(line)["gen"]["skeleton"])
            self.assertEqual(len(skeletons), len(set(skeletons)))
            self.assertEqual(len(skeletons), second.state.rows_accepted)

    def test_manifest_is_valid_json_with_criteria_and_rounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            loop = ContinuousLoop(cfg(out, max_rounds=2), log=QUIET)
            loop.run()
            doc = json.loads((out / "run.json").read_text())
            self.assertEqual(doc["task"], "T-gen-loop")
            self.assertEqual(len(doc["rounds"]), 2)
            self.assertIn("min_new_skeletons_per_hour", doc["criteria"]["stop"])
            self.assertIn("min_novelty", doc["criteria"]["batch"])
            self.assertTrue(doc["stopped"])


class GateFileTest(unittest.TestCase):
    def test_gate_json_carries_the_stop_criterion_and_the_curve(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            loop = ContinuousLoop(cfg(out, max_rounds=3), log=QUIET)
            gate = loop.run()
            on_disk = json.loads((out / "gate" / "gate.json").read_text())
            self.assertEqual(on_disk["pass"], gate["pass"])
            self.assertEqual(on_disk["task"], "T-gen-loop")
            self.assertTrue(on_disk["stop_criterion"]["stopped"])
            self.assertEqual(len(on_disk["diversity_curve"]), 3)
            self.assertEqual(len(on_disk["batch_gate_rounds"]), 3)
            acc = on_disk["rows_accepted_vs_rejected"]
            self.assertEqual(acc["accepted"] + acc["rejected"],
                             sum(r["kept"] for r in on_disk["batch_gate_rounds"]))
            self.assertEqual(on_disk["corpus_rows_on_disk"], acc["accepted"])

    def test_corpus_holds_one_row_per_skeleton(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            loop = ContinuousLoop(cfg(out, max_rounds=3), log=QUIET)
            gate = loop.run()
            self.assertEqual(gate["totals"]["rows_per_skeleton"], 1.0)
            self.assertTrue(gate["checks"]["corpus_beats_the_killed_loop"])
            self.assertTrue(gate["checks"]["rejected_batches_are_out_of_corpus"])

    def test_gate_passes_mid_flight_while_still_above_the_floor(self):
        """A healthy running loop is not a failed one."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            loop = ContinuousLoop(cfg(out, max_rounds=5, rate_window=2),
                                  log=QUIET)
            loop.run()
            loop.state.stopped = ""  # as the gate sees it between rounds
            mid = build_gate(loop)
            self.assertEqual(mid["state"], "running")
            self.assertTrue(mid["stop_criterion"]["above_floor"])
            self.assertTrue(mid["checks"]["self_limiting"])
            self.assertTrue(mid["pass"], mid["checks"])
            # ...and the same loop with a floor it cannot meet does not pass.
            loop.cfg.min_rate = 1e18
            low = build_gate(loop)
            self.assertFalse(low["checks"]["self_limiting"])
            self.assertFalse(low["pass"])

    def test_an_interrupted_run_is_still_self_limiting(self):
        """SIGTERM from an operator restart is not a failure to self-limit."""
        with tempfile.TemporaryDirectory() as tmp:
            loop = ContinuousLoop(cfg(Path(tmp), max_rounds=4, rate_window=2),
                                  log=QUIET)
            loop.run()
            loop.state.stopped = "signal SIGTERM"
            gate = build_gate(loop)
            self.assertTrue(gate["checks"]["self_limiting"])
            self.assertTrue(gate["pass"], gate["checks"])
            self.assertEqual(gate["stop_criterion"]["stopped"], "signal SIGTERM")

    def test_gate_records_the_novelty_decay(self):
        with tempfile.TemporaryDirectory() as tmp:
            loop = ContinuousLoop(cfg(Path(tmp), per_cell=4, max_rounds=6),
                                  log=QUIET)
            gate = loop.run()
            self.assertTrue(gate["novelty"]["decaying"])
            self.assertEqual(gate["novelty"]["first_round"], 1.0)
            self.assertLess(gate["novelty"]["last_round"],
                            gate["novelty"]["first_round"])

    def test_novelty_decays_as_the_corpus_grows(self):
        """The curve the old loop never published: information per batch falls."""
        with tempfile.TemporaryDirectory() as tmp:
            loop = ContinuousLoop(cfg(Path(tmp), per_cell=4, max_rounds=8),
                                  log=QUIET)
            loop.run()
            curve = [r["measurement"]["novelty"] for r in loop.state.rounds]
            self.assertEqual(curve[0], 1.0)
            self.assertLess(curve[-1], curve[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
