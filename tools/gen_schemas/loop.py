"""Continuous, self-limiting decision-schema loop (#T-gen-loop).

#T-gen-schemas made the *unit* right (a complete decision schema) and bounded
one run with a quota. This is the loop that keeps running on that contract —
and, unlike the loop it replaces, it knows when to stop.

The failure mode being prevented is exactly the audited one: the old loop ran
11 h, wrote 258 700 rows and carried 190 skeletons — about **17 new skeletons
per hour**. It never stopped because nothing measured whether it was still
adding information. Three mechanisms fix that:

1. **Per-batch gate.** Every round generates a batch, the batch is scored
   against the corpus that already exists (cross-batch Jaccard novelty,
   domain/language/K coverage, distractor hardness, `unknown` rate), and a
   batch that would degrade the corpus is REJECTED WHOLE. Rejected batches are
   written to ``rejected/`` and never enter ``corpus/``.
2. **Stop criterion, not a rate target.** The loop watches *marginal diversity
   per hour* — new unique skeletons added per wall-clock hour over a trailing
   window — and stops when it falls under ``MIN_NEW_SKELETONS_PER_HOUR``. It
   stops; it does not spin, and it does not idle waiting for more work.
3. **Resumable manifest.** ``run.json`` plus an append-only normalized-text
   index let the loop be killed and restarted without regenerating what the
   corpus already holds: on resume the dedup index is rebuilt from disk, so
   everything already written is duplicate-pressure against the next round.

Why 250 new skeletons/hour is the floor
---------------------------------------
Under it, a full 24 h of runtime adds fewer than 6 000 new skeletons — less
than five seconds of the bounded #T-gen-schemas run (1 152 schemas in ~0.3 s).
At that point the job slot costs more than the information it buys, and the
right move is to widen the contract (more domains, more languages, a real
proposer behind ``JEV_GEN_ENDPOINT``) rather than to let this run keep going.
The floor is ~14x the information rate of the loop #T-halt-contam killed, so
"still above the floor" always means "strictly better than the thing we shot".

The rate is measured on WALL CLOCK, pacing interval included. That interval is
part of the real cost of keeping a job alive, so it belongs in the denominator.
"""
from __future__ import annotations

import json
import signal
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from data.firewall import ContaminationScanner

from .budget import Budget
from .dedup import SchemaDeduper
from .domains import DOMAIN_IDS, MAX_K, MIN_K
from .generator import DEFAULT_DEDUP_THRESHOLD, DEFAULT_UNKNOWN_RATE, generate
from .vocab import LANGS

ROOT = Path(__file__).resolve().parent.parent.parent
GATE_DIR = ROOT / "artifacts" / "gates" / "T-gen-loop"

# --- criteria, fixed before any number was measured -----------------------
# Stop criterion (see the module docstring for the arithmetic behind 250).
MIN_NEW_SKELETONS_PER_HOUR = 250.0
RATE_WINDOW_ROUNDS = 4          # trailing window the rate is averaged over
MAX_CONSECUTIVE_REJECTS = 3     # the corpus stopped accepting information

# Per-batch gate. A batch failing ANY of these is rejected whole.
MIN_BATCH_ROWS = 48             # under this a batch cannot span its own axes
MIN_BATCH_NOVELTY = 0.30        # share of the batch surviving corpus dedup
MAX_DOMAIN_SHARE = 0.40         # 6 domains; uniform is 0.167
MAX_LANGUAGE_SHARE = 0.55       # 4 languages; uniform is 0.25
MIN_K_VALUES_IN_BATCH = 3
MIN_HARD_NEGATIVE_RATE = 0.80   # same floor as the #T-gen-schemas gate
UNKNOWN_RATE_RANGE = (0.05, 0.35)

# Backstop only — NOT the stop criterion. Keeps a misconfigured run finite.
DEFAULT_MAX_ROWS = 200_000


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _share(counter: Counter, total: int) -> dict[str, float]:
    return {str(k): round(v / total, 4) for k, v in counter.items()}


# --- corpus index (the thing that makes the loop resumable) ---------------

class CorpusIndex:
    """Dedup index + skeleton set for everything already in the corpus.

    Persisted as an append-only ``state/norm.jsonl`` next to the corpus, so a
    restart rebuilds exactly the pressure the previous run had accumulated
    instead of starting from an empty index and regenerating it all.
    """

    def __init__(self, state_dir: Path,
                 threshold: float = DEFAULT_DEDUP_THRESHOLD) -> None:
        self.path = state_dir / "norm.jsonl"
        self.threshold = threshold
        self.deduper = SchemaDeduper(threshold)
        self.skeletons: set[str] = set()
        self.rows = 0

    def load(self) -> int:
        if not self.path.exists():
            return 0
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self.deduper.add(rec["n"])
                self.skeletons.add(rec["s"])
                self.rows += 1
        return self.rows

    def similarity(self, norm_text: str) -> float:
        return self.deduper.similarity(norm_text)[0]

    def commit(self, schemas) -> int:
        """Add accepted schemas to the index and the on-disk state. New skeletons."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = 0
        with open(self.path, "a") as f:
            for s in schemas:
                if s.skeleton not in self.skeletons:
                    new += 1
                self.skeletons.add(s.skeleton)
                self.deduper.add(s.norm_text)
                self.rows += 1
                f.write(json.dumps({"s": s.skeleton, "n": s.norm_text},
                                   ensure_ascii=False) + "\n")
        return new


# --- per-batch gate -------------------------------------------------------

def measure_batch(kept, produced: int, cross_dupes: int,
                  index: CorpusIndex) -> dict:
    """Everything the batch gate judges on. Pure measurement, no verdict."""
    n = len(kept)
    m = {
        "produced": produced,
        "cross_batch_duplicates": cross_dupes,
        "kept": n,
        "novelty": round(n / produced, 4) if produced else 0.0,
        "corpus_dedup_rejection_rate": (round(cross_dupes / produced, 4)
                                        if produced else 0.0),
    }
    if not n:
        m.update({"unique_skeletons": 0, "new_skeletons": 0,
                  "domains": {}, "languages": {}, "k_values": {},
                  "hard_negative_rate": 0.0, "mean_hard_negatives": 0.0,
                  "unknown_rate": 0.0})
        return m
    skels = {s.skeleton for s in kept}
    m.update({
        "unique_skeletons": len(skels),
        "new_skeletons": len(skels - index.skeletons),
        "domains": _share(Counter(s.domain for s in kept), n),
        "languages": _share(Counter(s.language for s in kept), n),
        "k_values": _share(Counter(s.k for s in kept), n),
        "hard_negative_rate": round(
            sum(1 for s in kept if s.hard_negatives) / n, 4),
        "mean_hard_negatives": round(
            sum(len(s.hard_negatives) for s in kept) / n, 3),
        "unknown_rate": round(sum(1 for s in kept if s.unknown) / n, 4),
    })
    return m


def judge_batch(m: dict) -> tuple[bool, dict]:
    """(accepted, per-check booleans). Every check must hold."""
    checks = {
        "rows": m["kept"] >= MIN_BATCH_ROWS,
        "novelty": m["novelty"] >= MIN_BATCH_NOVELTY,
        "domain_balance": (max(m["domains"].values(), default=1.0)
                           <= MAX_DOMAIN_SHARE),
        "language_balance": (max(m["languages"].values(), default=1.0)
                             <= MAX_LANGUAGE_SHARE),
        "k_coverage": len(m["k_values"]) >= MIN_K_VALUES_IN_BATCH,
        "hard_negatives": m["hard_negative_rate"] >= MIN_HARD_NEGATIVE_RATE,
        "unknown_rate": (UNKNOWN_RATE_RANGE[0] <= m["unknown_rate"]
                         <= UNKNOWN_RATE_RANGE[1]),
    }
    return all(checks.values()), checks


# --- the loop -------------------------------------------------------------

@dataclass
class LoopConfig:
    out: Path
    seed: int = 20260921
    per_cell: int = 3
    interval: float = 30.0
    unknown_rate: float = DEFAULT_UNKNOWN_RATE
    dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD
    domains: tuple[str, ...] = DOMAIN_IDS
    languages: tuple[str, ...] = LANGS
    ks: tuple[int, ...] = tuple(range(MIN_K, MAX_K + 1))
    max_rounds: int = 0          # 0 = unbounded; the diversity gate stops it
    max_hours: float = 0.0       # 0 = unbounded
    max_rows: int = DEFAULT_MAX_ROWS
    min_rate: float = MIN_NEW_SKELETONS_PER_HOUR
    rate_window: int = RATE_WINDOW_ROUNDS
    firewall: bool = True
    run_id: str = "gen-loop-v1"
    gate_dir: Path = GATE_DIR

    def criteria(self) -> dict:
        return {
            "stop": {
                "min_new_skeletons_per_hour": self.min_rate,
                "rate_window_rounds": self.rate_window,
                "max_consecutive_rejected_batches": MAX_CONSECUTIVE_REJECTS,
                "rate_basis": "wall clock, pacing interval included",
            },
            "batch": {
                "min_rows": MIN_BATCH_ROWS,
                "min_novelty": MIN_BATCH_NOVELTY,
                "max_domain_share": MAX_DOMAIN_SHARE,
                "max_language_share": MAX_LANGUAGE_SHARE,
                "min_k_values": MIN_K_VALUES_IN_BATCH,
                "min_hard_negative_rate": MIN_HARD_NEGATIVE_RATE,
                "unknown_rate_range": list(UNKNOWN_RATE_RANGE),
            },
            "backstops": {"max_rounds": self.max_rounds,
                          "max_hours": self.max_hours,
                          "max_rows": self.max_rows},
        }


@dataclass
class LoopState:
    rounds: list = field(default_factory=list)
    rows_accepted: int = 0
    rows_rejected: int = 0
    batches_accepted: int = 0
    batches_rejected: int = 0
    elapsed_sec: float = 0.0
    stopped: str = ""
    next_round: int = 1


class ContinuousLoop:
    def __init__(self, cfg: LoopConfig, log=print) -> None:
        self.cfg = cfg
        self.log = log
        self.out = cfg.out
        self.corpus_dir = self.out / "corpus"
        self.rejected_dir = self.out / "rejected"
        self.state_dir = self.out / "state"
        self.manifest = self.out / "run.json"
        self.index = CorpusIndex(self.state_dir, cfg.dedup_threshold)
        self.state = LoopState()
        self.scanner = ContaminationScanner() if cfg.firewall else None
        self._stop_signal = ""
        self._resumed: dict | None = None

    # --- persistence ------------------------------------------------------
    def resume(self) -> dict:
        """Rebuild from disk. Returns what was found.

        Idempotent: loading the normalized-text index twice would double the
        dedup pressure and the row count, so a second call is a no-op.
        """
        if self._resumed is not None:
            return self._resumed
        loaded = self.index.load()
        found = {"norm_rows": loaded, "manifest": False}
        if self.manifest.exists():
            data = json.loads(self.manifest.read_text())
            t = data.get("totals", {})
            self.state.rounds = data.get("rounds", [])
            self.state.rows_accepted = t.get("rows_accepted", 0)
            self.state.rows_rejected = t.get("rows_rejected", 0)
            self.state.batches_accepted = t.get("batches_accepted", 0)
            self.state.batches_rejected = t.get("batches_rejected", 0)
            self.state.elapsed_sec = t.get("elapsed_sec", 0.0)
            self.state.next_round = len(self.state.rounds) + 1
            found["manifest"] = True
            found["rounds"] = len(self.state.rounds)
        if loaded and loaded != self.state.rows_accepted:
            # The index is the authority: it is what dedup actually holds.
            self.log(f"[loop] manifest says {self.state.rows_accepted} accepted "
                     f"rows, dedup index holds {loaded}; trusting the index")
            self.state.rows_accepted = loaded
        self._resumed = found
        return found

    def write_manifest(self) -> Path:
        self.out.mkdir(parents=True, exist_ok=True)
        doc = {
            "task": "T-gen-loop",
            "run_id": self.cfg.run_id,
            "contract": "gen-schemas/v1 (#T-gen-schemas, commit 39e6322)",
            "seed": self.cfg.seed,
            "updated_at": utcnow(),
            "config": {
                "per_cell": self.cfg.per_cell,
                "interval_sec": self.cfg.interval,
                "unknown_rate": self.cfg.unknown_rate,
                "dedup_threshold": self.cfg.dedup_threshold,
                "domains": list(self.cfg.domains),
                "languages": list(self.cfg.languages),
                "ks": list(self.cfg.ks),
                "firewall": self.cfg.firewall,
            },
            "criteria": self.cfg.criteria(),
            "totals": self.totals(),
            "stopped": self.state.stopped,
            "rounds": self.state.rounds,
        }
        tmp = self.manifest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
        tmp.replace(self.manifest)
        return self.manifest

    def totals(self) -> dict:
        hours = self.state.elapsed_sec / 3600.0
        return {
            "rounds": len(self.state.rounds),
            "batches_accepted": self.state.batches_accepted,
            "batches_rejected": self.state.batches_rejected,
            "rows_accepted": self.state.rows_accepted,
            "rows_rejected": self.state.rows_rejected,
            "unique_skeletons": len(self.index.skeletons),
            "elapsed_sec": round(self.state.elapsed_sec, 1),
            "skeletons_per_hour": (round(len(self.index.skeletons) / hours, 1)
                                   if hours > 0 else None),
            "rows_per_skeleton": (
                round(self.state.rows_accepted / len(self.index.skeletons), 3)
                if self.index.skeletons else None),
        }

    # --- one round --------------------------------------------------------
    def round_seed(self, n: int) -> int:
        return (self.cfg.seed * 1_000_003 + n) % (2 ** 31)

    def run_round(self, n: int) -> dict:
        t0 = time.time()
        budget = Budget.plan(per_cell=self.cfg.per_cell,
                             domains=self.cfg.domains,
                             languages=self.cfg.languages, ks=self.cfg.ks)
        res = generate(budget, seed=self.round_seed(n),
                       unknown_rate=self.cfg.unknown_rate,
                       dedup_threshold=self.cfg.dedup_threshold,
                       scanner=self.scanner,
                       id_prefix=f"{self.cfg.run_id}-r{n:05d}",
                       log=lambda *a, **k: None)

        # Cross-batch stage: the corpus that ALREADY exists is the filter.
        kept, cross_dupes = [], 0
        for s in res.schemas:
            if self.index.similarity(s.norm_text) >= self.cfg.dedup_threshold:
                cross_dupes += 1
                continue
            kept.append(s)

        m = measure_batch(kept, len(res.schemas), cross_dupes, self.index)
        accepted, checks = judge_batch(m)
        shard = f"batch-{n:05d}.jsonl"
        target = (self.corpus_dir if accepted else self.rejected_dir) / shard
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w") as f:
            for s in kept:
                f.write(json.dumps(s.to_row(), ensure_ascii=False) + "\n")

        new_skeletons = 0
        if accepted:
            new_skeletons = self.index.commit(kept)
            self.state.rows_accepted += len(kept)
            self.state.batches_accepted += 1
        else:
            self.state.rows_rejected += len(kept)
            self.state.batches_rejected += 1

        seconds = time.time() - t0
        entry = {
            "round": n,
            "seed": self.round_seed(n),
            "at": utcnow(),
            "seconds": round(seconds, 3),
            "generator_stopped": res.stopped,
            "generator_attempts": res.attempts,
            "generator_rejections": dict(sorted(res.rejections.items())),
            "accepted": accepted,
            "checks": checks,
            "failed_checks": [k for k, ok in checks.items() if not ok],
            "measurement": m,
            "new_skeletons": new_skeletons,
            "corpus_rows": self.state.rows_accepted,
            "corpus_skeletons": len(self.index.skeletons),
            "file": str(target.relative_to(self.out)),
        }
        return entry

    # --- stop criterion ---------------------------------------------------
    def marginal_rate(self) -> float | None:
        """New unique skeletons per wall-clock hour over the trailing window."""
        window = self.state.rounds[-self.cfg.rate_window:]
        if len(window) < self.cfg.rate_window:
            return None  # warm-up: not enough rounds to average over
        seconds = sum(r["seconds"] + self.cfg.interval for r in window)
        if seconds <= 0:
            return None
        return (sum(r["new_skeletons"] for r in window) / seconds) * 3600.0

    def should_stop(self, started: float) -> str:
        c = self.cfg
        if self._stop_signal:
            return self._stop_signal
        if c.max_rounds and len(self.state.rounds) >= c.max_rounds:
            return f"backstop: max_rounds={c.max_rounds}"
        if c.max_rows and self.state.rows_accepted >= c.max_rows:
            return f"backstop: max_rows={c.max_rows}"
        if c.max_hours and (time.time() - started) / 3600.0 >= c.max_hours:
            return f"backstop: max_hours={c.max_hours}"
        recent = self.state.rounds[-MAX_CONSECUTIVE_REJECTS:]
        if (len(recent) >= MAX_CONSECUTIVE_REJECTS
                and not any(r["accepted"] for r in recent)):
            return (f"diversity gate: {MAX_CONSECUTIVE_REJECTS} consecutive "
                    f"rejected batches — the corpus stopped accepting information")
        rate = self.marginal_rate()
        if rate is not None and rate < c.min_rate:
            return (f"diversity gate: marginal diversity {rate:.1f} new "
                    f"skeletons/h over the last {c.rate_window} rounds is under "
                    f"the {c.min_rate:.0f}/h floor")
        return ""

    # --- driver -----------------------------------------------------------
    def install_signal_handlers(self) -> None:
        def handler(signum, _frame):
            self._stop_signal = f"signal {signal.Signals(signum).name}"
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, handler)
            except ValueError:  # not on the main thread (tests)
                pass

    def run(self) -> dict:
        started = time.time()
        found = self.resume()
        if found["manifest"] or found["norm_rows"]:
            self.log(f"[loop] resumed: {found['norm_rows']} indexed rows, "
                     f"{found.get('rounds', 0)} previous rounds")
        self.log(f"[loop] contract gen-schemas/v1 · stop when marginal "
                 f"diversity < {self.cfg.min_rate:.0f} new skeletons/h over "
                 f"{self.cfg.rate_window} rounds")

        while True:
            n = self.state.next_round
            entry = self.run_round(n)
            self.state.rounds.append(entry)
            self.state.next_round = n + 1
            self.state.elapsed_sec += entry["seconds"] + self.cfg.interval
            rate = self.marginal_rate()
            self.log(
                f"[loop] round {n} {'ACCEPT' if entry['accepted'] else 'REJECT'} "
                f"kept={entry['measurement']['kept']}/"
                f"{entry['measurement']['produced']} "
                f"novelty={entry['measurement']['novelty']:.3f} "
                f"new-skeletons={entry['new_skeletons']} "
                f"corpus={entry['corpus_rows']} rows/"
                f"{entry['corpus_skeletons']} skeletons "
                f"rate={'--' if rate is None else f'{rate:.0f}/h'}"
                + (f" failed={entry['failed_checks']}"
                   if entry["failed_checks"] else ""))
            self.write_manifest()
            write_gate(self, self.cfg.gate_dir, log=lambda *a, **k: None)

            stop = self.should_stop(started)
            if stop:
                self.state.stopped = stop
                break
            if self.cfg.interval > 0:
                # Woken by SIGTERM in a second or less, not at the end of a nap.
                slept = 0.0
                while slept < self.cfg.interval and not self._stop_signal:
                    time.sleep(min(1.0, self.cfg.interval - slept))
                    slept += 1.0

        self.log(f"[loop] STOPPED — {self.state.stopped}")
        self.write_manifest()
        gate = write_gate(self, self.cfg.gate_dir, log=self.log)
        return gate


# --- the #T-gen-loop verification gate ------------------------------------


def diversity_curve(loop: ContinuousLoop) -> list[dict]:
    """Corpus skeletons against cumulative wall-clock, round by round."""
    curve, elapsed = [], 0.0
    for r in loop.state.rounds:
        elapsed += r["seconds"] + loop.cfg.interval
        curve.append({
            "round": r["round"],
            "elapsed_sec": round(elapsed, 1),
            "accepted": r["accepted"],
            "rows": r["corpus_rows"],
            "skeletons": r["corpus_skeletons"],
            "new_skeletons": r["new_skeletons"],
            "novelty": r["measurement"]["novelty"],
            "marginal_per_hour": (
                round(r["new_skeletons"] / (r["seconds"] + loop.cfg.interval)
                      * 3600.0, 1) if (r["seconds"] + loop.cfg.interval) else None),
        })
    return curve


def build_gate(loop: ContinuousLoop) -> dict:
    """The #T-gen-loop gate, valid MID-FLIGHT as well as after the stop.

    A gate that only passed once the job died would call a healthy continuous
    loop a failure for as long as it was working, and would call an operator's
    restart a failure forever. So the self-limiting check has two honest ways
    to hold: the run stopped on its own criterion, or it was still above the
    floor it will stop at when it was last measured — the same statement,
    evaluated before and after the fact.
    """
    totals = loop.totals()
    rounds = loop.state.rounds
    corpus_rows = corpus_schemas(loop.corpus_dir)
    rate = loop.marginal_rate()
    running = not loop.state.stopped
    novelties = [r["measurement"]["novelty"] for r in rounds]
    checks = {
        "batch_gate_ran": len(rounds) > 0,
        "rejected_batches_are_out_of_corpus": corpus_rows == totals["rows_accepted"],
        "corpus_beats_the_killed_loop": (
            (totals["rows_per_skeleton"] or 0) <= 1.05),
        "manifest_resumable": loop.manifest.exists() and loop.index.path.exists(),
        "no_spin": all(r["seconds"] < 3600 for r in rounds),
        "novelty_is_measured": len(novelties) > 0,
        # Self-limiting means the loop never ran on below its own floor: it
        # either stopped on the diversity gate, or it was still above the floor
        # when it was interrupted (an operator restart, a backstop) or measured.
        "self_limiting": (loop.state.stopped.startswith("diversity gate")
                          or rate is None or rate >= loop.cfg.min_rate),
    }
    return {
        "task": "T-gen-loop",
        "pass": all(checks.values()),
        "state": "running" if running else "stopped",
        "generated_at": utcnow(),
        "run_id": loop.cfg.run_id,
        "seed": loop.cfg.seed,
        "contract": "gen-schemas/v1 (#T-gen-schemas, commit 39e6322)",
        "stop_criterion": {
            "primary": (f"marginal diversity < {loop.cfg.min_rate:.0f} new unique "
                        f"skeletons per wall-clock hour, averaged over the last "
                        f"{loop.cfg.rate_window} rounds"),
            "secondary": (f"{MAX_CONSECUTIVE_REJECTS} consecutive batches "
                          f"rejected by the per-batch gate"),
            "rationale": (
                "Under 250 new skeletons/h a full day of runtime adds fewer than "
                "6 000 skeletons - less than five seconds of the bounded "
                "#T-gen-schemas run. The loop #T-halt-contam killed was running "
                "at ~17 new skeletons/h (190 skeletons in 11 h), so this floor is "
                "~14x its information rate."),
            "marginal_new_skeletons_per_hour": (round(rate, 1)
                                                if rate is not None else None),
            "above_floor": (rate is None or rate >= loop.cfg.min_rate),
            "stopped": loop.state.stopped,
        },
        "criteria": loop.cfg.criteria(),
        "checks": checks,
        "totals": totals,
        "corpus_rows_on_disk": corpus_rows,
        "rows_accepted_vs_rejected": {
            "accepted": totals["rows_accepted"],
            "rejected": totals["rows_rejected"],
            "batches_accepted": totals["batches_accepted"],
            "batches_rejected": totals["batches_rejected"],
            "rejection_reasons": dict(Counter(
                c for r in rounds for c in r["failed_checks"])),
        },
        "novelty": {
            "first_round": novelties[0] if novelties else None,
            "last_round": novelties[-1] if novelties else None,
            "decaying": (len(novelties) > 1 and novelties[-1] < novelties[0]),
        },
        "diversity_curve": diversity_curve(loop),
        "batch_gate_rounds": [
            {"round": r["round"], "accepted": r["accepted"],
             "failed_checks": r["failed_checks"],
             "kept": r["measurement"]["kept"],
             "produced": r["measurement"]["produced"],
             "novelty": r["measurement"]["novelty"],
             "hard_negative_rate": r["measurement"]["hard_negative_rate"],
             "unknown_rate": r["measurement"]["unknown_rate"],
             "domains": len(r["measurement"]["domains"]),
             "languages": len(r["measurement"]["languages"]),
             "k_values": len(r["measurement"]["k_values"])}
            for r in rounds],
        "killed_loop_reference": {
            "task": "T-halt-contam", "rows": 258_700, "skeletons": 190,
            "hours": 11, "new_skeletons_per_hour": 17.3,
            "rows_per_skeleton": 1361.6,
        },
    }


def corpus_schemas(corpus_dir: Path) -> int:
    if not corpus_dir.exists():
        return 0
    n = 0
    for shard in sorted(corpus_dir.glob("batch-*.jsonl")):
        with open(shard) as f:
            n += sum(1 for line in f if line.strip())
    return n


def write_gate(loop: ContinuousLoop, gate_dir: Path = GATE_DIR,
               log=print) -> dict:
    gate = build_gate(loop)
    gate_dir.mkdir(parents=True, exist_ok=True)
    path = gate_dir / "gate.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(gate, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)
    log(f"[gate] {'PASS' if gate['pass'] else 'FAIL'} -> {path}")
    for name, ok in gate["checks"].items():
        log(f"  {'ok  ' if ok else 'FAIL'} {name}")
    return gate
