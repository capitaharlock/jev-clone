#!/usr/bin/env python3
"""Budgeted decision-schema generator (#T-gen-schemas).

Replaces the endless email-template loop of #T-gen-loop. That loop asked
Q-W-E-N for *templates*, dropped everything that did not match four
hardcoded `STATES` or contained a `{}`, and so produced one task — Spanish
email triage, four fixed labels — 258 700 times (audit-2026-09-21, findings
C and D). Three things change here:

* the unit is a COMPLETE DECISION SCHEMA: domain, state format, state,
  question, K options (K variable, 3..8) with plausible distractors, the
  correct answer, and a controlled fraction answered `unknown` because the
  state does not carry it;
* the run is bounded by a QUOTA per domain x language x K — it ends when the
  budget is full, not when someone kills it;
* every candidate is deduped by 3-gram Jaccard (`data.leakage`) against
  everything already generated, with proper nouns and figures masked first,
  and each rejection is counted with its reason.

Diversity is published to `<out>/diversity.json` and appended to
`<out>/diversity-history.jsonl`; the gate writes
`artifacts/gates/T-gen-schemas/gate.json`.

#T-gen-loop adds the CONTINUOUS mode on top of that one-shot run: `--loop`
keeps generating batches, gates every batch against the corpus that already
exists, and STOPS when marginal diversity per hour falls under the floor —
the measurement the old loop never had. See `tools/gen_schemas/loop.py`.

Run (no sklearn or torch needed — this only writes data):
  python tools/data_gen_loop.py --per-cell 8      # one bounded run
  python tools/data_gen_loop.py --gate            # #T-gen-schemas gate only
  python tools/data_gen_loop.py --loop            # continuous, self-limiting
Long runs belong to the daemon's job runner, never to a nohup.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.firewall import ContaminationScanner  # noqa: E402
from tools.gen_schemas import (Budget, LANGS, MAX_K, MIN_K,  # noqa: E402
                               DOMAIN_IDS, diversity, generate)
from tools.gen_schemas import gate as gate_mod  # noqa: E402
from tools.gen_schemas.loop import ContinuousLoop, LoopConfig  # noqa: E402

# Gitignored, and deliberately NOT artifacts/data-prefetch/manifest.json:
# nothing here joins a training mix by being written. #T-halt-contam put the
# previous corpus in quarantine for exactly that reason.
OUT_DIR = ROOT / "artifacts" / "data-prefetch" / "gen-schemas"
LOOP_DIR = ROOT / "artifacts" / "data-prefetch" / "gen-loop"


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:  # an --out outside the repo (tests, scratch runs)
        return str(path)


def write_rows(schemas, out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for s in schemas:
            f.write(json.dumps(s.to_row(), ensure_ascii=False) + "\n")
    return len(schemas)


def write_reports(report: dict, out_dir: Path, rows: int, path: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = dict(report, generated_at=utcnow(), rows=rows,
                  file=rel(path))
    (out_dir / "diversity.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    trend = {"ts": report["generated_at"], "rows": rows,
             "unique_skeletons": report["unique_skeletons"],
             "skeletons_per_schema": report["skeletons_per_schema"],
             "domains": len(report["domains"]),
             "languages": len(report["languages"]),
             "unknown_rate": report["unknown_rate"],
             "dedup_rejection_rate": report.get("dedup", {}).get("rejection_rate")}
    with open(out_dir / "diversity-history.jsonl", "a") as f:
        f.write(json.dumps(trend) + "\n")


def parse_csv(value: str | None, known: tuple, name: str) -> tuple:
    if not value:
        return known
    picked = tuple(v.strip() for v in value.split(",") if v.strip())
    unknown = [v for v in picked if v not in known]
    if unknown:
        raise SystemExit(f"unknown {name}: {unknown}; known: {list(known)}")
    return picked


def run_loop(args, log) -> int:
    """#T-gen-loop: continuous generation, stopped by the diversity gate."""
    ks = (tuple(int(k) for k in args.ks.split(",")) if args.ks
          else tuple(range(MIN_K, MAX_K + 1)))
    cfg = LoopConfig(
        out=args.loop_out, seed=args.seed, per_cell=args.per_cell,
        interval=args.interval, unknown_rate=args.unknown_rate,
        dedup_threshold=args.dedup_threshold,
        domains=parse_csv(args.domains, DOMAIN_IDS, "domain"),
        languages=parse_csv(args.languages, LANGS, "language"), ks=ks,
        max_rounds=args.max_rounds, max_hours=args.max_hours,
        firewall=not args.no_firewall, run_id=args.run_id,
    )
    if args.min_rate is not None:
        cfg.min_rate = args.min_rate
    if args.rate_window is not None:
        cfg.rate_window = args.rate_window
    loop = ContinuousLoop(cfg, log=log)
    loop.install_signal_handlers()
    gate = loop.run()
    log(f"[loop] corpus {rel(loop.corpus_dir)} · manifest {rel(loop.manifest)}")
    return 0 if gate["pass"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-cell", type=int, default=8,
                    help="quota per (domain, language, K) cell")
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--unknown-rate", type=float, default=0.15)
    ap.add_argument("--dedup-threshold", type=float, default=0.7)
    ap.add_argument("--domains", help="csv subset of " + ",".join(DOMAIN_IDS))
    ap.add_argument("--languages", help="csv subset of " + ",".join(LANGS))
    ap.add_argument("--ks", help=f"csv option cardinalities in [{MIN_K},{MAX_K}]")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "schemas.jsonl")
    ap.add_argument("--no-firewall", action="store_true",
                    help="skip the benchmark firewall (never in a real run)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--gate", action="store_true",
                    help="run the #T-gen-schemas verification gate and exit")
    ap.add_argument("--loop", action="store_true",
                    help="#T-gen-loop: continuous, diversity-gated generation")
    ap.add_argument("--loop-out", type=Path, default=LOOP_DIR,
                    help="corpus root for --loop (corpus/, rejected/, run.json)")
    ap.add_argument("--interval", type=float, default=30.0,
                    help="seconds between batches; counted in the rate")
    ap.add_argument("--min-rate", type=float,
                    help="stop under this many new skeletons per hour")
    ap.add_argument("--rate-window", type=int,
                    help="rounds the marginal rate is averaged over")
    ap.add_argument("--max-rounds", type=int, default=0, help="0 = unbounded")
    ap.add_argument("--max-hours", type=float, default=0.0, help="0 = unbounded")
    ap.add_argument("--run-id", default="gen-loop-v1")
    args = ap.parse_args(argv)
    # Flushing: --loop is a daemon job, and a block-buffered stdout leaves the
    # cockpit's Jobs log empty for minutes at a time.
    log = ((lambda *a: None) if args.quiet
           else (lambda *a: print(*a, flush=True)))

    if args.gate:
        return gate_mod.main(["--per-cell", str(args.per_cell),
                              "--seed", str(args.seed)])

    if args.loop:
        return run_loop(args, log)

    ks = (tuple(int(k) for k in args.ks.split(",")) if args.ks
          else tuple(range(MIN_K, MAX_K + 1)))
    budget = Budget.plan(per_cell=args.per_cell,
                         domains=parse_csv(args.domains, DOMAIN_IDS, "domain"),
                         languages=parse_csv(args.languages, LANGS, "language"),
                         ks=ks)
    log(f"[gen] budget {budget.total()} schemas over {len(budget.cells())} "
          f"cells (domain x language x K)")

    t0 = time.time()
    res = generate(budget, seed=args.seed, unknown_rate=args.unknown_rate,
                   dedup_threshold=args.dedup_threshold, log=log,
                   scanner=None if args.no_firewall else ContaminationScanner())
    report = diversity(res.schemas, res.rejections, res.deduper, budget,
                       res.proposer)
    rows = write_rows(res.schemas, args.out)
    write_reports(report, args.out.parent, rows, args.out)

    log(f"[gen] {rows} schemas in {time.time() - t0:.1f}s -> {rel(args.out)}")
    log(f"[gen] stopped: {res.stopped} (attempts {res.attempts})")
    log(f"[gen] skeletons={report['unique_skeletons']} "
        f"domains={len(report['domains'])} langs={len(report['languages'])} "
        f"K={sorted(report['k_distribution'])} "
        f"unknown={report['unknown_rate']:.3f} "
        f"hard-neg={report['hard_negative_rate']:.3f}")
    log(f"[gen] rejections {report['rejections']} "
        f"(dedup rate {report['dedup']['rejection_rate']:.3f})")
    if budget.overflow():
        print(f"[gen] QUOTA OVERFLOW {budget.overflow()}", flush=True)
        return 1
    return 0 if budget.report()["full"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
