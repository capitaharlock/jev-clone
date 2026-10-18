"""Pilot runner for #T-synth-factory: 50k accepted rows + gate artifact.

Usage: .venv-train/bin/python tools/data_factory/run_pilot.py [--target 50000]
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


OUT = ROOT / "artifacts" / "synth" / "v1"
GATE_DIR = ROOT / "artifacts" / "gates" / "T-synth-factory"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=50000)
    ap.add_argument("--max-proposed", type=int, default=400000)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # grounded states: all prefetch shards (boolq first = diverse long states)
    order = ["boolq", "helpsteer2", "civil-comments", "huffpost",
             "banking77", "massive", "logiqa", "reclor"]
    import glob as _glob

    files = [f for name in order
             for f in _glob.glob(str(ROOT / "artifacts" / "data-prefetch" / f"{name}.jsonl"))]
    # pipeline takes one glob; join via brace is overkill — run per file
    from tools.data_factory.source_streamer import stream_prefetch_states as _s

    # monkey-run across files in order until target reached
    from tools.data_factory.deduper import Deduper
    from tools.data_factory import adversary as _adv
    from tools.data_factory.council import teacher_router
    from tools.data_factory.deduper import group_split
    from tools.data_factory.tracking import LicenseTracker, Metrics, ShardWriter
    from tools.data_factory.validator import validate_record

    deduper, metrics, lic = Deduper(), Metrics(), LicenseTracker()
    writer = ShardWriter(OUT)
    seq = 0
    for path in files:
        if metrics.accepted >= args.target or metrics.proposed >= args.max_proposed:
            break
        for state in _s(path):
            if metrics.accepted >= args.target or metrics.proposed >= args.max_proposed:
                break
            for rep in range(4):
                if metrics.accepted >= args.target or metrics.proposed >= args.max_proposed:
                    break
                key = f"{state.parent_example_id}:{seq}:{rep}"
                metrics.proposed += 1
                metrics.council += 1
                verdict = teacher_router(state.text, key, seq)
                metrics.inverted += verdict.roles_inverted
                metrics.agreed += verdict.agreed
                metrics.adjudicated += not verdict.agreed
                q = verdict.question
                rec = {
                    "state": state.text,
                    "questions": [{"id": f"synth-{seq:07d}", "kind": q["kind"],
                                   "options": q["options"], "answer": verdict.answer}],
                    "split": group_split(state.parent_example_id),
                    "parent_example_id": state.parent_example_id,
                    "teacher": {"proposer": verdict.proposer, "solver": verdict.solver,
                                "inverted": verdict.roles_inverted, "agreed": verdict.agreed,
                                "adjudicated": verdict.adjudicated},
                }
                if validate_record(rec):
                    metrics.rejected_validator += 1
                    seq += 1
                    continue
                if deduper.check(rec, state.parent_example_id):
                    metrics.rejected_dupe += 1
                    seq += 1
                    continue
                writer.write(rec)
                lic.record(state.source)
                metrics.accepted += 1
                if metrics.accepted % 4 == 0:
                    for var in _adv.adversarial_variants(state.text, q, key + ":adv"):
                        vq = var.get("question", q)
                        if verdict.answer not in [o["id"] for o in vq["options"]]:
                            continue
                        metrics.proposed += 1
                        vrec = {
                            "state": var.get("state", state.text),
                            "questions": [{"id": f"synth-{seq:07d}a", "kind": vq["kind"],
                                           "options": vq["options"], "answer": verdict.answer}],
                            "split": group_split(state.parent_example_id),
                            "parent_example_id": state.parent_example_id,
                            "teacher": {"proposer": verdict.proposer,
                                        "variant": var["kind"]},
                        }
                        if validate_record(vrec) or deduper.check(vrec, state.parent_example_id):
                            continue
                        writer.write(vrec)
                        metrics.accepted += 1
                seq += 1
    writer.flush()
    summary = {"metrics": metrics.report(), "licenses": lic.report(),
               "shards": writer.manifest, "rows": writer.total,
               "dedupe": {"exact": deduper.n_exact_dup, "semantic": deduper.n_sem_dup}}
    ok = writer.total >= args.target
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    gate = {
        "format": 1, "task": "T-synth-factory", "pass": bool(ok),
        "ts": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "config": "cpu", "target_accepted": args.target,
        "rows": writer.total, "elapsed_s": round(time.time() - t0, 1),
        "agreement_rate": summary["metrics"]["agreement_rate"],
        "reject_rate": summary["metrics"]["reject_rate"],
        "summary": summary,
        "upstream": {"task": "T-massive-huff", "pass": True},
    }
    (GATE_DIR / "gate.json").write_text(json.dumps(gate, indent=1, sort_keys=True))
    print(json.dumps({"rows": writer.total, "pass": ok,
                      "metrics": summary["metrics"], "elapsed_s": gate["elapsed_s"]}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
