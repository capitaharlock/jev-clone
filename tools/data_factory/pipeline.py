"""End-to-end pipeline: stream -> council -> adversary -> validate -> dedupe -> shards."""
from __future__ import annotations

from . import adversary as _adv
from .council import teacher_router
from .deduper import Deduper, group_split
from .source_streamer import stream_prefetch_states
from .tracking import LicenseTracker, Metrics, ShardWriter
from .validator import validate_record


def run(pattern: str, out_dir: str, target_accepted: int = 50000,
        max_proposed: int = 200000, seed_offset: int = 0) -> dict:
    deduper = Deduper()
    metrics = Metrics()
    lic = LicenseTracker()
    writer = ShardWriter(out_dir)
    seq = seed_offset
    for state in stream_prefetch_states(pattern):
        if metrics.accepted >= target_accepted or metrics.proposed >= max_proposed:
            break
        # cycle states with sequence salt so one pass yields many items
        for rep in range(4):
            if metrics.accepted >= target_accepted or metrics.proposed >= max_proposed:
                break
            key = f"{state.parent_example_id}:{seq}:{rep}"
            metrics.proposed += 1
            metrics.council += 1
            verdict = teacher_router(state.text, key, seq)
            if verdict.roles_inverted:
                metrics.inverted += 1
            if verdict.agreed:
                metrics.agreed += 1
            if not verdict.agreed:
                metrics.adjudicated += 1
            q = verdict.question
            rec = {
                "state": state.text,
                "questions": [{
                    "id": f"synth-{seq:07d}",
                    "kind": q["kind"],
                    "options": q["options"],
                    "answer": verdict.answer,
                }],
                "split": group_split(state.parent_example_id),
                "parent_example_id": state.parent_example_id,
                "teacher": {"proposer": verdict.proposer, "solver": verdict.solver,
                            "inverted": verdict.roles_inverted,
                            "agreed": verdict.agreed,
                            "adjudicated": verdict.adjudicated},
            }
            errs = validate_record(rec)
            if errs:
                metrics.rejected_validator += 1
                seq += 1
                continue
            dupe = deduper.check(rec, state.parent_example_id)
            if dupe:
                metrics.rejected_dupe += 1
                seq += 1
                continue
            writer.write(rec)
            lic.record(state.source)
            metrics.accepted += 1
            # one adversarial variant per 4th accepted item
            if metrics.accepted % 4 == 0:
                for var in _adv.adversarial_variants(state.text, q, key + ":adv"):
                    vq = var.get("question", q)
                    opt_ids = [o["id"] for o in vq["options"]]
                    if verdict.answer not in opt_ids:
                        continue
                    metrics.proposed += 1
                    vrec = {
                        "state": var.get("state", state.text),
                        "questions": [{
                            "id": f"synth-{seq:07d}a",
                            "kind": vq["kind"],
                            "options": vq["options"],
                            "answer": verdict.answer,
                        }],
                        "split": group_split(state.parent_example_id),
                        "parent_example_id": state.parent_example_id,
                        "teacher": {"proposer": verdict.proposer, "variant": var["kind"]},
                    }
                    if validate_record(vrec):
                        continue
                    if deduper.check(vrec, state.parent_example_id):
                        continue
                    writer.write(vrec)
                    metrics.accepted += 1
            seq += 1
    writer.flush()
    return {
        "metrics": metrics.report(),
        "licenses": lic.report(),
        "shards": writer.manifest,
        "rows": writer.total,
        "dedupe": {"exact": deduper.n_exact_dup, "semantic": deduper.n_sem_dup},
    }
