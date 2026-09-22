"""Prefetch + convert all training sources to the universal schema (#T-prefetch).

Downloads (via `datasets` + HF hub cache) every source listed in
`.meshkore/docs/source-register.md`, converts P0 rows with `data/adapters.py`
into `data/schema.py` Examples, writes one JSONL per dataset under
`artifacts/data-prefetch/`, and emits `manifest.json` (repo + revision + SHA).

Defensive by design: each dataset is wrapped in try/except, unknown columns
fall back to a logged raw-sample dump, and the run never dies halfway.
Eval-only sources (CLINC150) are snapshotted but NEVER converted for train.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
import traceback

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "artifacts", "data-prefetch",
)

# HF split -> universal schema split
SPLIT_MAP = {"train": "train", "validation": "calibration",
             "valid": "calibration", "dev": "calibration",
             "test": "test", "tests": "test"}

P0_JOBS = [
    # (job id, hf id, split names to try, kind)
    ("huffpost", "khalidalt/HuffPost", ["train"], "huffpost"),
    ("banking77", "PolyAI/banking77", ["train", "test"], "banking77"),
    ("boolq", "google/boolq", ["train", "validation"], "boolq"),
    ("civil-comments", "google/civil_comments",
     ["train", "validation", "test"], "civil"),
    ("helpsteer2", "nvidia/HelpSteer2",
     ["train", "validation"], "helpsteer2"),
]

P1_JOBS = [
    ("massive", "AmazonScience/massive", ["train", "validation", "test"]),
    ("logiqa20", "datatune/LogiQA2.0", ["train", "validation", "test"]),
    ("reclor", "sxiong/ReClor", ["train", "validation", "test"]),
]


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm_split(name: str) -> str:
    return SPLIT_MAP.get(name, "train")


def convert_p0(job_id, hf_id, splits, kind):
    from datasets import load_dataset
    import data.adapters as A

    adapt = {"huffpost": A.adapt_huffpost, "banking77": A.adapt_banking77,
             "boolq": A.adapt_boolq, "civil": A.adapt_civil,
             "helpsteer2": A.adapt_helpsteer2}[kind]
    ds = load_dataset(hf_id, trust_remote_code=False)
    try:
        revision = ds.cache_files[0]["filename"].split("___")[0] \
            if getattr(ds, "cache_files", None) else "unresolved"
    except Exception:
        revision = "unresolved"
    n_rows, n_ex, skipped = 0, 0, 0
    out_path = os.path.join(OUT_DIR, f"{job_id}.jsonl")
    with open(out_path, "w") as f:
        for split in splits:
            if split not in ds:
                continue
            table = ds[split]
            label_names = None
            try:
                feat = table.features.get("label")
                if hasattr(feat, "names"):
                    label_names = feat.names
            except Exception:
                pass
            for i, r in enumerate(table):
                try:
                    row = dict(r)
                    row["split"] = _norm_split(split)
                    if kind == "huffpost":
                        row = {"headline": row.get("headline", ""),
                               "description": row.get("short_description",
                                                      row.get("description", "")),
                               "category": str(row.get("category", "")).upper(),
                               "split": row["split"]}
                        if row["category"] not in A.HUFFPOST_UNIVERSE:
                            skipped += 1
                            continue
                    elif kind == "banking77" and isinstance(row.get("label"), int):
                        row["label"] = (label_names[row["label"]]
                                        if label_names else str(row["label"]))
                    elif kind == "civil":
                        tox = row.get("toxicity", row.get("toxic"))
                        row = {"text": row.get("text", ""),
                               "toxic": (tox >= 0.5) if isinstance(tox, float)
                               else bool(tox) if tox is not None else None,
                               "slice": "all", "split": row["split"]}
                    n_rows += 1
                    for ex in adapt([row], seed=0) if kind in (
                            "huffpost", "banking77") else adapt([row]):
                        ex.split = row["split"]
                        f.write(json.dumps(dataclasses.asdict(ex),
                                           ensure_ascii=False) + "\n")
                        n_ex += 1
                except Exception:
                    skipped += 1
            del table
    return {"job": job_id, "hf_id": hf_id, "status": "converted",
            "rows": n_rows, "examples": n_ex, "skipped": skipped,
            "sha256": _sha(out_path), "revision": str(revision)}


def dump_p1(job_id, hf_id, splits):
    """P1: normalized raw dump (no P1 adapter exists yet; never train-direct)."""
    from datasets import load_dataset

    try:
        ds = load_dataset(hf_id, trust_remote_code=False)
    except Exception as e:
        if "no longer supported" in str(e) or "script" in str(e).lower():
            return snapshot_fallback(job_id, hf_id, str(e))
        raise
    avail = [s for s in splits if s in ds]
    sample_cols = ds[avail[0]].column_names if avail else []
    sample = [dict(r) for r in ds[avail[0]].select(range(min(3, len(ds[avail[0]]))))] \
        if avail else []
    n_rows = 0
    out_path = os.path.join(OUT_DIR, f"{job_id}.raw.jsonl")
    locales = None
    with open(out_path, "w") as f:
        for split in avail:
            table = ds[split]
            if "locale" in table.column_names:
                table = table.filter(lambda r: r.get("locale") in ("en-US", "es-ES"))
                locales = ["en-US", "es-ES"]
            for r in table:
                f.write(json.dumps({"split": _norm_split(split),
                                    **{k: (v if isinstance(v, (str, int, float, bool))
                                           or v is None else str(v))
                                        for k, v in dict(r).items()}},
                                   ensure_ascii=False) + "\n")
                n_rows += 1
            del table
    return {"job": job_id, "hf_id": hf_id, "status": "raw-dump",
            "rows": n_rows, "columns": sample_cols, "sample": sample,
            "locales": locales, "sha256": _sha(out_path)}


def snapshot_fallback(job_id, hf_id, reason):
    """Legacy-script datasets (datasets>=3 refuses them): raw hub snapshot.

    Files land in the HF cache once; conversion to the universal schema
    happens later from parquet. Never train-direct."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=hf_id, repo_type="dataset")
    files = []
    for root, _, names in os.walk(path):
        files.extend(os.path.relpath(os.path.join(root, n), path)
                     for n in names)
    return {"job": job_id, "hf_id": hf_id, "status": "raw-snapshot",
            "reason": reason, "snapshot_path": path,
            "files": len(files), "sha256": hashlib.sha256(
                "\n".join(sorted(files)).encode()).hexdigest()}


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = {"jobs": []}
    for job in P0_JOBS:
        try:
            rec = convert_p0(*job)
        except Exception as e:
            if "no longer supported" in str(e):
                try:
                    rec = snapshot_fallback(job[0], job[1], str(e))
                except Exception as e2:
                    rec = {"job": job[0], "hf_id": job[1],
                           "status": f"BLOCKED: {e2}",
                           "trace": traceback.format_exc(limit=5)}
            else:
                rec = {"job": job[0], "hf_id": job[1], "status": f"BLOCKED: {e}",
                       "trace": traceback.format_exc(limit=5)}
        manifest["jobs"].append(rec)
        print(f"[{rec['job']}] {rec['status']}", flush=True)
    for job in P1_JOBS:
        try:
            rec = dump_p1(*job)
        except Exception as e:
            rec = {"job": job[0], "hf_id": job[1], "status": f"BLOCKED: {e}",
                   "trace": traceback.format_exc(limit=5)}
        manifest["jobs"].append(rec)
        print(f"[{rec['job']}] {rec['status']}", flush=True)
    ok = sum(1 for r in manifest["jobs"] if not str(r["status"]).startswith("BLOCKED"))
    manifest["summary"] = {"ok": ok, "total": len(manifest["jobs"])}
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"PREFETCH DONE ok={ok}/{len(manifest['jobs'])}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
