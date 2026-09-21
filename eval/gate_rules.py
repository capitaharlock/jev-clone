"""Coherence rules for EVERY gate in the repo (#T-release-gate, rule 3).

A gate artifact may publish a bad number. What it may not do is publish a
GREEN one that contradicts itself. Three conditions of publication, from
`.meshkore/docs/release-criteria.md` §5 — the thresholds live in that file,
never here:

* **C1** — `cohen_kappa <= floor` in a gate that claims green. Agreement
  indistinguishable from chance cannot sit next to a PASS. The case that
  produced the rule is `artifacts/gates/T-release/release.json`:
  `exact_agree_rate 0.852` published beside `cohen_kappa 0.0`.
* **C2** — a metric with no `model_version` in a gate that claims green. A
  number nobody can attribute to a checkpoint is not reproducible, and it is
  the exact hole through which a parameter-free `cosine-char3-softmax` score
  got published as if it were the product's.
* **C3** — a quality metric measured on a split with no seal. A `seed` alone
  is not a seal: finding C (`i % 10`) was perfectly reproducible and still
  measured contamination.

These are ERRORS, never warnings: `scan()` returns a non-empty `errors` list
and the CLI exits non-zero. Offending gates are MARKED (a
`coherence-invalid.json` next to the artifact), never deleted — deleting a
bad published number is just another way of lying about it.

Scope: the unit is the gate DIRECTORY. `artifacts/gates/<task>/` is one
publication, so a green `gate.json` answers for the numbers its sibling files
publish.

CLI (stdlib only, any python3, run from the repo root):

    python3 -m eval.gate_rules scan            # audit, exit 1 if any error
    python3 -m eval.gate_rules scan --mark     # + write coherence-invalid.json
    python3 -m eval.gate_rules check <dir>     # one gate directory
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GATES_DIR = ROOT / "artifacts" / "gates"
CRITERIA_MD = ROOT / ".meshkore" / "docs" / "release-criteria.md"
MARKER_NAME = "coherence-invalid.json"

#: Keys whose value is a claim of quality about a model.
QUALITY_METRIC_KEYS = frozenset({
    "accuracy", "ece", "brier", "nll", "cohen_kappa", "kappa",
    "exact_agree_rate", "agreement_rate", "f1", "macro_f1", "precision",
    "recall", "auroc", "auc", "detection_rate", "false_positive_rate",
    "reject_rate", "overall_error", "avg_selective_risk",
})
#: Keys whose value is a claim about serving cost.
LATENCY_METRIC_KEYS = frozenset({
    "p50_ms", "p95_ms", "p99_ms", "warm_p50_ms", "warm_p95_ms",
    "warm_p99_ms", "mean_ms", "cold_ms",
})
METRIC_KEYS = QUALITY_METRIC_KEYS | LATENCY_METRIC_KEYS
KAPPA_KEYS = frozenset({"cohen_kappa", "kappa"})

#: A seal is a content digest, not an intention.
SEAL_KEYS = frozenset({"split_sha256", "manifest_sha256", "splits_sha256"})
SEALED_CHECK_KEYS = frozenset({"group_split_sealed"})
GREEN_VERDICTS = frozenset({"GO", "PASS", "OK", "GREEN"})
HEX64 = re.compile(r"^[0-9a-f]{64}$")

#: Fallbacks used only when the criteria file cannot be read at all; the
#: scan then says so in `criteria_source` instead of pretending it read it.
FALLBACK_COHERENCE = {
    "cohen_kappa_floor": 0.10,
    "require_model_version": True,
    "require_sealed_split": True,
}


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- traversal

def walk(obj, path: str = ""):
    """Yield every (json-pointer-ish path, key, value) leaf and node."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            here = f"{path}/{key}"
            yield here, key, value
            yield from walk(value, here)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            here = f"{path}[{i}]"
            yield from walk(value, here)


def find_keys(obj, keys) -> list[tuple[str, object]]:
    return [(p, v) for p, k, v in walk(obj) if k in keys]


# ------------------------------------------------------------- green claims

def green_claim(doc) -> dict:
    """Does this artifact claim to be green, and on what evidence?

    A top-level `pass` is authoritative: `pass: false` is not green no matter
    what nested sub-check passed. Only when there is no top-level `pass` does
    a `*verdict*` key decide.
    """
    if isinstance(doc, dict) and "pass" in doc:
        value = doc["pass"]
        return {"green": value is True, "by": "pass", "value": value}
    verdicts = [(p, v) for p, k, v in walk(doc)
                if "verdict" in k and isinstance(v, str)]
    for path, value in verdicts:
        if value.strip().upper() in GREEN_VERDICTS:
            return {"green": True, "by": path, "value": value}
    if verdicts:
        return {"green": False, "by": verdicts[0][0], "value": verdicts[0][1]}
    return {"green": False, "by": None, "value": None,
            "note": "no pass and no verdict: publishes numbers, claims nothing"}


def metrics_in(doc) -> dict:
    """The metric keys this artifact publishes, with example locations."""
    quality, latency = {}, {}
    for path, key, value in walk(doc):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if key in QUALITY_METRIC_KEYS:
            quality.setdefault(key, []).append(path)
        elif key in LATENCY_METRIC_KEYS:
            latency.setdefault(key, []).append(path)
    trim = lambda d: {k: v[:4] for k, v in d.items()}  # noqa: E731
    return {"quality": trim(quality), "latency": trim(latency),
            "n_quality": sum(len(v) for v in quality.values()),
            "n_latency": sum(len(v) for v in latency.values())}


def split_seal(doc, root: Path | None = None) -> dict:
    """Is the split this artifact measured on sealed by a content digest?"""
    root = root or ROOT
    seals, bad = [], []
    for path, key, value in walk(doc):
        if key in SEAL_KEYS and isinstance(value, str) and HEX64.match(value):
            seals.append({"at": path, "key": key, "sha256": value})
        elif key in SEALED_CHECK_KEYS and isinstance(value, dict):
            if value.get("pass") is True:
                seals.append({"at": path, "key": key, "check": "passed"})
    # Any referenced in-repo split manifest must still hash to what was
    # recorded next to it: a stale seal is not a seal.
    for path, key, value in walk(doc):
        if key != "manifest" or not isinstance(value, str):
            continue
        if "artifacts/splits/" not in value:
            continue
        target = root / value
        recorded = None
        for p2, k2, v2 in walk(doc):
            if k2 == "manifest_sha256" and p2.rsplit("/", 1)[0] == path.rsplit("/", 1)[0]:
                recorded = v2
                break
        if not target.exists():
            bad.append({"manifest": value, "why": "referenced but missing"})
        elif recorded and sha256_file(target) != recorded:
            bad.append({"manifest": value, "why": "sha256 mismatch",
                        "recorded": recorded, "actual": sha256_file(target)})
    return {"sealed": bool(seals) and not bad, "seals": seals[:6],
            "broken": bad}


# ------------------------------------------------------------------- rules

def check_document(doc, name: str, coherence: dict,
                   root: Path | None = None) -> dict:
    """Apply C1/C2/C3 to a single parsed artifact."""
    claim = green_claim(doc)
    metrics = metrics_in(doc)
    has_mv = bool([v for _, v in find_keys(doc, {"model_version"})
                   if isinstance(v, str) and v.strip()])
    kappas = [(p, v) for p, v in find_keys(doc, KAPPA_KEYS)
              if isinstance(v, (int, float)) and not isinstance(v, bool)]
    seal = split_seal(doc, root)
    return {"file": name, "claim": claim, "metrics": metrics,
            "model_version": has_mv, "kappa": kappas, "seal": seal}


def check_gate_dir(path: Path, coherence: dict | None = None,
                   root: Path | None = None) -> dict:
    """Apply C1/C2/C3 to one `artifacts/gates/<task>/` directory."""
    coherence = coherence or FALLBACK_COHERENCE
    root = root or ROOT
    floor = coherence["cohen_kappa_floor"]["value"] \
        if isinstance(coherence.get("cohen_kappa_floor"), dict) \
        else coherence.get("cohen_kappa_floor", 0.10)

    docs, unreadable = {}, []
    for f in sorted(path.glob("*.json")):
        if f.name == MARKER_NAME:
            continue
        try:
            docs[f.name] = json.loads(f.read_text())
        except (ValueError, OSError) as exc:
            unreadable.append({"file": f.name, "error": str(exc)})

    facts = {n: check_document(d, n, coherence, root) for n, d in docs.items()}
    green = {n: f for n, f in facts.items() if f["claim"]["green"]}
    # The directory is one publication: its numbers are everything it holds.
    all_kappa = [(n, p, v) for n, f in facts.items() for p, v in f["kappa"]]
    any_mv = any(f["model_version"] for f in facts.values())
    any_seal = any(f["seal"]["sealed"] for f in facts.values())
    broken_seals = [(n, b) for n, f in facts.items() for b in f["seal"]["broken"]]
    n_quality = sum(f["metrics"]["n_quality"] for f in facts.values())
    n_latency = sum(f["metrics"]["n_latency"] for f in facts.values())

    errors = []
    if green:
        claimed_by = sorted(green)
        low = [{"file": n, "at": p, "cohen_kappa": v}
               for n, p, v in all_kappa if v <= floor]
        if low:
            errors.append({
                "rule": "C1", "severity": "error",
                "why": f"green claim with cohen_kappa <= {floor}: agreement "
                       "indistinguishable from chance cannot sit next to a PASS",
                "green_claimed_by": claimed_by, "offending": low[:8]})
        if (n_quality or n_latency) and not any_mv:
            errors.append({
                "rule": "C2", "severity": "error",
                "why": "green claim publishes metrics with no model_version: "
                       "the numbers are not attributable to any checkpoint",
                "green_claimed_by": claimed_by,
                "metrics": {n: f["metrics"] for n, f in facts.items()
                            if f["metrics"]["n_quality"]
                            or f["metrics"]["n_latency"]}})
        if n_quality and not any_seal:
            errors.append({
                "rule": "C3", "severity": "error",
                "why": "green claim publishes quality metrics with no sealed "
                       "split: a seed is not a seal, it does not say which "
                       "rows were actually scored",
                "green_claimed_by": claimed_by,
                "quality_metrics": sorted(
                    {k for f in facts.values()
                     for k in f["metrics"]["quality"]})})
        if broken_seals:
            errors.append({
                "rule": "C3", "severity": "error",
                "why": "green claim references a split manifest that is "
                       "missing or no longer hashes to its recorded digest",
                "green_claimed_by": claimed_by,
                "broken": [{"file": n, **b} for n, b in broken_seals][:8]})

    try:
        rel = str(path.relative_to(root))
    except ValueError:
        rel = str(path)
    return {
        "gate": path.name,
        "path": rel,
        "files": sorted(docs),
        "unreadable": unreadable,
        "green": {n: f["claim"] for n, f in green.items()} or None,
        "publishes": {"quality_metrics": n_quality, "latency_metrics": n_latency},
        "model_version": any_mv,
        "sealed_split": any_seal,
        "cohen_kappa": [{"file": n, "at": p, "value": v}
                        for n, p, v in all_kappa],
        "errors": errors,
        "pass": not errors,
    }


# ---------------------------------------------------------------- scanning

def invalid_marker(result: dict, criteria_sha: str | None) -> dict:
    return {
        "format": 1,
        "marked_by": "T-release-gate",
        "marked_utc": utcnow(),
        "gate": result["gate"],
        "rules": ".meshkore/docs/release-criteria.md §5 (C1/C2/C3)",
        "criteria_sha": criteria_sha,
        "verdict": "INVALID — published green against a coherence rule",
        "kept": "this artifact is MARKED, not deleted: a bad published "
                "number is part of the record",
        "green_claimed_by": sorted(result["green"] or {}),
        "errors": result["errors"],
    }


def scan(gates_dir: Path | None = None, coherence: dict | None = None,
         mark: bool = False, criteria_sha: str | None = None,
         root: Path | None = None) -> dict:
    gates_dir = gates_dir or GATES_DIR
    root = root or ROOT
    results, marked = [], []
    for d in sorted(p for p in gates_dir.iterdir() if p.is_dir()):
        if not any(d.glob("*.json")):
            continue
        r = check_gate_dir(d, coherence, root)
        results.append(r)
        marker = d / MARKER_NAME
        if r["errors"]:
            if mark:
                marker.write_text(
                    json.dumps(invalid_marker(r, criteria_sha), indent=2) + "\n")
                marked.append(str(marker.relative_to(root)))
        elif mark and marker.exists():
            marker.unlink()
    failed = [r for r in results if r["errors"]]
    return {
        "format": 1,
        "generated_utc": utcnow(),
        "rules": "C1 kappa<=floor · C2 metric without model_version · "
                 "C3 unsealed split — all ERRORS, never warnings",
        "criteria_sha": criteria_sha,
        "n_gates": len(results),
        "n_invalid": len(failed),
        "invalid": sorted(r["gate"] for r in failed),
        "marked": marked,
        "results": results,
        "pass": not failed,
    }


# -------------------------------------------------------------------- CLI

def _coherence_from_criteria():
    """Thresholds come from the criteria file, never from constants here."""
    try:
        from . import release_gate as RG
        crit = RG.load_criteria()
        return crit["criteria"]["coherence"], crit["criteria_sha"], \
            str(CRITERIA_MD.relative_to(ROOT))
    except Exception as exc:  # noqa: BLE001 — reported, not swallowed
        return FALLBACK_COHERENCE, None, f"fallback ({exc})"


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(prog="eval.gate_rules")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="apply C1/C2/C3 to every published gate")
    s.add_argument("--mark", action="store_true",
                   help=f"write {MARKER_NAME} next to each offending gate")
    s.add_argument("--out", default="")
    c = sub.add_parser("check", help="apply C1/C2/C3 to one gate directory")
    c.add_argument("dir")
    args = ap.parse_args(argv)

    coherence, criteria_sha, source = _coherence_from_criteria()
    if args.cmd == "check":
        r = check_gate_dir(Path(args.dir).resolve(), coherence)
        print(json.dumps(r, indent=2))
        return 0 if r["pass"] else 1

    report = scan(coherence=coherence, mark=args.mark,
                  criteria_sha=criteria_sha)
    report["criteria_source"] = source
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pass": report["pass"], "n_gates": report["n_gates"],
                      "n_invalid": report["n_invalid"],
                      "invalid": report["invalid"],
                      "marked": report["marked"]}, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
