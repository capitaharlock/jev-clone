"""The release gate: applies `.meshkore/docs/release-criteria.md` mechanically
(#T-release-gate, finding F of the 2026-09-21 audit).

The criterion was written BEFORE the measurement it judges, and it lives in a
document, not in this file. Every threshold this module applies is parsed out
of `.meshkore/docs/release-criteria.md`; there is not a single quality
constant here. That is the whole design: a threshold in code can be nudged in
the same commit that makes it pass and nobody sees it; a threshold in a signed,
hashed document cannot. Each verdict publishes the `criteria_sha` of the exact
file it applied, so a criterion moved after the fact is detectable.

No human judgement sits in the middle. The gate reads the criteria, reads the
evidence artifacts of #T-unseen-labels and #T-pointer-head, and emits GO or
NO-GO. Three standing rules:

* **Missing evidence is a FAILED criterion**, never a skipped one. "We have not
  measured the teacher agreement yet" must not read the same as "we measured it
  and it was fine".
* **The coherence rules of §5 gate the gate.** A release cannot be built on
  evidence that itself publishes a green against C1/C2/C3.
* **An unsigned criterion still produces a verdict**, marked
  `criteria_signed: false`. Signing pins the criterion; it does not enable it.

A NO-GO is a correct outcome of this gate, not a failure of it.

CLI (stdlib only, any python3, run from the repo root):

    python3 -m eval.release_gate gate        # -> artifacts/gates/T-release-gate/
    python3 -m eval.release_gate criteria    # show what was parsed + its sha
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import gate_rules as R  # noqa: E402

TASK = "T-release-gate"
CRITERIA_MD = ROOT / ".meshkore" / "docs" / "release-criteria.md"
GATE_DIR = ROOT / "artifacts" / "gates" / TASK
GATE_PATH = GATE_DIR / "gate.json"
AUDIT_PATH = GATE_DIR / "coherence-audit.json"

MACHINE_MARKER = "<!-- criteria-machine-block -->"
SIGNATURE_MARKER = "<!-- criteria-signature -->"
FENCE = re.compile(r"^```[a-zA-Z]*\s*$")


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------ the criterion

def _fenced_after(lines: list[str], marker: str) -> str:
    """The first fenced block following `marker`."""
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == marker)
    except StopIteration:
        raise ValueError(f"release-criteria.md has no {marker}") from None
    opened = None
    body: list[str] = []
    for ln in lines[start + 1:]:
        if opened is None:
            if FENCE.match(ln):
                opened = True
            continue
        if ln.strip().startswith("```"):
            return "\n".join(body)
        body.append(ln)
    raise ValueError(f"unterminated fenced block after {marker}")


def _parse_signature(block: str) -> dict:
    """A three-key YAML block, parsed without a YAML dependency."""
    sig: dict = {}
    for ln in block.splitlines():
        if not ln.strip() or ln.strip().startswith("#"):
            continue
        if ":" not in ln:
            continue
        key, _, raw = ln.partition(":")
        key, raw = key.strip(), raw.strip()
        if key == "signature" and not raw:
            continue
        if raw in ("null", "~", ""):
            sig[key] = None
        elif raw.lower() in ("true", "false"):
            sig[key] = raw.lower() == "true"
        else:
            try:
                sig[key] = int(raw)
            except ValueError:
                sig[key] = raw.strip("'\"")
    return sig


def load_criteria(path: Path | None = None) -> dict:
    """Parse the criterion out of the markdown and seal it with its sha256."""
    path = Path(path) if path else CRITERIA_MD
    raw = path.read_bytes()
    lines = raw.decode("utf-8").splitlines()
    criteria = json.loads(_fenced_after(lines, MACHINE_MARKER))
    signature = _parse_signature(_fenced_after(lines, SIGNATURE_MARKER))
    signed = bool(signature.get("signed_by")
                  and signature["signed_by"] != "pending-operator"
                  and signature.get("signed_utc"))
    for name, spec in criteria.get("thresholds", {}).items():
        if not isinstance(spec, dict) or not str(spec.get("why", "")).strip():
            raise ValueError(f"threshold {name!r} has no `why`: a number "
                             "without a reason is not a criterion")
    return {
        "criteria": criteria,
        "criteria_sha": hashlib.sha256(raw).hexdigest(),
        "criteria_path": str(path.relative_to(ROOT)) if ROOT in path.parents
                         else str(path),
        "criteria_version": criteria.get("criteria_version"),
        "signature": signature,
        "criteria_signed": signed,
    }


# -------------------------------------------------------------- evaluation

def _get(doc, pointer: str):
    """`a.b.c` lookup; None when any hop is absent."""
    cur = doc
    for part in pointer.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _outcome(name: str, spec: dict, ok: bool | None, measured,
             detail: str, **extra) -> dict:
    """`ok is None` means evidence absent, which counts as FAILED."""
    return {
        "criterion": name,
        "threshold": spec.get("value"),
        "why_this_number": spec.get("why"),
        "measured": measured,
        "pass": bool(ok),
        "evidence": "present" if ok is not None else "ABSENT",
        "detail": detail,
        **extra,
    }


def _cut_is_exempt(unseen: dict, cut: str) -> str | None:
    cuts = _get(unseen, "criteria.paired_seen_unseen_table.cuts") or {}
    return (cuts.get(cut) or {}).get("exempt")


def evaluate(criteria: dict, unseen: dict | None, latency: dict | None,
             teacher: dict | None) -> list[dict]:
    """Every threshold in the criterion, applied to the evidence."""
    th = criteria["thresholds"]
    out: list[dict] = []
    cuts = (unseen or {}).get("headline", {}).get("cuts", {})
    overall = (unseen or {}).get("headline", {}).get("overall", {})
    table = (unseen or {}).get("table", {})

    # --- 3.1 accuracy on labels never seen ------------------------------
    spec = th["unseen_accuracy_all_min"]
    acc_all = overall.get("accuracy_unseen")
    out.append(_outcome(
        "unseen_accuracy_all_min", spec,
        None if acc_all is None else acc_all >= spec["value"], acc_all,
        "ALL cut, unseen labels, uncalibrated accuracy"))

    spec = th["unseen_accuracy_cut_min"]
    per_cut, worst = {}, None
    for cut, row in sorted(cuts.items()):
        if cut == "ALL" or _cut_is_exempt(unseen or {}, cut):
            continue
        value = row.get("accuracy", [None, None])[1]
        per_cut[cut] = value
        if value is not None and (worst is None or value < worst[1]):
            worst = (cut, value)
    out.append(_outcome(
        "unseen_accuracy_cut_min", spec,
        None if not per_cut else all(v is not None and v >= spec["value"]
                                     for v in per_cut.values()),
        per_cut, "every non-exempt cut on its own",
        worst_cut=worst))

    spec = th["unseen_above_chance_every_cut"]
    chance = {c: r.get("unseen_beats_chance") for c, r in sorted(cuts.items())
              if c != "ALL" and not _cut_is_exempt(unseen or {}, c)}
    out.append(_outcome(
        "unseen_above_chance_every_cut", spec,
        None if not chance else all(chance.values()), chance,
        "CI95 lower bound above the cut's own chance rate",
        below_chance=sorted(c for c, v in chance.items() if not v)))

    spec = th["seen_unseen_accuracy_drop_max"]
    drop = overall.get("accuracy_drop")
    out.append(_outcome(
        "seen_unseen_accuracy_drop_max", spec,
        None if drop is None else drop <= spec["value"], drop,
        "accuracy_seen - accuracy_unseen on the ALL cut"))

    # --- 3.2 calibration -------------------------------------------------
    for name, cut in (("unseen_ece_max", "unseen"), ("seen_ece_max", "seen")):
        spec = th[name]
        ece = _get(table, f"ALL.{cut}.calibrated.ece")
        out.append(_outcome(
            name, spec, None if ece is None else ece <= spec["value"], ece,
            f"calibrated ECE of the {cut} cut, ALL",
            raw_ece=_get(table, f"ALL.{cut}.raw.ece"),
            fit_cut=_get(unseen or {}, "calibration.fit_cut")))

    # --- 3.3 agreement with the teacher ----------------------------------
    spec = th["teacher_cohen_kappa_min"]
    kappa = _get(teacher or {}, "cohen_kappa")
    out.append(_outcome(
        "teacher_cohen_kappa_min", spec,
        None if kappa is None else kappa >= spec["value"], kappa,
        "Cohen kappa of the release candidate against the distillation "
        "teacher on the unseen cut",
        evidence_artifact=criteria["evidence"].get("teacher_agreement"),
        note=None if kappa is not None else
        "no artifact in the repo measures the trained checkpoint against the "
        "teacher; by §2 absent evidence is a FAILED criterion, not a skipped "
        "one. The repo's only kappa (T-gold, 0.0) is synthetic-pilot "
        "inter-annotator agreement, not model-vs-teacher"))

    # --- 3.4 coverage at fixed risk --------------------------------------
    spec = th["coverage_min_at_risk"]
    curve = _get(table, f"ALL.unseen.risk_coverage.{spec['strategy']}.curve")
    reached, best = None, None
    if curve:
        under = [p for p in curve if p["risk"] <= spec["risk_max"]]
        reached = max((p["coverage"] for p in under), default=0.0)
        best = min(p["risk"] for p in curve)
    out.append(_outcome(
        "coverage_min_at_risk", spec,
        None if curve is None else reached >= spec["value"], reached,
        f"max coverage on the unseen cut at selective risk <= "
        f"{spec['risk_max']} with the `{spec['strategy']}` strategy",
        risk_max=spec["risk_max"], strategy=spec["strategy"],
        best_risk_on_curve=best))

    # --- 3.5 latency ------------------------------------------------------
    spec = th["latency_p95_max_ms"]
    pointer = criteria["evidence"]["latency"]["pointer"]
    p95 = _get(latency or {}, pointer)
    attributed = bool([v for _, v in R.find_keys(latency or {},
                                                 {"model_version"})
                       if isinstance(v, str) and v.strip()])
    ok = None if p95 is None else (p95 <= spec["value"] and attributed)
    out.append(_outcome(
        "latency_p95_max_ms", spec, ok, p95,
        f"{pointer} of {criteria['evidence']['latency']['artifact']}",
        under_budget=None if p95 is None else p95 <= spec["value"],
        model_version_present=attributed,
        note=None if attributed else
        "the number is under budget but the artifact names no model_version: "
        "by C2 an unattributable metric cannot sustain a release"))

    # --- 3.6 required behaviour on `unknown` ------------------------------
    spec = th["unknown_abstain_band"]
    lo, hi = spec["value"]
    abstain = _get(table, "ALL.unseen.raw.abstain_rate")
    out.append(_outcome(
        "unknown_abstain_band", spec,
        None if abstain is None else lo <= abstain <= hi, abstain,
        f"abstention rate on the unseen cut must sit inside [{lo}, {hi}]"))

    spec = th["unknown_ranks_best"]
    rc = _get(table, "ALL.unseen.risk_coverage") or {}
    risks = {s: v.get("avg_selective_risk") for s, v in rc.items()
             if isinstance(v, dict)}
    unknown_risk = risks.get("unknown")
    rivals = {s: v for s, v in risks.items() if s != "unknown" and v is not None}
    out.append(_outcome(
        "unknown_ranks_best", spec,
        None if unknown_risk is None or not rivals
        else all(unknown_risk <= v for v in rivals.values()),
        risks, "avg_selective_risk of `unknown` vs every rival strategy"))

    spec = th["unknown_beats_random"]
    beats = _get(table, "ALL.unseen.risk_coverage.unknown.beats_random")
    out.append(_outcome(
        "unknown_beats_random", spec,
        None if beats is None else bool(beats) is spec["value"], beats,
        "the `unknown` risk-coverage curve must beat random selection"))

    return out


# ------------------------------------------------------------------- gate

def _load_json(rel: str | None) -> dict | None:
    if not rel:
        return None
    path = ROOT / rel
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except ValueError:
        return None


def compose(loaded: dict, results: list[dict], audit: dict,
            evidence_sha: dict) -> dict:
    failed = [r["criterion"] for r in results if not r["pass"]]
    absent = [r["criterion"] for r in results if r["evidence"] == "ABSENT"]
    blocking = sorted(audit["invalid"])
    # The gate cannot be greener than the evidence it stands on.
    evidence_gates = {"T-unseen-labels", "T-pointer-head"}
    tainted = sorted(evidence_gates & set(blocking))
    go = not failed and not tainted
    if go:
        verdict = "GO — every criterion of release-criteria.md v%s passes" % \
            loaded["criteria_version"]
    else:
        parts = []
        if failed:
            parts.append("failed [%s]" % ", ".join(failed))
        if absent:
            parts.append("absent evidence [%s]" % ", ".join(absent))
        if tainted:
            parts.append("evidence marked incoherent [%s]" % ", ".join(tainted))
        verdict = "NO-GO — " + "; ".join(parts)
    return {
        "format": 1,
        "task": TASK,
        "generated_utc": utcnow(),
        "verdict": "GO" if go else "NO-GO",
        "verdict_line": verdict,
        "criteria_path": loaded["criteria_path"],
        "criteria_sha": loaded["criteria_sha"],
        "criteria_version": loaded["criteria_version"],
        "criteria_signed": loaded["criteria_signed"],
        "signature": loaded["signature"],
        "signature_note": "an unsigned criterion still produces a verdict: "
                          "signing pins the criterion, it does not enable it",
        "model_version": _get(evidence_sha, "unseen.model_version"),
        "evidence": evidence_sha,
        "n_criteria": len(results),
        "n_failed": len(failed),
        "failed_criteria": failed,
        "absent_evidence": absent,
        "criteria_results": results,
        "coherence": {
            "rules": audit["rules"],
            "n_gates": audit["n_gates"],
            "n_invalid": audit["n_invalid"],
            "invalid": blocking,
            "blocking_this_release": tainted,
            "artifact": str(AUDIT_PATH.relative_to(ROOT)),
        },
        "honesty": "thresholds are read from the markdown, never from "
                   "constants in the code; numbers are read from the "
                   "artifacts of #T-unseen-labels and #T-pointer-head "
                   "unchanged. A NO-GO is the correct outcome of this gate, "
                   "not a failure of it",
        "pass": go,
    }


def run(criteria_path: Path | None = None, write: bool = True,
        mark: bool = True) -> dict:
    loaded = load_criteria(criteria_path)
    criteria = loaded["criteria"]
    ev = criteria["evidence"]
    unseen_rel = ev["unseen"]
    latency_rel = ev["latency"]["artifact"]
    teacher_rel = ev.get("teacher_agreement")

    unseen = _load_json(unseen_rel)
    latency = _load_json(latency_rel)
    teacher = _load_json(teacher_rel)

    results = evaluate(criteria, unseen, latency, teacher)
    audit = R.scan(coherence=criteria["coherence"], mark=mark,
                   criteria_sha=loaded["criteria_sha"])
    audit["criteria_source"] = loaded["criteria_path"]

    evidence_sha = {}
    for name, rel, doc in (("unseen", unseen_rel, unseen),
                           ("latency", latency_rel, latency),
                           ("teacher", teacher_rel, teacher)):
        path = ROOT / rel if rel else None
        evidence_sha[name] = {
            "path": rel,
            "present": doc is not None,
            "sha256": R.sha256_file(path) if path and path.exists() else None,
            "model_version": (doc or {}).get("model_version"),
        }

    gate = compose(loaded, results, audit, evidence_sha)
    if write:
        GATE_DIR.mkdir(parents=True, exist_ok=True)
        AUDIT_PATH.write_text(json.dumps(audit, indent=2) + "\n")
        GATE_PATH.write_text(json.dumps(gate, indent=2) + "\n")
    return gate


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(prog="eval.release_gate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate", help="apply the criterion and write gate.json")
    g.add_argument("--criteria", default="")
    g.add_argument("--no-write", action="store_true")
    g.add_argument("--no-mark", action="store_true",
                   help="do not write coherence-invalid.json markers")
    c = sub.add_parser("criteria", help="show the parsed criterion and its sha")
    c.add_argument("--criteria", default="")
    args = ap.parse_args(argv)

    path = Path(args.criteria) if args.criteria else None
    if args.cmd == "criteria":
        loaded = load_criteria(path)
        print(json.dumps({k: v for k, v in loaded.items()
                          if k != "criteria"} | {
            "thresholds": {n: s.get("value") for n, s
                           in loaded["criteria"]["thresholds"].items()}},
            indent=2))
        return 0

    gate = run(path, write=not args.no_write, mark=not args.no_mark)
    print(json.dumps({
        "verdict": gate["verdict"],
        "verdict_line": gate["verdict_line"],
        "criteria_sha": gate["criteria_sha"],
        "criteria_signed": gate["criteria_signed"],
        "failed_criteria": gate["failed_criteria"],
        "absent_evidence": gate["absent_evidence"],
        "coherence": {k: gate["coherence"][k]
                      for k in ("n_gates", "n_invalid", "invalid")},
    }, indent=2))
    # A NO-GO is a verdict, not a crash: exit 0 unless the gate could not be
    # computed. The coherence rules DO fail the process (rule 3).
    return 1 if gate["coherence"]["n_invalid"] else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
