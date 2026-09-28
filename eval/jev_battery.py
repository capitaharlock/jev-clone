"""Jev teacher on the same 400 development rows (#T-teacher-probe).

The dashboard comparison is only honest if Jev is scored on battery-dev,
not on a published BANKING77 citation. This runner spends one API call per
row, caches answers on disk, and stops at a hard dollar cap.

Permutation control is skipped on purpose: it would double the spend and
the operator funded a test balance.

CLI:
    set -a; . .meshkore/credentials/jev.env; set +a
    PYTHONPATH=. python3 -m eval.jev_battery ping
    PYTHONPATH=. python3 -m eval.jev_battery run --budget-usd 0.25
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import metrics_suite as MS  # noqa: E402
from eval import preflight_refs as P  # noqa: E402
from eval import teacher as TE  # noqa: E402

TASK = "T-teacher-probe"
GATE_DIR = ROOT / "artifacts" / "gates" / TASK
GATE_PATH = GATE_DIR / "battery.json"
PICKS_PATH = GATE_DIR / "battery.picks.jsonl"
PERM_PATH = GATE_DIR / "battery.reversed.jsonl"
PERM_SAMPLE = 24
PERM_MAX_USD = 0.01
DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"


def _load_env_file() -> None:
    path = ROOT / ".meshkore" / "credentials" / "jev.env"
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _ensure_endpoint() -> None:
    if not os.environ.get(TE.ENDPOINT_ENV, "").strip():
        os.environ[TE.ENDPOINT_ENV] = DEFAULT_ENDPOINT


def ping() -> dict:
    _load_env_file()
    _ensure_endpoint()
    client = TE.TeacherClient(TE.Budget(max_calls=1, max_usd=0.02))
    out = client.decide(
        "Help! My payouts have failed for 3 days.",
        "Which team should handle this?",
        [{"id": "billing", "text": "Payments, invoicing, refunds"},
         {"id": "technical", "text": "Bugs, outages, integrations"}],
    )
    return {"config": TE.config_card(), "answer": {
        "choice": out.get("choice"),
        "cached": out.get("cached"),
        "unparsed": out.get("unparsed"),
        "off_menu": out.get("off_menu"),
    }, "budget": client.budget.card()}


def run(budget_usd: float, max_calls: int, limit: int | None) -> dict:
    _load_env_file()
    _ensure_endpoint()
    if not TE.configured():
        raise TE.NotConfigured(
            "Jev teacher is not configured: put the key in "
            ".meshkore/credentials/jev-api-key and the endpoint in "
            ".meshkore/credentials/jev.env")
    episodes = P.load_cut()
    if limit:
        episodes = episodes[: int(limit)]
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    done = P._trace_log(PICKS_PATH)
    client = TE.TeacherClient(
        TE.Budget(max_calls=max_calls, max_usd=budget_usd))
    rows = []
    errors = 0
    for i, ep in enumerate(episodes, 1):
        ids = [c["id"] for c in ep["candidates"]]
        k = len(ids)
        seen = done.get(ep["id"])
        if seen is None:
            try:
                got = client.decide(ep["state"], ep["question"],
                                    ep["candidates"], question_id="q")
            except TE.BudgetExceeded:
                break
            pick = got.get("choice")
            seen = {
                "row_id": ep["id"],
                "choice": pick,
                "abstained": pick not in ids,
                "unparsed": bool(got.get("unparsed")),
                "off_menu": bool(got.get("off_menu")),
                "cached": bool(got.get("cached")),
            }
            P._trace_append(PICKS_PATH, seen)
            done[ep["id"]] = seen
        pick = seen.get("choice")
        index = ids.index(pick) if pick in ids else k
        if seen.get("unparsed") or seen.get("off_menu"):
            errors += 1
        rows.append(P.row_of(ep, index, P._onehot(k, index)))
        if i % 25 == 0:
            print(f"[jev] {i}/{len(episodes)} calls={client.budget.calls} "
                  f"cache={client.budget.cached} usd={client.budget.usd:.4f}",
                  flush=True)
    tracking = P.tracking_control(
        [ep for ep in episodes if ep["id"] in done],
        rows,
        source=f"{TASK} Jev battery-dev {P.CUT_NAME}")
    scored_eps = [ep for ep in episodes if ep["id"] in done]
    # The suite will not publish without the permutation control. Same
    # protocol as the Qwen column: a fixed 24-row sample (sha256 order,
    # chosen before measuring), options offered in reverse, pick compared by
    # id. Its own hard budget, separate from the 400 main calls.
    perm_client = TE.TeacherClient(
        TE.Budget(max_calls=PERM_SAMPLE, max_usd=PERM_MAX_USD))

    def choose(state, question, candidates):
        return perm_client.decide(state, question, candidates,
                                  question_id="q")

    picks = {rid: v.get("choice") for rid, v in done.items()}
    permutation = P.qwen_permutation(scored_eps, picks, choose=choose,
                                     sample=PERM_SAMPLE, log=PERM_PATH,
                                     temperature=None)
    permutation["budget"] = perm_client.budget.card()
    doc = MS.report(
        rows,
        cut={**P.cut_descriptor(scored_eps), "subset": "whole cut"},
        model_version=f"jev:{TE.model()}",
        task=TASK,
        tracking=tracking,
        permutation=permutation,
        calibration=P.dev_temperature(rows),
        notes=[
            "Jev teacher scored on battery-dev, same rows as Qwen/Laya/MiniLM",
            f"permutation control on a fixed {PERM_SAMPLE}-row sample, own "
            f"budget ({PERM_SAMPLE} calls, ${PERM_MAX_USD}), as for Qwen",
            f"key_fingerprint {TE.fingerprint()}",
        ])
    gate = {
        "task": TASK,
        "cut": "battery-dev",
        "measured_utc": P.utcnow(),
        "config": TE.config_card(),
        "budget": client.budget.card(),
        "n_scored": len(rows),
        "n_cut": len(episodes),
        "rows_sha256": P.rows_sha(scored_eps),
        "parse_errors": errors,
        "permutation_budget": permutation["budget"],
        "report": MS.require(doc),
    }
    GATE_PATH.write_text(json.dumps(gate, indent=2, ensure_ascii=False) + "\n")
    return gate


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(prog="eval.jev_battery")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ping")
    r = sub.add_parser("run")
    r.add_argument("--budget-usd", type=float, default=0.25)
    r.add_argument("--max-calls", type=int, default=400)
    r.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "ping":
            out = ping()
            print(json.dumps(out, indent=2, ensure_ascii=False))
            return 0 if out["answer"]["choice"] else 1
        gate = run(args.budget_usd, args.max_calls, args.limit)
        rk = (gate["report"] or {}).get("ranking") or {}
        print(json.dumps({
            "n": gate["n_scored"],
            "forced": rk.get("accuracy"),
            "ci95": rk.get("accuracy_ci95"),
            "budget": gate["budget"],
            "path": str(GATE_PATH.relative_to(ROOT)),
        }, indent=2, ensure_ascii=False))
        return 0 if gate["n_scored"] else 1
    except TE.TeacherError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
