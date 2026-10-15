"""Scheduled external evidence scan for the TinyJev research loop.

The service discovers candidates. It never downloads weights, runs external
code, changes training data, or treats external text as gold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "research-loop"
STATE = OUT / "state.json"
EVENTS = OUT / "events.jsonl"
FINDINGS = OUT / "findings.jsonl"
RUNS = OUT / "runs"
REPORTS = OUT / "reports"
USER_AGENT = "TinyJev-research-loop/1.0"
POLL_SECONDS = int(os.environ.get("RESEARCH_LOOP_POLL_SECONDS", "60"))
DAILY_SECONDS = 24 * 3600
WEEKLY_SECONDS = 7 * DAILY_SECONDS
AUDIT_SECONDS = 48 * 3600
STOP = {"value": False}

KEYWORDS = {
    "decision": 4, "decision-making": 4, "preference": 4, "listwise": 5,
    "pairwise": 3, "rerank": 5, "ranking": 3, "abstention": 5,
    "selective prediction": 5, "calibration": 4, "uncertainty": 3,
    "reward model": 4, "multicriteria": 5, "multi-criteria": 5,
    "pointer": 3, "entailment": 2, "synthetic data": 2,
}


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(value: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def request(url: str, accept: str = "application/json") -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": accept})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
            time.sleep(2 ** attempt)
        except urllib.error.URLError:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"request retries exhausted: {url}")


def request_json(url: str) -> object:
    return json.loads(request(url).decode("utf-8"))


def text(node: ET.Element | None, name: str) -> str:
    child = node.find(name) if node is not None else None
    return " ".join((child.text or "").split()) if child is not None else ""


def arxiv(limit: int) -> list[dict]:
    query = ('all:"decision making" OR all:"preference learning" OR '
             'all:reranking OR all:abstention OR all:"reward model"')
    url = ("https://export.arxiv.org/api/query?"
           + urllib.parse.urlencode({
               "search_query": query, "start": 0, "max_results": limit,
               "sortBy": "submittedDate", "sortOrder": "descending"}))
    root = ET.fromstring(request(url, "application/atom+xml"))
    atom = "{http://www.w3.org/2005/Atom}"
    rows = []
    for entry in root.findall(f"{atom}entry"):
        page = text(entry, f"{atom}id")
        rows.append({
            "id": "arxiv:" + page.rsplit("/", 1)[-1],
            "source": "arXiv", "kind": "paper",
            "title": text(entry, f"{atom}title"),
            "summary": text(entry, f"{atom}summary"),
            "url": page, "published": text(entry, f"{atom}published"),
            "updated": text(entry, f"{atom}updated"),
            "revision": page.rsplit("/", 1)[-1], "license": "unknown",
        })
    return rows


def github(limit: int) -> list[dict]:
    query = ('"decision model" OR reranker OR listwise OR '
             '"preference learning" in:name,description,readme')
    url = ("https://api.github.com/search/repositories?"
           + urllib.parse.urlencode({
               "q": query, "sort": "updated", "order": "desc",
               "per_page": limit}))
    payload = request_json(url)
    rows = []
    for item in payload.get("items", []):
        license_info = item.get("license") or {}
        rows.append({
            "id": "github:" + item["full_name"],
            "source": "GitHub", "kind": "repository",
            "title": item["full_name"],
            "summary": item.get("description") or "",
            "url": item["html_url"], "published": item.get("created_at"),
            "updated": item.get("pushed_at"),
            "revision": item.get("default_branch"),
            "license": license_info.get("spdx_id") or "unknown",
            "stars": item.get("stargazers_count"),
        })
    return rows


def huggingface(kind: str, limit: int) -> list[dict]:
    endpoint = "datasets" if kind == "dataset" else "models"
    url = (f"https://huggingface.co/api/{endpoint}?"
           + urllib.parse.urlencode({
               "search": "decision ranking preference",
               "sort": "lastModified", "direction": "-1", "limit": limit}))
    payload = request_json(url)
    rows = []
    for item in payload:
        item_id = item.get("id") or item.get("modelId")
        if not item_id:
            continue
        card = item.get("cardData") or {}
        tags = item.get("tags") or []
        license_id = card.get("license") or next(
            (tag.split(":", 1)[1] for tag in tags
             if isinstance(tag, str) and tag.startswith("license:")), "unknown")
        rows.append({
            "id": f"hf-{kind}:{item_id}",
            "source": "Hugging Face", "kind": kind,
            "title": item_id,
            "summary": " ".join(str(tag) for tag in tags[:20]),
            "url": f"https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{item_id}",
            "published": item.get("createdAt"),
            "updated": item.get("lastModified"),
            "revision": item.get("sha"), "license": license_id,
            "downloads": item.get("downloads"),
        })
    return rows


def jev_pages() -> list[dict]:
    pages = [
        ("jev:blog", "JEv introduction",
         "https://typesafe.ai/blog/introducing-system-one-models-and-jev"),
        ("jev:models", "JEv model documentation",
         "https://docs.typesafe.ai/models"),
    ]
    rows = []
    for item_id, title, url in pages:
        body = request(url, "text/html")
        rows.append({
            "id": item_id, "source": "JEv", "kind": "official",
            "title": title, "summary": "official page content check",
            "url": url, "published": None, "updated": utcnow(),
            "revision": hashlib.sha256(body).hexdigest(),
            "license": "reference-only",
        })
    return rows


SOURCES = {
    "arxiv": arxiv,
    "github": github,
    "huggingface_datasets": lambda n: huggingface("dataset", n),
    "huggingface_models": lambda n: huggingface("model", n),
    "jev": lambda _: jev_pages(),
}


def relevance(item: dict) -> tuple[int, list[str]]:
    haystack = (item.get("title", "") + " " + item.get("summary", "")).lower()
    matched = [key for key in KEYWORDS if key in haystack]
    score = sum(KEYWORDS[key] for key in matched)
    if item.get("license") not in (None, "", "unknown", "other"):
        score += 1
    return score, matched


def known_ids() -> set[str]:
    try:
        return {json.loads(line)["id"] for line in
                FINDINGS.read_text(encoding="utf-8").splitlines() if line}
    except (OSError, ValueError, KeyError):
        return set()


def next_due(completed: str, cadence: str) -> str:
    base = parse_utc(completed) or datetime.now(timezone.utc)
    if cadence == "monthly":
        local = datetime.now().astimezone()
        year, month = local.year, local.month + 1
        if month == 13:
            year, month = year + 1, 1
        slot = local.replace(
            year=year, month=month, day=1, hour=19, minute=0,
            second=0, microsecond=0)
        slot += timedelta(days=(4 - slot.weekday()) % 7)
        return slot.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    seconds = {
        "daily": DAILY_SECONDS,
        "audit": AUDIT_SECONDS,
        "weekly": WEEKLY_SECONDS,
    }[cadence]
    return (base + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")


def due(state: dict, cadence: str) -> bool:
    previous = state.get(cadence) or {}
    scheduled = parse_utc(previous.get("next_due_utc"))
    return scheduled is None or datetime.now(timezone.utc) >= scheduled


def event(kind: str, **detail) -> None:
    append_jsonl(EVENTS, {"at": utcnow(), "event": kind, **detail})


def report_markdown(run: dict) -> str:
    lines = [
        f"# External research {run['cadence']} · {run['completed_utc'][:10]}",
        "",
        f"Status: `{run['status']}`. Sources: {run['sources_ok']}/"
        f"{len(run['sources'])}. Findings: {run['findings']}; "
        f"new: {run['new_findings']}; relevant: {run['relevant_findings']}.",
        "",
        "External material is discovery evidence only. No item entered "
        "training or supplied benchmark gold.",
        "",
        "## Candidates",
        "",
    ]
    for item in run["top_findings"]:
        lines += [
            f"- [{item['title']}]({item['url']}) · {item['source']} / "
            f"{item['kind']} · score {item['score']} · "
            f"license `{item.get('license') or 'unknown'}` · "
            f"revision `{item.get('revision') or 'unknown'}`",
        ]
    lines += ["", "## Source status", ""]
    for name, detail in run["sources"].items():
        suffix = f" · {detail.get('error')}" if detail.get("error") else ""
        lines.append(f"- {name}: `{detail['status']}` · "
                     f"{detail.get('count', 0)} items{suffix}")
    lines += [
        "", "## Decision", "",
        "Review the top candidates against the active error cells. Before an "
        "experiment, record the pinned revision, license, contamination risk, "
        "cost and a falsifiable expected gain. Reject the candidate if it "
        "cannot be compared on a held-out arena.",
        "",
    ]
    return "\n".join(lines)


def run_audit() -> dict:
    started = utcnow()
    event("audit-started")
    loop = ROOT / "artifacts" / "autoloop"
    state = read_json(loop / "state.json")
    champion = read_json(loop / "champion" / "champion.json")
    benchmark = read_json(loop / "champion" / "benchmark_v2_full.json")
    history = []
    try:
        history = [json.loads(line) for line in
                   (loop / "history.jsonl").read_text(
                       encoding="utf-8").splitlines() if line]
    except (OSError, ValueError):
        pass
    scored = [row for row in history if row.get("candidate")]
    last = scored[-1] if scored else {}
    recent = scored[-12:]
    promoted = [row for row in recent if row.get("promoted")]
    cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
    failures = [
        row for row in history if row.get("failed_phase")
        and (parse_utc(row.get("at")) or datetime.min.replace(
            tzinfo=timezone.utc)) >= cutoff
    ]
    champion_mean = (champion.get("summary") or {}).get("mean")
    candidate_mean = (last.get("candidate") or {}).get("mean")
    streak = int(state.get("streak_no_promote") or 0)
    if failures:
        verdict = "correct"
        reason = f"{len(failures)} failed phase(s) in the recent history"
    elif promoted:
        verdict = "continue"
        reason = "a recent candidate replaced the champion"
    elif streak >= 8:
        verdict = "change-arm"
        reason = f"{streak} scored cycles without promotion"
    else:
        verdict = "continue"
        reason = "the promotion window is still open"
    completed = utcnow()
    run = {
        "cadence": "audit", "status": "ok", "verdict": verdict,
        "reason": reason, "started_utc": started,
        "completed_utc": completed, "next_due_utc": next_due(completed, "audit"),
        "cycle": state.get("cycle"), "phase": state.get("phase"),
        "champion_cycle": champion.get("cycle"),
        "champion_mean": champion_mean, "candidate_mean": candidate_mean,
        "streak_no_promote": streak, "recent_scored": len(recent),
        "recent_promoted": len(promoted), "recent_failures": len(failures),
        "benchmark_v2_n": benchmark.get("n"),
        "benchmark_v2_score": benchmark.get("forced"),
    }
    stamp = started.replace(":", "").replace("-", "")
    write_json(RUNS / f"{stamp}-audit.json", run)
    REPORTS.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Results audit · {started[:10]}", "",
        f"Verdict: `{verdict}`. {reason}.", "",
        f"- active cycle: {run['cycle']} / {run['phase']}",
        f"- champion: cycle {run['champion_cycle']}, mean of orders "
        f"{champion_mean}",
        f"- last candidate mean: {candidate_mean}",
        f"- cycles without promotion: {streak}",
        f"- benchmark v2: {run['benchmark_v2_score']} over "
        f"{run['benchmark_v2_n']} cases",
        f"- recent failures: {len(failures)}",
        "",
        "The sealed battery remained closed. This audit can change the next "
        "experimental arm, but cannot promote a checkpoint by itself.",
        "",
    ]
    (REPORTS / f"{started[:10]}-audit.md").write_text(
        "\n".join(lines), encoding="utf-8")
    event("audit-completed", verdict=verdict, reason=reason)
    return run


def run_scan(cadence: str, limit: int | None = None) -> dict:
    limit = limit or (20 if cadence == "daily" else 50)
    started = utcnow()
    event("scan-started", cadence=cadence)
    source_status, items = {}, []
    for name, loader in SOURCES.items():
        try:
            found = loader(limit)
            items.extend(found)
            source_status[name] = {"status": "ok", "count": len(found)}
            event("source-completed", cadence=cadence, source=name,
                  count=len(found))
        except Exception as exc:
            message = f"{type(exc).__name__}: {str(exc)[:240]}"
            source_status[name] = {
                "status": "error", "count": 0, "error": message}
            event("source-failed", cadence=cadence, source=name, error=message)
    by_id = {item["id"]: item for item in items}
    seen = known_ids()
    ranked = []
    for item in by_id.values():
        score, matched = relevance(item)
        item = {**item, "score": score, "matched": matched,
                "discovered_utc": started, "cadence": cadence}
        ranked.append(item)
        if item["id"] not in seen:
            append_jsonl(FINDINGS, item)
    ranked.sort(key=lambda row: (
        row["score"], row.get("updated") or "", row["id"]), reverse=True)
    completed = utcnow()
    ok = sum(1 for source in source_status.values()
             if source["status"] == "ok")
    status = "ok" if ok == len(SOURCES) else ("partial" if ok else "error")
    run = {
        "cadence": cadence, "status": status,
        "started_utc": started, "completed_utc": completed,
        "next_due_utc": (
            (datetime.now(timezone.utc) + timedelta(hours=1))
            .strftime("%Y-%m-%dT%H:%M:%SZ")
            if status == "error" or (
                status == "partial" and cadence == "daily")
            else next_due(completed, cadence)),
        "sources": source_status, "sources_ok": ok,
        "findings": len(ranked),
        "new_findings": sum(item["id"] not in seen for item in ranked),
        "relevant_findings": sum(item["score"] >= 4 for item in ranked),
        "top_findings": ranked[:12],
    }
    stamp = started.replace(":", "").replace("-", "")
    write_json(RUNS / f"{stamp}-{cadence}.json", run)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"{started[:10]}-{cadence}.md").write_text(
        report_markdown(run), encoding="utf-8")
    event("scan-completed", cadence=cadence, status=status,
          findings=run["findings"], new=run["new_findings"])
    return run


def update_state(state: dict, cadence: str, run: dict) -> dict:
    state[cadence] = {
        key: run[key] for key in (
            "status", "started_utc", "completed_utc", "next_due_utc",
            "sources_ok", "findings", "new_findings", "relevant_findings")}
    state["recent_findings"] = run["top_findings"][:8]
    state["last_sources"] = run["sources"]
    state["last_run"] = {
        "cadence": cadence, "status": run["status"],
        "completed_utc": run["completed_utc"],
        "report": f"artifacts/research-loop/reports/"
                  f"{run['started_utc'][:10]}-{cadence}.md",
    }
    return state


def heartbeat(state: dict, status: str = "running") -> None:
    state.update({
        "status": status, "pid": os.getpid(), "updated_utc": utcnow(),
        "policy": {
            "daily_hours": 24, "audit_hours": 48, "weekly_days": 7,
            "auto_ingest": False, "auto_execute_external_code": False,
        },
    })
    write_json(STATE, state)


def cycle(state: dict, force: str | None = None) -> dict:
    cadences = (["daily", "audit", "weekly", "monthly"] if force == "all"
                else [force] if force else
                [name for name in ("daily", "audit", "weekly", "monthly")
                 if due(state, name)])
    for cadence in cadences:
        heartbeat(state, f"scanning-{cadence}")
        if cadence == "audit":
            run = run_audit()
            state["audit"] = run
        else:
            run = run_scan(cadence)
            update_state(state, cadence, run)
        heartbeat(state)
    return state


def on_term(_signum, _frame) -> None:
    STOP["value"] = True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--force", choices=("daily", "audit", "weekly", "monthly", "all"))
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    state = read_json(STATE)
    heartbeat(state)
    while not STOP["value"]:
        state = cycle(state, args.force)
        args.force = None
        if args.once:
            break
        for _ in range(max(1, POLL_SECONDS)):
            if STOP["value"]:
                break
            time.sleep(1)
        heartbeat(state)
    heartbeat(state, "stopped" if STOP["value"] else "idle")
    return 0


if __name__ == "__main__":
    sys.exit(main())
