"""Refuse commits that put build output or big blobs back in git (#T-repo-clean).

The repository shipped 9 145 files of `target/` and ~250 MiB of per-run
pickles inside its history. Untracking them is a one-off; keeping them out
is a standing job, which is this script.

Two rules, both evaluated against what git would actually carry — the index
or a commit, never loose files in the working tree:

  1. size — no blob over `--max-bytes` (default 5 MiB).
  2. ignored — no path that `.gitignore` already claims. A file only reaches
     the index against an ignore rule via `git add -f`, so this catches the
     deliberate re-add that rule 1 misses when the file happens to be small
     (a 2 KiB `target/…/lib-foo.json`, of which there were thousands).

Rule 2 reads `.gitignore` through `git check-ignore`, so the two never drift:
add a rule there and this check enforces it on the next commit.

Modes:
    (default)          every path in the index — what a fresh clone gets.
    --staged           only paths staged on top of HEAD; for a pre-commit hook.
    --range A..B       paths added or modified by each commit in the range;
                       for CI on a push with enough history fetched.

Usage:
    python tools/repo_size_check.py
    python tools/repo_size_check.py --staged
    python tools/repo_size_check.py --range origin/main..HEAD
    python tools/repo_size_check.py --json artifacts/gates/T-repo-clean/size.json

Exit: 0 clean · 1 violations found · 2 the check could not run.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

DEFAULT_MAX_BYTES = 5 * 1024 * 1024


class CheckError(Exception):
    """The check itself could not run (not a repo, bad range, …) — exit 2."""


def git(repo: str, *args: str, check: bool = True) -> str:
    proc = subprocess.run(("git", "-C", repo) + args,
                          capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise CheckError(f"git {' '.join(args)}: {proc.stderr.strip()}")
    return proc.stdout


def _blob_sizes(repo: str, shas: list[str]) -> dict[str, int]:
    """Size of every blob, in one `cat-file --batch-check` round-trip."""
    if not shas:
        return {}
    uniq = sorted(set(shas))
    proc = subprocess.run(("git", "-C", repo, "cat-file", "--batch-check"),
                          input="\n".join(uniq) + "\n",
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise CheckError(f"git cat-file: {proc.stderr.strip()}")
    sizes: dict[str, int] = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        # "<sha> missing" for anything this repo does not have (shallow clone).
        if len(parts) == 3 and parts[1] == "blob":
            sizes[parts[0]] = int(parts[2])
    return sizes


def _with_sizes(repo: str, entries: list[tuple[str, str]]) -> list[tuple[str, int]]:
    """[(path, blob_sha)] -> [(path, bytes)], dropping blobs we do not have."""
    sizes = _blob_sizes(repo, [sha for _, sha in entries])
    return [(path, sizes[sha]) for path, sha in entries if sha in sizes]


def index_entries(repo: str) -> list[tuple[str, str]]:
    out = git(repo, "ls-files", "-s", "-z")
    entries = []
    for record in out.split("\0"):
        if not record:
            continue
        meta, path = record.split("\t", 1)
        _mode, sha, _stage = meta.split()
        entries.append((path, sha))
    return entries


def staged_entries(repo: str) -> list[tuple[str, str]]:
    has_head = subprocess.run(("git", "-C", repo, "rev-parse", "--verify", "-q", "HEAD"),
                              capture_output=True, text=True).returncode == 0
    # --abbrev=40: `--raw` abbreviates SHAs by default and cat-file would
    # then answer with the full one, so the size lookup would never match.
    args = ["diff", "--cached", "--diff-filter=AM", "--raw", "-z", "--abbrev=40"]
    if not has_head:  # first commit: everything staged is new
        return index_entries(repo)
    return _parse_raw(git(repo, *args))


def range_entries(repo: str, rev_range: str) -> list[tuple[str, str]]:
    revs = git(repo, "rev-list", rev_range).split()
    entries: list[tuple[str, str]] = []
    for rev in revs:
        entries.extend(_parse_raw(git(
            repo, "diff-tree", "-r", "-m", "--no-commit-id", "--root",
            "--diff-filter=AM", "--raw", "-z", "--abbrev=40", rev)))
    return entries


def _parse_raw(out: str) -> list[tuple[str, str]]:
    """Parse `--raw -z`: ':mode mode shaA shaB STATUS\\0path\\0' records."""
    fields = out.split("\0")
    entries: list[tuple[str, str]] = []
    i = 0
    while i < len(fields):
        meta = fields[i]
        if not meta.startswith(":"):
            i += 1
            continue
        parts = meta.split()
        dst_sha = parts[3]
        path = fields[i + 1] if i + 1 < len(fields) else ""
        if path and dst_sha and set(dst_sha) != {"0"}:
            entries.append((path, dst_sha))
        i += 2
    return entries


def ignored_paths(repo: str, paths: list[str]) -> set[str]:
    """Which of `paths` .gitignore claims — `--no-index` so tracked files count."""
    if not paths:
        return set()
    proc = subprocess.run(
        ("git", "-C", repo, "check-ignore", "--no-index", "--stdin", "-z"),
        input="\0".join(paths) + "\0", capture_output=True, text=True)
    if proc.returncode not in (0, 1):  # 1 = nothing matched, which is the good case
        raise CheckError(f"git check-ignore: {proc.stderr.strip()}")
    return {p for p in proc.stdout.split("\0") if p}


def check(repo: str, mode: str, rev_range: str | None,
          max_bytes: int) -> list[dict]:
    if mode == "staged":
        entries = staged_entries(repo)
    elif mode == "range":
        entries = range_entries(repo, rev_range or "")
    else:
        entries = index_entries(repo)

    sized = _with_sizes(repo, entries)
    ignored = ignored_paths(repo, sorted({path for path, _ in sized}))

    violations: list[dict] = []
    for path, size in sorted(set(sized)):
        if size > max_bytes:
            violations.append({
                "path": path, "rule": "too-large", "bytes": size,
                "limit_bytes": max_bytes,
                "detail": f"{size / 1048576:.2f} MiB > {max_bytes / 1048576:.2f} MiB limit",
            })
        if path in ignored:
            violations.append({
                "path": path, "rule": "ignored-path", "bytes": size,
                "limit_bytes": max_bytes,
                "detail": ".gitignore already excludes this path — it can only "
                          "be here via `git add -f`",
            })
    return violations


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="repo_size_check.py",
        description="Fail when a commit adds a big blob or an ignored path.")
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), help="repository root (default: this repo)")
    ap.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES,
                    help=f"per-file limit in bytes (default {DEFAULT_MAX_BYTES})")
    ap.add_argument("--staged", action="store_true",
                    help="check only what is staged on top of HEAD")
    ap.add_argument("--range", dest="rev_range", metavar="A..B",
                    help="check what the commits in this range add or modify")
    ap.add_argument("--json", dest="json_out", metavar="PATH",
                    help="also write the report as JSON")
    args = ap.parse_args(argv)

    if args.staged and args.rev_range:
        ap.error("--staged and --range are mutually exclusive")
    mode = "staged" if args.staged else "range" if args.rev_range else "index"

    try:
        violations = check(args.repo, mode, args.rev_range, args.max_bytes)
    except CheckError as exc:
        print(f"repo-size-check: cannot run: {exc}", file=sys.stderr)
        return 2

    scope = args.rev_range if mode == "range" else mode
    if violations:
        print(f"repo-size-check: {len(violations)} violation(s) in {scope}\n")
        for v in violations:
            print(f"  {v['rule']:<13} {v['path']}\n"
                  f"  {'':<13} {v['detail']}")
        print("\nBig artifacts belong outside git — register them in "
              ".meshkore/docs/source-register.md\n"
              "(repo, revision, sha256, where the bytes live) and let "
              ".gitignore keep them out.")
    else:
        print(f"repo-size-check: clean ({scope}, limit "
              f"{args.max_bytes / 1048576:.0f} MiB)")

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"scope": scope, "max_bytes": args.max_bytes,
                       "pass": not violations, "violations": violations},
                      fh, indent=2, sort_keys=True)
            fh.write("\n")

    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
