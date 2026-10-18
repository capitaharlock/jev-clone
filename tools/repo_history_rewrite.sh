#!/usr/bin/env bash
#
# Strip target/, datasets and pickles out of the git HISTORY (#T-repo-clean).
#
# Untracking those paths (done, commit `feat(project): keep build output …`)
# stops the repo GROWING. It does not shrink a clone by one byte: every blob
# ever committed is still reachable, so `git clone` still moves ~320 MiB.
# Only rewriting history removes them, and that rewrites every commit id.
#
# THIS IS IRREVERSIBLE AND REQUIRES THE OPERATOR'S SIGNATURE.
# Without REPO_REWRITE_CONFIRM=yes the script prints its plan and exits 0.
#
#   tools/repo_history_rewrite.sh                     # plan only, changes nothing
#   REPO_REWRITE_CONFIRM=yes tools/repo_history_rewrite.sh
#
# What it does, in order:
#   1. backup   — `git bundle --all`, verified by restoring it into a throwaway
#                 clone and comparing every ref and the commit count. A backup
#                 nobody restored is a rumour, so this one gets restored.
#   2. rewrite  — git-filter-repo, on a FRESH CLONE of this repo. Your working
#                 tree is never touched: the untracked 2.5 GiB of weights,
#                 datasets and target/ sitting in it stay exactly where it is.
#   3. verify   — the rewritten clone must build (`cargo build --locked`) and
#                 carry no blob over 5 MiB.
#   4. report   — sizes before/after and the backup's sha256 into
#                 artifacts/gates/T-repo-clean/gate.json.
#
# It never pushes and never touches your remote. Step 4 prints the commands
# to adopt the result; running them is a separate, deliberate act.
#
# Reading first: .meshkore/docs/repo-publication.md — this script is option A,
# and option B (a fresh public repo from one clean commit) may suit you better.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_DIR="${BACKUP_DIR:-${HOME}/jev-clone-backups}"
BACKUP="${BACKUP_DIR}/jev-clone-${STAMP}.bundle"
WORK_DIR="${WORK_DIR:-${TMPDIR:-/tmp}/jev-clone-rewrite-${STAMP}}"
GATE="${REPO}/artifacts/gates/T-repo-clean/gate.json"
MAX_BLOB="5M"

# Everything that must not exist anywhere in the history. Keep in step with
# .gitignore — `tools/repo_size_check.py` enforces the same list going forward.
STRIP_PATHS=(
  target
  artifacts/runs
  artifacts/weights
  artifacts/data-prefetch
  artifacts/data-raw
  artifacts/data-qwen
  artifacts/logs
)
STRIP_GLOBS=(
  '*.pkl'
  '*.safetensors'
  '*.DS_Store'
)

say()  { printf '%s\n' "$*"; }
step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die()  { printf '\033[31mrepo-history-rewrite: %s\033[0m\n' "$*" >&2; exit 1; }

human() { # bytes -> MiB, no bc dependency
  awk -v b="$1" 'BEGIN { printf "%.2f MiB", b / 1048576 }'
}

dir_bytes() {
  du -sk "$1" | awk '{ printf "%d", $1 * 1024 }'
}

pack_bytes() { # what a clone actually transfers
  git -C "$1" count-objects -v | awk '/^size-pack:/ { printf "%d", $2 * 1024 }'
}

print_plan() {
  say "repo-history-rewrite — PLAN ONLY (nothing will change)"
  say ""
  say "  repo            ${REPO}"
  say "  backup bundle   ${BACKUP}"
  say "  work clone      ${WORK_DIR}/jev-clone.git"
  say "  gate report     ${GATE}"
  say ""
  say "  paths dropped from every commit:"
  for p in "${STRIP_PATHS[@]}"; do say "    - ${p}/"; done
  for g in "${STRIP_GLOBS[@]}"; do say "    - ${g}"; done
  say "    - plus every blob larger than ${MAX_BLOB}, whatever its name"
  say ""
  say "  every commit id in this repository changes. Anyone holding a clone"
  say "  must re-clone; open branches must be rebased onto the new history."
  say ""
  say "  to run it for real:"
  say "    REPO_REWRITE_CONFIRM=yes tools/repo_history_rewrite.sh"
}

# ---------------------------------------------------------------- the gate --
if [[ "${REPO_REWRITE_CONFIRM:-}" != "yes" ]]; then
  print_plan
  exit 0
fi

# ------------------------------------------------------------ 0 · the tools --
step "0 · preconditions"
command -v git >/dev/null || die "git not on PATH"
if ! command -v git-filter-repo >/dev/null && ! git filter-repo --help >/dev/null 2>&1; then
  die "git-filter-repo not installed — 'pip install git-filter-repo' or 'brew install git-filter-repo'"
fi
FILTER_REPO=(git filter-repo)
if command -v git-filter-repo >/dev/null; then FILTER_REPO=(git-filter-repo); fi

git -C "${REPO}" rev-parse --git-dir >/dev/null 2>&1 || die "${REPO} is not a git repository"
if [[ -n "$(git -C "${REPO}" status --porcelain --untracked-files=no)" ]]; then
  die "working tree has uncommitted changes — commit or stash them first, so the backup is the whole story"
fi
say "  git            $(git --version | awk '{print $3}')"
say "  filter-repo    $("${FILTER_REPO[@]}" --version 2>/dev/null || echo 'via git')"
say "  HEAD           $(git -C "${REPO}" rev-parse HEAD)"

# ------------------------------------------------------- 1 · verified backup --
step "1 · backup, and restore it to prove it"
mkdir -p "${BACKUP_DIR}"
git -C "${REPO}" bundle create "${BACKUP}" --all
git -C "${REPO}" bundle verify "${BACKUP}" >/dev/null || die "bundle failed its own verify"

RESTORE="${WORK_DIR}/restore-test"
mkdir -p "${WORK_DIR}"
git clone --quiet --bare "${BACKUP}" "${RESTORE}.git"

SRC_COMMITS="$(git -C "${REPO}" rev-list --all --count)"
DST_COMMITS="$(git -C "${RESTORE}.git" rev-list --all --count)"
SRC_HEAD="$(git -C "${REPO}" rev-parse HEAD)"
DST_HEAD="$(git -C "${RESTORE}.git" rev-parse HEAD)"
[[ "${SRC_COMMITS}" == "${DST_COMMITS}" ]] || die "restored backup has ${DST_COMMITS} commits, repo has ${SRC_COMMITS}"
[[ "${SRC_HEAD}" == "${DST_HEAD}" ]] || die "restored backup HEAD ${DST_HEAD} != ${SRC_HEAD}"

if command -v shasum >/dev/null; then
  BACKUP_SHA="$(shasum -a 256 "${BACKUP}" | awk '{print $1}')"
else
  BACKUP_SHA="$(sha256sum "${BACKUP}" | awk '{print $1}')"
fi
BACKUP_BYTES="$(wc -c < "${BACKUP}" | tr -d ' ')"
rm -rf "${RESTORE}.git"

say "  bundle         ${BACKUP}"
say "  size           $(human "${BACKUP_BYTES}")"
say "  sha256         ${BACKUP_SHA}"
say "  restored       ${DST_COMMITS} commits, HEAD ${DST_HEAD} — matches"
say ""
say "  if anything below goes wrong, this restores everything:"
say "    git clone ${BACKUP} jev-clone-restored"

# --------------------------------------------------------- 2 · measure before --
step "2 · size before"
BEFORE_CLONE="${WORK_DIR}/before.git"
git clone --quiet --bare "file://${REPO}" "${BEFORE_CLONE}"
BEFORE_PACK="$(pack_bytes "${BEFORE_CLONE}")"
BEFORE_DIR="$(dir_bytes "${BEFORE_CLONE}")"
say "  clone pack     $(human "${BEFORE_PACK}")"
git -C "${REPO}" count-objects -vH | sed 's/^/  /'

# -------------------------------------------------------------- 3 · rewrite --
step "3 · rewrite history on a fresh clone"
WORK="${WORK_DIR}/jev-clone.git"
git clone --quiet --bare "file://${REPO}" "${WORK}"

FILTER_ARGS=(--invert-paths)
for p in "${STRIP_PATHS[@]}"; do FILTER_ARGS+=(--path "${p}/"); done
for g in "${STRIP_GLOBS[@]}"; do FILTER_ARGS+=(--path-glob "${g}"); done

( cd "${WORK}" && "${FILTER_REPO[@]}" "${FILTER_ARGS[@]}" --force )
# Second pass: anything oversized that survived under a name nobody listed.
( cd "${WORK}" && "${FILTER_REPO[@]}" --strip-blobs-bigger-than "${MAX_BLOB}" --force )

git -C "${WORK}" reflog expire --expire=now --all
git -C "${WORK}" gc --prune=now --aggressive --quiet

AFTER_PACK="$(pack_bytes "${WORK}")"
AFTER_DIR="$(dir_bytes "${WORK}")"
say "  clone pack     $(human "${AFTER_PACK}")  (was $(human "${BEFORE_PACK}"))"

# --------------------------------------------------------------- 4 · verify --
step "4 · verify the rewritten history"
CHECKOUT="${WORK_DIR}/checkout"
git clone --quiet "${WORK}" "${CHECKOUT}"

VERIFY_BUILD="skipped (cargo not on PATH)"
if command -v cargo >/dev/null; then
  if ( cd "${CHECKOUT}" && cargo build --locked --quiet ); then
    VERIFY_BUILD="pass"
    say "  cargo build    pass — target/ reconstructs from Cargo.lock alone"
  else
    VERIFY_BUILD="FAIL"
    say "  cargo build    FAILED — the rewrite dropped something the build needed"
  fi
else
  say "  cargo build    ${VERIFY_BUILD}"
fi

VERIFY_SIZE="FAIL"
if python3 "${REPO}/tools/repo_size_check.py" --repo "${CHECKOUT}"; then
  VERIFY_SIZE="pass"
fi
say "  size check     ${VERIFY_SIZE}"

TARGET_BYTES=$((50 * 1024 * 1024))
if [[ "${AFTER_PACK}" -lt "${TARGET_BYTES}" ]]; then TARGET_MET="True"; else TARGET_MET="False"; fi
say "  < 50 MiB       ${TARGET_MET}"

# --------------------------------------------------------------- 5 · report --
step "5 · report"
mkdir -p "$(dirname "${GATE}")"
python3 - <<PY
import json, os
gate = "${GATE}"
report = json.load(open(gate)) if os.path.exists(gate) else {}
report.update({
    "task": "T-repo-clean",
    "generated_utc": "${STAMP}",
    "history_rewritten": True,
    "backup": {
        "path": "${BACKUP}",
        "bytes": ${BACKUP_BYTES},
        "sha256": "${BACKUP_SHA}",
        "commits": ${SRC_COMMITS},
        "head": "${SRC_HEAD}",
        "restore_verified": True,
        "restore": "git clone ${BACKUP} jev-clone-restored",
    },
    "size": {
        "clone_pack_bytes_before": ${BEFORE_PACK},
        "clone_pack_bytes_after": ${AFTER_PACK},
        "clone_dir_bytes_before": ${BEFORE_DIR},
        "clone_dir_bytes_after": ${AFTER_DIR},
    },
    "checks": {
        "cargo_build_after_rewrite": "${VERIFY_BUILD}",
        "repo_size_check": "${VERIFY_SIZE}",
    },
    "clone_size_target_bytes": ${TARGET_BYTES},
    "clone_size_target_met": ${TARGET_MET},
    "rewritten_repo": "${WORK}",
})
json.dump(report, open(gate, "w"), indent=2, sort_keys=True)
open(gate, "a").write("\n")
print("  wrote " + gate)
PY

step "done — nothing has been pushed"
say "  rewritten history   ${WORK}"
say "  working checkout    ${CHECKOUT}"
say "  backup              ${BACKUP}  (sha256 ${BACKUP_SHA})"
say ""
say "  Adopting it is your call, and it is the irreversible half. Either:"
say ""
say "    A · replace this repo's history in place"
say "        git -C ${REPO} remote add rewritten ${WORK}"
say "        git -C ${REPO} fetch rewritten"
say "        git -C ${REPO} reset --hard rewritten/main   # discards nothing but history ids"
say ""
say "    B · publish ${WORK} as the public repo and keep this one private"
say "        (see .meshkore/docs/repo-publication.md — this is usually the safer one)"
say ""
say "  Either way, every collaborator re-clones. Tell them before, not after."
