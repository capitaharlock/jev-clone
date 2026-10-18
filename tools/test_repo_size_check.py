"""Tests for #T-repo-clean's size check (stdlib unittest).

The load-bearing test is `test_ten_mib_file_fails`: it injects a real 10 MiB
file on purpose and asserts the check refuses it. A guard that has never been
seen to fire is not a guard.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import repo_size_check  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEN_MIB = 10 * 1024 * 1024


def _git(repo: str, *args: str) -> str:
    return subprocess.run(
        ("git", "-C", repo,
         "-c", "user.name=test", "-c", "user.email=test@example.com",
         "-c", "commit.gpgsign=false") + args,
        capture_output=True, text=True, check=True).stdout


def _write(repo: str, rel: str, data: bytes) -> str:
    path = os.path.join(repo, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


class SizeCheckTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = self._tmp.name
        _git(self.repo, "init", "-q", "-b", "main")
        _write(self.repo, ".gitignore", b"target/\n*.pkl\n")
        _write(self.repo, "src/main.rs", b"fn main() {}\n")
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-qm", "init")

    def tearDown(self):
        self._tmp.cleanup()

    def run_check(self, *argv: str) -> int:
        return repo_size_check.main(["--repo", self.repo, *argv])

    def violations(self, mode: str = "index", rev_range=None) -> list[dict]:
        return repo_size_check.check(self.repo, mode, rev_range,
                                     repo_size_check.DEFAULT_MAX_BYTES)

    # --- the clean baseline -------------------------------------------------

    def test_clean_repo_passes(self):
        self.assertEqual(self.run_check(), 0)
        self.assertEqual(self.run_check("--staged"), 0)

    def test_untracked_big_file_is_not_a_violation(self):
        """Only what git would carry counts — a loose 10 MiB file is nobody's problem."""
        _write(self.repo, "scratch.bin", b"\0" * TEN_MIB)
        self.assertEqual(self.run_check(), 0)

    # --- rule 1: size -------------------------------------------------------

    def test_ten_mib_file_fails(self):
        _write(self.repo, "artifacts/big.bin", b"\0" * TEN_MIB)
        _git(self.repo, "add", "artifacts/big.bin")

        self.assertEqual(self.run_check("--staged"), 1)
        self.assertEqual(self.run_check(), 1)

        found = [v for v in self.violations("staged")
                 if v["path"] == "artifacts/big.bin"]
        self.assertEqual(len(found), 1, found)
        self.assertEqual(found[0]["rule"], "too-large")
        self.assertEqual(found[0]["bytes"], TEN_MIB)

    def test_ten_mib_file_fails_after_commit(self):
        _write(self.repo, "artifacts/big.bin", b"\0" * TEN_MIB)
        _git(self.repo, "add", "artifacts/big.bin")
        _git(self.repo, "commit", "-qm", "oops")

        self.assertEqual(self.run_check(), 1)
        self.assertEqual(self.run_check("--range", "HEAD~1..HEAD"), 1)
        self.assertEqual([v["path"] for v in self.violations("range", "HEAD~1..HEAD")],
                         ["artifacts/big.bin"])

    def test_limit_is_five_mib(self):
        _write(self.repo, "just_under.bin", b"\0" * (5 * 1024 * 1024))
        _git(self.repo, "add", "just_under.bin")
        self.assertEqual(self.run_check(), 0)

        _write(self.repo, "just_over.bin", b"\0" * (5 * 1024 * 1024 + 1))
        _git(self.repo, "add", "just_over.bin")
        self.assertEqual(self.run_check(), 1)

    # --- rule 2: ignored paths ---------------------------------------------

    def test_force_added_ignored_path_fails_even_when_tiny(self):
        """The real regression: 9 145 small files of `target/`, not one big one."""
        _write(self.repo, "target/debug/lib-foo.json", b"{}\n")
        _git(self.repo, "add", "-f", "target/debug/lib-foo.json")

        self.assertEqual(self.run_check(), 1)
        found = [v for v in self.violations() if v["rule"] == "ignored-path"]
        self.assertEqual([v["path"] for v in found], ["target/debug/lib-foo.json"])

    def test_ignored_suffix_rule_is_enforced(self):
        _write(self.repo, "models/massive.pkl", b"tiny")
        _git(self.repo, "add", "-f", "models/massive.pkl")
        self.assertEqual([v["rule"] for v in self.violations()], ["ignored-path"])

    def test_both_rules_report_separately(self):
        _write(self.repo, "target/blob.bin", b"\0" * TEN_MIB)
        _git(self.repo, "add", "-f", "target/blob.bin")
        rules = sorted(v["rule"] for v in self.violations())
        self.assertEqual(rules, ["ignored-path", "too-large"])

    # --- plumbing -----------------------------------------------------------

    def test_json_report_is_written(self):
        import json
        _write(self.repo, "artifacts/big.bin", b"\0" * TEN_MIB)
        _git(self.repo, "add", "artifacts/big.bin")
        out = os.path.join(self._tmp.name, "report", "size.json")

        self.assertEqual(self.run_check("--json", out), 1)
        with open(out, encoding="utf-8") as fh:
            report = json.load(fh)
        self.assertFalse(report["pass"])
        self.assertEqual(report["violations"][0]["path"], "artifacts/big.bin")

    def test_max_bytes_is_configurable(self):
        _write(self.repo, "small.bin", b"\0" * 2048)
        _git(self.repo, "add", "small.bin")
        self.assertEqual(self.run_check("--max-bytes", "1024"), 1)

    def test_non_repo_exits_two(self):
        with tempfile.TemporaryDirectory() as empty:
            self.assertEqual(repo_size_check.main(["--repo", empty]), 2)

    def test_staged_and_range_are_exclusive(self):
        with self.assertRaises(SystemExit):
            self.run_check("--staged", "--range", "HEAD~1..HEAD")


class ThisRepoTest(unittest.TestCase):
    """#T-repo-clean's gate: this repository itself must stay clean."""

    def test_index_is_clean(self):
        violations = repo_size_check.check(
            REPO_ROOT, "index", None, repo_size_check.DEFAULT_MAX_BYTES)
        self.assertEqual(violations, [], f"{len(violations)} violation(s)")


if __name__ == "__main__":
    unittest.main()
