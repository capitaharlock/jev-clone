"""Group-wise sealed splits: skeleton + domain + language (#T-split-domain).

Finding C of the 2026-09-21 audit: every split in this repo that used the
row INDEX (``i % 10``) put the same template in train and in test. With
258 700 rows grown out of 190 skeletons that yields acc 1,0000 and
logloss 0,00074 — a leak, not capacity.

This module is the replacement. The unit of a split is never a row: it is
the **group** ``(normalized skeleton, domain, language)``. A group falls
whole into train or whole into test, never into both.

- ``normalize_skeleton`` strips the surface variation a template generator
  injects: digits, amounts, dates, months, weekdays, URLs, e-mails and
  proper nouns become placeholders. Two rows minted from the same template
  collapse to the same skeleton.
- ``group_split_of`` is the deterministic, seeded, index-free primitive the
  data generators call instead of ``i % 10``.
- ``seal_split`` writes ``artifacts/splits/<name>/{train,test}.ids`` plus a
  manifest carrying seed, skeleton version and the sha256 of every ids
  file. ``verify_split`` refuses a split whose bytes moved: a split is
  never silently regenerated.
- ``audit_file`` is the retroactive audit — per dataset: distinct
  skeletons, largest group, and the train<->test Jaccard of the split the
  file was SHIPPED with, contrasted against the group split.

``data/firewall.py`` and ``data/leakage.py`` are reused as they are: the
audit's exact layer is ``leakage.text_hash`` and its paraphrase layer is
``leakage.LeakageDetector`` (word-3-gram Jaccard). This module adds the
split that uses them, it does not re-implement them.

CLI (run from the repo root, any python3):

    python3 -m eval.splits seal   --file <jsonl> --name <name> [--force]
    python3 -m eval.splits verify --name <name>
    python3 -m eval.splits audit  [--max-rows N]      # -> artifacts/gates/
    python3 -m eval.splits scan                       # index-modulo scanner
    python3 -m eval.splits contrast --file <jsonl>    # needs sklearn
    python3 -m eval.splits gate                       # audit + scan -> gate.json
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.leakage import LeakageDetector, jaccard, normalize, text_hash  # noqa: E402

SPLIT_VERSION = 1
SKELETON_VERSION = 1
DEFAULT_SEED = 20260921
DEFAULT_TEST_FRAC = 0.2
DEFAULT_CALIB_FRAC = 0.0

SPLITS_DIR = ROOT / "artifacts" / "splits"
GATE_DIR = ROOT / "artifacts" / "gates" / "T-split-domain"
PREFETCH = ROOT / "artifacts" / "data-prefetch"
QUARANTINE = PREFETCH / "quarantine" / "synth-loop-20260921.jsonl"

# Every published number that rested on the row-index split is invalid as of
# this date. Written into the dashboard and into the affected artifacts.
INVALIDATED_ON = "2026-09-21"
INVALID_REASON = (
    "split by row index (i % 10): the same skeleton appeared in train and "
    "in test, so the metric measured leakage, not capacity "
    "(#T-split-domain, audit 2026-09-21 finding C)"
)
# Datasets whose shipped split was assigned by row index.
INDEX_SPLIT_DATASETS = ("synth-loop", "email-triage")

SPLIT_WORDS = {"train", "test", "calibration", "valid", "validation", "ood"}


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- skeleton -------------------------------------------------------------

_URL = re.compile(r"https?://\S+|\bwww\.\S+", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_TIME = re.compile(r"\b\d{1,2}[:h]\d{2}\b")
_DATE = re.compile(r"\b\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b")
_AMOUNT = re.compile(
    r"[€$£¥]\s?\d[\d.,]*|\b\d[\d.,]*\s?(?:[€$£¥]|eur\b|euros?\b|usd\b|"
    r"dollars?\b|pounds?\b|gbp\b)",
    re.I,
)
_NUM = re.compile(r"\b\d[\d.,]*\b")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
# A capitalized token: first letter upper, rest letters/apostrophes.
_CAP = re.compile(r"\b[^\W\d_][^\W\d_'’.-]*\b", re.UNICODE)
_OPENERS = re.compile(r"[\s\"'“”‘’(\[«»¿¡]+$")
_SENT_END = ".!?:;-–—\n"

MONTHS = {
    "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
    "septiembre", "setiembre", "octubre", "noviembre", "diciembre",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "gener", "febrer", "març", "maig", "juny", "juliol", "setembre",
    "octubre", "novembre", "desembre",
}
WEEKDAYS = {
    "lunes", "martes", "miércoles", "miercoles", "jueves", "viernes",
    "sábado", "sabado", "domingo",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday",
}

# A line whose capitalized share is above this is treated as title-cased
# (English headlines): capitalization carries no proper-noun signal there,
# so masking it would collapse unrelated rows into one fake group. Short
# lines are exempt — a 4-word subject is not a headline.
TITLE_CASE_RATIO = 0.6
TITLE_CASE_MIN_TOKENS = 6


def _looks_title_cased(line: str) -> bool:
    toks = _WORD.findall(line)
    if len(toks) < TITLE_CASE_MIN_TOKENS:
        return False
    caps = sum(1 for t in toks if t[:1].isupper())
    return caps / len(toks) >= TITLE_CASE_RATIO


def _mask_proper_nouns(line: str) -> str:
    """Replace proper nouns with ``<name>`` inside one line.

    Heuristic, deliberately conservative — it may merge two groups, which
    only makes the split stricter; what it must not do is SPLIT one
    template across groups, which is what re-opens the leak. A capitalized
    token is a proper noun when it is not sentence-initial, or when it is
    sentence-initial and followed by a comma (the vocative: "Jordi, ...").
    """
    if _looks_title_cased(line):
        return line

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if not tok[:1].isupper():
            return tok
        if tok.lower() in MONTHS or tok.lower() in WEEKDAYS:
            return tok  # handled by the month/weekday pass
        before = _OPENERS.sub("", line[: m.start()])
        initial = not before or before[-1] in _SENT_END
        after = line[m.end() : m.end() + 1]
        if initial and after != ",":
            return tok
        return "<name>"

    return _CAP.sub(repl, line)


def normalize_skeleton(text: str) -> str:
    """The row's template: surface variation replaced by placeholders."""
    lines = [_mask_proper_nouns(ln) for ln in (text or "").split("\n")]
    out = "\n".join(lines)
    out = _URL.sub(" <url> ", out)
    out = _EMAIL.sub(" <email> ", out)
    out = _AMOUNT.sub(" <amount> ", out)
    out = _DATE.sub(" <date> ", out)
    out = _TIME.sub(" <time> ", out)
    out = _NUM.sub(" <num> ", out)
    out = normalize(out)  # data/leakage.py: lowercase + whitespace collapse
    out = " ".join(
        "<month>" if w in MONTHS else "<day>" if w in WEEKDAYS else w
        for w in out.split()
    )
    return out


# --- language -------------------------------------------------------------

_LOCALE_TAG = re.compile(r"^\s*\[([a-zA-Z]{2})[-_][A-Za-z]{2}\]")
STOPWORDS = {
    "es": {"de", "la", "el", "que", "y", "en", "los", "del", "las", "por",
           "con", "una", "para", "tu", "su", "no", "es", "un"},
    "en": {"the", "of", "and", "to", "in", "is", "that", "for", "it", "on",
           "was", "with", "as", "you", "your", "are"},
    "fr": {"le", "la", "les", "des", "une", "est", "pour", "dans", "avec",
           "vous", "que", "qui"},
    "de": {"der", "die", "das", "und", "ist", "nicht", "mit", "für", "sie",
           "ein", "eine", "auf"},
    "it": {"il", "la", "che", "di", "per", "una", "sono", "con", "non",
           "del", "alle", "questa"},
    "pt": {"o", "a", "de", "que", "para", "não", "uma", "com", "do", "da",
           "você"},
}
LANG_MIN_HITS = 2


def detect_language(text: str, default: str = "und") -> str:
    """Locale tag first (MASSIVE ships ``[it-IT] …``), else stopwords."""
    m = _LOCALE_TAG.match(text or "")
    if m:
        return m.group(1).lower()
    words = set(normalize(text or "").split())
    best, best_hits = default, 0
    for lang, sw in STOPWORDS.items():
        hits = len(words & sw)
        if hits > best_hits:
            best, best_hits = lang, hits
    return best if best_hits >= LANG_MIN_HITS else default


# --- group key ------------------------------------------------------------

@dataclass(frozen=True)
class GroupKey:
    skeleton: str
    domain: str
    lang: str

    def digest(self) -> str:
        blob = f"{SKELETON_VERSION}|{self.domain}|{self.lang}|{self.skeleton}"
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def group_key(text: str, domain: str, lang: str | None = None) -> GroupKey:
    return GroupKey(
        skeleton=normalize_skeleton(text),
        domain=domain,
        lang=lang if lang is not None else detect_language(text),
    )


def group_digest(text: str, domain: str, lang: str | None = None) -> str:
    return group_key(text, domain, lang).digest()


def split_of_digest(
    digest: str,
    seed: int = DEFAULT_SEED,
    test_frac: float = DEFAULT_TEST_FRAC,
    calib_frac: float = DEFAULT_CALIB_FRAC,
) -> str:
    """Deterministic split of ONE group. Never sees a row index.

    The hash is taken over ``seed|digest`` only, so a group keeps its side
    when the corpus grows — no row ever migrates across the fence.
    """
    h = hashlib.sha256(f"{seed}|{digest}".encode()).hexdigest()[:16]
    u = int(h, 16) / float(1 << 64)
    if u < test_frac:
        return "test"
    if u < test_frac + calib_frac:
        return "calibration"
    return "train"


def group_split_of(
    text: str,
    domain: str,
    lang: str | None = None,
    seed: int = DEFAULT_SEED,
    test_frac: float = DEFAULT_TEST_FRAC,
    calib_frac: float = DEFAULT_CALIB_FRAC,
) -> str:
    """Split primitive for data generators — the replacement for ``i % 10``."""
    return split_of_digest(
        group_digest(text, domain, lang), seed, test_frac, calib_frac
    )


# --- corpus iteration -----------------------------------------------------

def row_id(row: dict, domain: str, ordinal: int) -> str:
    qs = row.get("questions") or []
    if qs and isinstance(qs[0], dict) and qs[0].get("id"):
        return str(qs[0]["id"])
    if row.get("id"):
        return str(row["id"])
    return f"{domain}#{ordinal}"


def iter_rows(path: Path, domain: str, max_rows: int | None = None):
    """Yield ``(row_id, state, shipped_split)`` from a universal-schema JSONL."""
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f):
            if max_rows is not None and n >= max_rows:
                return
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield row_id(row, domain, n), row.get("state", ""), row.get("split")


# --- building + sealing a split -------------------------------------------

@dataclass
class SplitResult:
    name: str
    domain: str
    seed: int
    test_frac: float
    calib_frac: float
    ids: dict[str, list[str]] = field(default_factory=dict)
    groups: dict[str, set[str]] = field(default_factory=dict)
    group_rows: dict[str, int] = field(default_factory=dict)
    samples: dict[str, str] = field(default_factory=dict)

    def counts(self) -> dict:
        return {
            "rows": {k: len(v) for k, v in sorted(self.ids.items())},
            "groups": {k: len(v) for k, v in sorted(self.groups.items())},
            "groups_total": len(self.group_rows),
        }

    def overlap(self) -> dict:
        tr, te = self.groups.get("train", set()), self.groups.get("test", set())
        shared = tr & te
        return {
            "shared_groups": len(shared),
            "jaccard": round(jaccard(tr, te), 6),
            "examples": sorted(shared)[:5],
        }


def build_split(
    rows,
    domain: str,
    name: str | None = None,
    seed: int = DEFAULT_SEED,
    test_frac: float = DEFAULT_TEST_FRAC,
    calib_frac: float = DEFAULT_CALIB_FRAC,
    keep_samples: int = 50_000,
) -> SplitResult:
    """Assign every row to a split THROUGH its group. Rows never vote."""
    res = SplitResult(name or domain, domain, seed, test_frac, calib_frac)
    for rid, state, _shipped in rows:
        key = group_key(state, domain)
        dig = key.digest()
        side = split_of_digest(dig, seed, test_frac, calib_frac)
        res.ids.setdefault(side, []).append(rid)
        res.groups.setdefault(side, set()).add(dig)
        res.group_rows[dig] = res.group_rows.get(dig, 0) + 1
        if dig not in res.samples and len(res.samples) < keep_samples:
            res.samples[dig] = key.skeleton[:200]
    for side in res.ids:
        res.ids[side].sort()
    return res


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def seal_split(
    result: SplitResult,
    source: Path | None = None,
    source_rows: int | None = None,
    root: Path | None = None,
    force: bool = False,
    source_sha: str | None = None,
) -> dict:
    """Write ``<root>/<name>/{train,test}.ids`` + manifest.json.

    Refuses to overwrite a sealed split unless ``force`` — a split that
    regenerates itself silently is how a fence moves without anybody
    noticing.
    """
    base = (root or SPLITS_DIR) / result.name
    manifest_path = base / "manifest.json"
    if manifest_path.exists() and not force:
        raise FileExistsError(
            f"split {result.name!r} is already sealed at {manifest_path}; "
            "pass force=True (CLI: --force) to re-seal it on purpose"
        )
    base.mkdir(parents=True, exist_ok=True)
    files = {}
    for side, ids in sorted(result.ids.items()):
        fn = f"{side}.ids"
        payload = "".join(f"{i}\n" for i in ids)
        (base / fn).write_text(payload, encoding="utf-8")
        files[fn] = {"rows": len(ids), "sha256": sha256_text(payload)}
    body = {
        "name": result.name,
        "task": "T-split-domain",
        "split_version": SPLIT_VERSION,
        "skeleton_version": SKELETON_VERSION,
        "created_at": utcnow(),
        "seed": result.seed,
        "grouped_by": ["skeleton", "domain", "language"],
        "fracs": {"test": result.test_frac, "calibration": result.calib_frac},
        "domain": result.domain,
        "source": {
            "path": (str(source.relative_to(ROOT)) if source and
                     str(source).startswith(str(ROOT)) else
                     (str(source) if source else None)),
            "sha256": source_sha if source_sha is not None else (
                sha256_file(source) if source else None),
            "rows": source_rows,
        },
        "counts": result.counts(),
        "overlap": result.overlap(),
        "files": files,
    }
    body["manifest_sha256"] = sha256_text(json.dumps(body, sort_keys=True))
    manifest_path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body


def load_split(name: str, root: Path | None = None) -> dict:
    base = (root or SPLITS_DIR) / name
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    ids = {}
    for fn in manifest["files"]:
        side = fn.rsplit(".", 1)[0]
        ids[side] = [
            ln for ln in (base / fn).read_text(encoding="utf-8").splitlines() if ln
        ]
    return {"manifest": manifest, "ids": ids}


def verify_split(name: str, root: Path | None = None, strict: bool = True) -> dict:
    """Recompute every sha256 in the manifest. A sealed split is immutable."""
    base = (root or SPLITS_DIR) / name
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if sha256_text(json.dumps(body, sort_keys=True)) != manifest.get("manifest_sha256"):
        errors.append(f"{name}: manifest_sha256 mismatch (manifest was edited)")
    for fn, meta in manifest["files"].items():
        path = base / fn
        if not path.exists():
            errors.append(f"{name}: {fn} missing")
            continue
        got = sha256_text(path.read_text(encoding="utf-8"))
        if got != meta["sha256"]:
            errors.append(f"{name}: {fn} sha256 mismatch (sealed bytes changed)")
    if errors and strict:
        raise ValueError("; ".join(errors))
    return {"name": name, "ok": not errors, "errors": errors, "manifest": manifest}


# --- retroactive audit ----------------------------------------------------

PARAPHRASE_SAMPLE = 300
PARAPHRASE_SIM = 0.5


def _paraphrase_overlap(
    train_skeletons: list[str],
    test_skeletons: list[str],
    sim: float = PARAPHRASE_SIM,
) -> dict:
    """Layer 2 of the contamination detector, reused as it is."""
    if not train_skeletons or not test_skeletons:
        return {"checked": 0, "hits": 0, "rate": 0.0, "sim_threshold": sim}
    detector = LeakageDetector(train_skeletons, sim_threshold=sim)
    hits = sum(1 for t in test_skeletons if detector.scan(t)[0])
    return {
        "checked": len(test_skeletons),
        "hits": hits,
        "rate": round(hits / len(test_skeletons), 6),
        "sim_threshold": sim,
    }


def audit_file(
    path: Path,
    domain: str,
    max_rows: int | None = 200_000,
    seed: int = DEFAULT_SEED,
    test_frac: float = DEFAULT_TEST_FRAC,
    paraphrase_sample: int = PARAPHRASE_SAMPLE,
) -> dict:
    """Per dataset: skeletons, largest group, shipped-split leakage."""
    rows_scanned = 0
    group_rows: dict[str, int] = {}
    samples: dict[str, str] = {}
    shipped_groups: dict[str, set[str]] = {}
    shipped_rows: dict[str, int] = {}
    shipped_hashes: dict[str, set[str]] = {}
    new_groups: dict[str, set[str]] = {}
    new_rows: dict[str, int] = {}
    test_rows_seen_in_train_skeleton = 0
    train_digests: set[str] = set()

    # pass 1 collects the shipped train skeletons so pass 2 can count the
    # test rows whose skeleton was already in train (the actual leak rate).
    for _rid, state, shipped in iter_rows(path, domain, max_rows):
        rows_scanned += 1
        key = group_key(state, domain)
        dig = key.digest()
        group_rows[dig] = group_rows.get(dig, 0) + 1
        if dig not in samples and len(samples) < 50_000:
            samples[dig] = key.skeleton[:200]
        side = shipped if shipped in SPLIT_WORDS else "unlabelled"
        shipped_groups.setdefault(side, set()).add(dig)
        shipped_rows[side] = shipped_rows.get(side, 0) + 1
        if side == "train":
            train_digests.add(dig)
            shipped_hashes.setdefault("train", set()).add(text_hash(key.skeleton))
        new = split_of_digest(dig, seed, test_frac)
        new_groups.setdefault(new, set()).add(dig)
        new_rows[new] = new_rows.get(new, 0) + 1

    for _rid, state, shipped in iter_rows(path, domain, max_rows):
        if shipped == "test" and group_digest(state, domain) in train_digests:
            test_rows_seen_in_train_skeleton += 1

    largest = max(group_rows.items(), key=lambda kv: kv[1], default=("", 0))
    top = sorted(group_rows.items(), key=lambda kv: -kv[1])[:5]
    tr = shipped_groups.get("train", set())
    te = shipped_groups.get("test", set())
    n_test_rows = shipped_rows.get("test", 0)

    # sorted(): set iteration order is process-random for strings, and an
    # audit artifact that cannot be reproduced is not evidence.
    tr_sk = [samples[d] for d in sorted(tr)[:paraphrase_sample] if d in samples]
    te_sk = [samples[d] for d in sorted(te)[:paraphrase_sample] if d in samples]
    exact_layer = len(
        {text_hash(s) for s in te_sk} & shipped_hashes.get("train", set())
    )

    ntr, nte = new_groups.get("train", set()), new_groups.get("test", set())
    return {
        "dataset": domain,
        "path": str(path.relative_to(ROOT)) if str(path).startswith(str(ROOT))
                else str(path),
        "rows_scanned": rows_scanned,
        "truncated": max_rows is not None and rows_scanned >= max_rows,
        "skeletons": len(group_rows),
        "skeleton_version": SKELETON_VERSION,
        "largest_group": {
            "rows": largest[1],
            "share": round(largest[1] / rows_scanned, 6) if rows_scanned else 0.0,
            "skeleton": samples.get(largest[0], "")[:160],
        },
        "top_groups": [
            {"rows": n, "skeleton": samples.get(d, "")[:120]} for d, n in top
        ],
        "shipped_split": {
            "rows": dict(sorted(shipped_rows.items())),
            "groups": {k: len(v) for k, v in sorted(shipped_groups.items())},
            "shared_groups_train_test": len(tr & te),
            "jaccard_train_test": round(jaccard(tr, te), 6),
            "test_rows_whose_skeleton_is_in_train": test_rows_seen_in_train_skeleton,
            "test_row_leak_rate": round(
                test_rows_seen_in_train_skeleton / n_test_rows, 6
            ) if n_test_rows else None,
            "exact_layer_shared_skeletons": exact_layer,
            "paraphrase_layer": _paraphrase_overlap(tr_sk, te_sk),
        },
        "group_split": {
            "seed": seed,
            "test_frac": test_frac,
            "rows": dict(sorted(new_rows.items())),
            "groups": {k: len(v) for k, v in sorted(new_groups.items())},
            "shared_groups_train_test": len(ntr & nte),
            "jaccard_train_test": round(jaccard(ntr, nte), 6),
        },
    }


def corpus_files() -> list[tuple[str, Path]]:
    """Every converted dataset of the corpus, quarantine included."""
    out = []
    if PREFETCH.exists():
        for p in sorted(PREFETCH.glob("*.jsonl")):
            if p.name.endswith(".raw.jsonl"):
                continue
            out.append((p.stem, p))
    if QUARANTINE.exists():
        out.append(("synth-loop", QUARANTINE))
    return out


def run_audit(max_rows: int | None = 200_000, seed: int = DEFAULT_SEED) -> dict:
    datasets = []
    for domain, path in corpus_files():
        datasets.append(audit_file(path, domain, max_rows=max_rows, seed=seed))
    return {
        "task": "T-split-domain",
        "generated_at": utcnow(),
        "skeleton_version": SKELETON_VERSION,
        "split_version": SPLIT_VERSION,
        "seed": seed,
        "max_rows_per_dataset": max_rows,
        "datasets": datasets,
    }


# --- index-modulo scanner (the CI grep) -----------------------------------

HASH_MARKERS = ("hashlib", "sha1", "sha256", "sha512", "md5", "blake2",
                "hexdigest", "digest", "crc32", "hash(", "_hash", "hash_")
SCAN_SKIP_DIRS = {".git", "target", "node_modules", "__pycache__", ".venv",
                  ".venv-train", ".pytest_cache", ".ruff_cache", "tmp",
                  "artifacts", ".meshkore"}


# An explicit, greppable opt-out for the ONE legitimate case: a fixture that
# reproduces the legacy leak on purpose. The gate lists every exemption, so a
# production split cannot quietly hide behind it.
ALLOW_PRAGMA = "index-split-fixture"


@dataclass
class ModuloViolation:
    path: str
    line: int
    function: str
    snippet: str


class _ModuloSplitVisitor(ast.NodeVisitor):
    """Flag a split literal decided by a modulo that is not hash-derived."""

    def __init__(self, src: str, path: str) -> None:
        self.src = src
        self.lines = src.splitlines()
        self.path = path
        self.func = "<module>"
        self.tests: list[ast.expr] = []
        self.assigned: dict[str, ast.expr] = {}
        self.violations: list[ModuloViolation] = []
        self.exemptions: list[ModuloViolation] = []

    def _exempt(self, lineno: int) -> bool:
        line = self.lines[lineno - 1] if 0 < lineno <= len(self.lines) else ""
        return ALLOW_PRAGMA in line

    # -- helpers
    def _seg(self, node: ast.AST) -> str:
        try:
            return ast.get_source_segment(self.src, node) or ""
        except Exception:
            return ""

    def _hash_derived(self, node: ast.AST, depth: int = 0) -> bool:
        seg = self._seg(node).lower()
        if any(m in seg for m in HASH_MARKERS):
            return True
        if depth > 3:
            return False
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id in self.assigned:
                if self._hash_derived(self.assigned[sub.id], depth + 1):
                    return True
        return False

    def _modulo_in(self, node: ast.AST) -> ast.BinOp | None:
        for sub in ast.walk(node):
            if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Mod):
                if not self._hash_derived(sub.left):
                    return sub
            if isinstance(sub, ast.Name) and sub.id in self.assigned:
                val = self.assigned[sub.id]
                for s2 in ast.walk(val):
                    if isinstance(s2, ast.BinOp) and isinstance(s2.op, ast.Mod):
                        if not self._hash_derived(s2.left):
                            return s2
        return None

    # -- visits
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        prev_func, prev_assigned = self.func, self.assigned
        self.func, self.assigned = node.name, dict(prev_assigned)
        self.generic_visit(node)
        self.func, self.assigned = prev_func, prev_assigned

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Assign(self, node: ast.Assign) -> None:
        self.generic_visit(node)
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                self.assigned[tgt.id] = node.value

    def visit_If(self, node: ast.If) -> None:
        self.tests.append(node.test)
        for child in node.body:
            self.visit(child)
        self.tests.pop()
        for child in node.orelse:
            self.visit(child)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.tests.append(node.test)
        self.visit(node.body)
        self.visit(node.orelse)
        self.tests.pop()
        self.visit(node.test)

    def visit_Constant(self, node: ast.Constant) -> None:
        if not (isinstance(node.value, str) and node.value in SPLIT_WORDS):
            return
        for test in self.tests:
            mod = self._modulo_in(test)
            if mod is not None:
                if self._exempt(getattr(mod, "lineno", node.lineno)):
                    self.exemptions.append(ModuloViolation(
                        self.path, getattr(mod, "lineno", node.lineno),
                        self.func, self._seg(mod) or self._seg(test)))
                    return
                self.violations.append(ModuloViolation(
                    self.path, getattr(mod, "lineno", node.lineno), self.func,
                    self._seg(mod) or self._seg(test),
                ))
                return


def scan_index_modulo(
    root: Path | None = None, exemptions: list | None = None
) -> list[ModuloViolation]:
    """Zero index-modulo splits in the repo, checked in CI.

    A text grep would trip over every comment that merely NAMES ``i % 10``
    (the quarantine notes do), so the check parses instead: a violation is
    a split literal returned under a condition driven by a modulo whose
    left side is not hash-derived — ``m = i % 10; if m < 8: "train"``.
    """
    base = root or ROOT
    out: list[ModuloViolation] = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in SCAN_SKIP_DIRS]
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            path = Path(dirpath) / fn
            try:
                src = path.read_text(encoding="utf-8")
                tree = ast.parse(src)
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            rel = str(path.relative_to(base)) if str(path).startswith(str(base)) \
                else str(path)
            visitor = _ModuloSplitVisitor(src, rel)
            visitor.visit(tree)
            seen = set()
            for v in visitor.violations:
                key = (v.path, v.line, v.function)
                if key not in seen:
                    seen.add(key)
                    out.append(v)
            if exemptions is not None:
                seen_x = set()
                for v in visitor.exemptions:
                    key = (v.path, v.line, v.function)
                    if key not in seen_x:
                        seen_x.add(key)
                        exemptions.append(v)
    return out


# --- invalidation of the historical numbers -------------------------------

def invalidation_stamp(affects: list[str] | None = None) -> dict:
    return {
        "invalidated_on": INVALIDATED_ON,
        "invalidated_by": "T-split-domain",
        "reason": INVALID_REASON,
        "affects": list(affects or INDEX_SPLIT_DATASETS),
        "replacement": "eval/splits.py — group split by skeleton+domain+language",
    }


def _mentions_index_split(obj) -> list[str]:
    """Datasets in this artifact whose numbers came from the index split."""
    blob = json.dumps(obj, ensure_ascii=False)
    return [d for d in INDEX_SPLIT_DATASETS if f'"{d}"' in blob or f"'{d}'" in blob]


def invalidate_artifacts(paths: list[Path], apply: bool = True) -> list[dict]:
    """Stamp ``invalidated`` into every artifact resting on the index split."""
    touched = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        affects = _mentions_index_split(data)
        if not affects:
            continue
        if data.get("invalidated", {}).get("invalidated_by") == "T-split-domain":
            continue
        stamp = invalidation_stamp(affects)
        if apply:
            data["invalidated"] = stamp
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        touched.append({
            "path": str(path.relative_to(ROOT)) if str(path).startswith(str(ROOT))
                    else str(path),
            "affects": affects,
        })
    return touched


def invalidation_targets() -> list[Path]:
    out: list[Path] = []
    runs = ROOT / "artifacts" / "runs"
    if runs.exists():
        out += sorted(runs.glob("*/metrics.json"))
    gen = ROOT / "artifacts" / "runs-gen"
    if gen.exists():
        out += sorted(gen.glob("*.json"))
    return out


# --- baseline contrast (old i%10 split vs group split) --------------------

CONTRAST_MAX_ROWS = 360_700
CONTRAST_TRAIN_CAP = 80_000
CONTRAST_EVAL_CAP = 20_000


def contrast_baseline(
    path: Path,
    domain: str = "synth-loop",
    max_rows: int = CONTRAST_MAX_ROWS,
    seed: int = DEFAULT_SEED,
    test_frac: float = DEFAULT_TEST_FRAC,
    train_cap: int = CONTRAST_TRAIN_CAP,
    eval_cap: int = CONTRAST_EVAL_CAP,
) -> dict:
    """Same rows, same model, two splits. The contrast IS the proof.

    Arm A reproduces the shipped ``i % 10`` split; arm B uses the group
    split. Identical TF-IDF + LogReg (the historical baseline recipe), so
    any gap between the two accuracies is the split and nothing else.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss

    rows: list[tuple[str, str, str, str]] = []  # state, answer, shipped, group
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f):
            if n >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qs = row.get("questions") or []
            if not qs:
                continue
            ans = qs[0].get("answer")
            if ans is None or ans == "unknown":
                continue
            state = row.get("state", "")
            rows.append((state, ans, row.get("split") or "train",
                         group_digest(state, domain)))

    def arm(kind: str) -> dict:
        Xtr, ytr, Xev, yev = [], [], [], []
        for state, ans, shipped, dig in rows:
            side = shipped if kind == "index" else split_of_digest(
                dig, seed, test_frac)
            if side == "train":
                if len(Xtr) < train_cap:
                    Xtr.append(state)
                    ytr.append(ans)
            elif side == "test":
                if len(Xev) < eval_cap:
                    Xev.append(state)
                    yev.append(ans)
        out = {"split": kind, "n_train": len(ytr), "n_eval": len(yev)}
        if not ytr or not yev:
            out["accuracy"] = out["logloss"] = None
            out["note"] = "empty arm"
            return out
        vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
        Xtr_v = vec.fit_transform(Xtr)
        clf = LogisticRegression(max_iter=200, C=1.0)
        clf.fit(Xtr_v, ytr)
        Xev_v = vec.transform(Xev)
        out["accuracy"] = round(float(accuracy_score(yev, clf.predict(Xev_v))), 6)
        try:
            out["logloss"] = round(float(log_loss(
                yev, clf.predict_proba(Xev_v), labels=list(clf.classes_))), 6)
        except ValueError:
            out["logloss"] = None
        return out

    a, b = arm("index"), arm("group")
    train_digests = {d for _s, _a, sh, d in rows if sh == "train"}
    test_digests = {d for _s, _a, sh, d in rows if sh == "test"}
    return {
        "task": "T-split-domain",
        "generated_at": utcnow(),
        "source": {
            "path": str(path.relative_to(ROOT)) if str(path).startswith(str(ROOT))
                    else str(path),
            "rows_used": len(rows),
        },
        "model": "tfidf-logreg-v1 (data/train_baseline.py recipe)",
        "caps": {"train": train_cap, "eval": eval_cap},
        "before_index_split": a,
        "after_group_split": b,
        "skeletons": len({d for _s, _a, _sh, d in rows}),
        "shipped_split_group_overlap": {
            "shared_groups": len(train_digests & test_digests),
            "jaccard": round(jaccard(train_digests, test_digests), 6),
        },
        "delta_accuracy": (
            round(b["accuracy"] - a["accuracy"], 6)
            if isinstance(a.get("accuracy"), float)
            and isinstance(b.get("accuracy"), float) else None
        ),
    }


# --- gate -----------------------------------------------------------------

def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path


def build_gate(audit: dict, violations: list[ModuloViolation],
               contrast: dict | None, sealed: list[dict],
               exemptions: list[ModuloViolation] | None = None) -> dict:
    per_dataset = {
        d["dataset"]: {
            "skeletons": d["skeletons"],
            "rows_scanned": d["rows_scanned"],
            "largest_group_rows": d["largest_group"]["rows"],
            "shipped_split_jaccard_train_test": d["shipped_split"][
                "jaccard_train_test"],
            "shipped_split_shared_groups": d["shipped_split"][
                "shared_groups_train_test"],
            "group_split_jaccard_train_test": d["group_split"][
                "jaccard_train_test"],
            "group_split_shared_groups": d["group_split"][
                "shared_groups_train_test"],
        }
        for d in audit["datasets"]
    }
    checks = {
        "group_split_zero_overlap": all(
            v["group_split_shared_groups"] == 0 for v in per_dataset.values()),
        "no_index_modulo_split_in_repo": not violations,
        "audit_covers_corpus": bool(per_dataset),
        "splits_sealed": bool(sealed),
        "sealed_splits_verify": all(s.get("ok", False) for s in sealed),
    }
    if contrast:
        a = contrast["before_index_split"].get("accuracy")
        b = contrast["after_group_split"].get("accuracy")
        # The contrast is the proof: same rows, same model, honest split.
        checks["contrast_accuracy_drops"] = (
            isinstance(a, float) and isinstance(b, float) and b < a)
        checks["contrast_subject_sealed_zero_overlap"] = any(
            s_["name"].startswith("synth-loop") and
            s_["overlap"]["shared_groups"] == 0 for s_ in sealed)
    return {
        "task": "T-split-domain",
        "generated_at": utcnow(),
        "pass": all(checks.values()),
        "checks": checks,
        "seed": audit["seed"],
        "skeleton_version": SKELETON_VERSION,
        "split_version": SPLIT_VERSION,
        "per_dataset": per_dataset,
        "index_modulo_exemptions": [
            {"path": v.path, "line": v.line, "function": v.function,
             "snippet": v.snippet, "pragma": ALLOW_PRAGMA}
            for v in (exemptions or [])],
        "index_modulo_violations": [
            {"path": v.path, "line": v.line, "function": v.function,
             "snippet": v.snippet} for v in violations],
        "sealed_splits": sealed,
        "contrast": contrast,
        "invalidated": invalidation_stamp(),
    }


# --- CLI ------------------------------------------------------------------

def _cmd_seal(args) -> int:
    path = Path(args.file).resolve()
    name = args.name or path.stem
    domain = args.domain or name
    rows = iter_rows(path, domain, args.max_rows)
    result = build_split(rows, domain, name=name, seed=args.seed,
                         test_frac=args.test_frac, calib_frac=args.calib_frac)
    total = sum(len(v) for v in result.ids.values())
    manifest = seal_split(result, source=path, source_rows=total,
                          force=args.force)
    print(json.dumps({"sealed": name, "counts": manifest["counts"],
                      "overlap": manifest["overlap"]}, indent=2))
    return 0


def _cmd_verify(args) -> int:
    names = [args.name] if args.name else sorted(
        p.name for p in SPLITS_DIR.glob("*") if (p / "manifest.json").exists())
    bad = 0
    for name in names:
        res = verify_split(name, strict=False)
        bad += 0 if res["ok"] else 1
        print(json.dumps({"name": name, "ok": res["ok"],
                          "errors": res["errors"]}))
    return 1 if bad else 0


def _cmd_audit(args) -> int:
    audit = run_audit(max_rows=args.max_rows, seed=args.seed)
    out = write_json(GATE_DIR / "audit.json", audit)
    print(f"wrote {out}")
    for d in audit["datasets"]:
        print(f"  {d['dataset']:>16}  rows={d['rows_scanned']:>8,}  "
              f"skeletons={d['skeletons']:>8,}  "
              f"largest={d['largest_group']['rows']:>7,}  "
              f"shipped J={d['shipped_split']['jaccard_train_test']:.4f}  "
              f"group J={d['group_split']['jaccard_train_test']:.4f}")
    return 0


def _cmd_scan(args) -> int:
    exemptions: list[ModuloViolation] = []
    violations = scan_index_modulo(exemptions=exemptions)
    for v in exemptions:
        print(f"{v.path}:{v.line}: {v.function}(): exempt fixture "
              f"({ALLOW_PRAGMA}): {v.snippet}")
    for v in violations:
        print(f"{v.path}:{v.line}: {v.function}(): index-modulo split: {v.snippet}")
    print(f"{len(violations)} index-modulo split(s)")
    return 1 if violations else 0


def _cmd_contrast(args) -> int:
    path = Path(args.file).resolve()
    res = contrast_baseline(path, domain=args.domain, max_rows=args.max_rows,
                            seed=args.seed, test_frac=args.test_frac)
    out = write_json(GATE_DIR / "contrast.json", res)
    print(json.dumps(res, indent=2))
    print(f"wrote {out}")
    return 0


def _cmd_invalidate(args) -> int:
    touched = invalidate_artifacts(invalidation_targets(), apply=not args.dry_run)
    payload = {
        "task": "T-split-domain",
        "generated_at": utcnow(),
        **invalidation_stamp(),
        "artifacts": touched,
        "dashboard": "tools/training_monitor.py — banner + per-task flag",
    }
    out = write_json(GATE_DIR / "invalidated.json", payload)
    print(f"{len(touched)} artifact(s) stamped; wrote {out}")
    return 0


def _cmd_gate(args) -> int:
    audit = run_audit(max_rows=args.max_rows, seed=args.seed)
    write_json(GATE_DIR / "audit.json", audit)
    exemptions: list[ModuloViolation] = []
    violations = scan_index_modulo(exemptions=exemptions)
    contrast_path = GATE_DIR / "contrast.json"
    contrast = json.loads(contrast_path.read_text()) if contrast_path.exists() \
        else None
    sealed = []
    if SPLITS_DIR.exists():
        for p in sorted(SPLITS_DIR.glob("*")):
            if (p / "manifest.json").exists():
                res = verify_split(p.name, strict=False)
                sealed.append({"name": p.name, "ok": res["ok"],
                               "errors": res["errors"],
                               "counts": res["manifest"]["counts"],
                               "overlap": res["manifest"]["overlap"]})
    gate = build_gate(audit, violations, contrast, sealed, exemptions)
    out = write_json(GATE_DIR / "gate.json", gate)
    print(json.dumps({"pass": gate["pass"], "checks": gate["checks"]}, indent=2))
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--seed", type=int, default=DEFAULT_SEED)
        p.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
        return p

    p = common(sub.add_parser("seal", help="build + seal a group split"))
    p.add_argument("--file", required=True)
    p.add_argument("--name")
    p.add_argument("--domain")
    p.add_argument("--calib-frac", type=float, default=DEFAULT_CALIB_FRAC)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=_cmd_seal)

    p = sub.add_parser("verify", help="recompute the sealed sha256s")
    p.add_argument("--name")
    p.set_defaults(fn=_cmd_verify)

    p = common(sub.add_parser("audit", help="retroactive per-dataset audit"))
    p.add_argument("--max-rows", type=int, default=200_000)
    p.set_defaults(fn=_cmd_audit)

    p = sub.add_parser("scan", help="index-modulo split scanner (CI)")
    p.set_defaults(fn=_cmd_scan)

    p = common(sub.add_parser("contrast", help="i%%10 vs group split baseline"))
    p.add_argument("--file", default=str(QUARANTINE))
    p.add_argument("--domain", default="synth-loop")
    p.add_argument("--max-rows", type=int, default=CONTRAST_MAX_ROWS)
    p.set_defaults(fn=_cmd_contrast)

    p = sub.add_parser("invalidate", help="stamp the index-split artifacts")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=_cmd_invalidate)

    p = common(sub.add_parser("gate", help="audit + scan -> gate.json"))
    p.add_argument("--max-rows", type=int, default=200_000)
    p.set_defaults(fn=_cmd_gate)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
