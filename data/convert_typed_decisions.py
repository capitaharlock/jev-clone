"""`LocalLLaMA/typed-decisions` (Hugging Face) → `episode-v1` (#T-ingest-laya).

The one public cut where we can sit next to Jev row by row: on its 400
test cases (2 000 decisions) Jev 1.13.0 publishes 0,727 and Laya's
fine-tuned checkpoint 0,766 (`TMP/laya/BENCHMARKS.md`). The train split
(1 200 cases, ≈ 6 000 decisions) is volume for the trainer; the test split
is **eval-only, forever** — every test episode carries `eval_only: true`
and `assert_trainable()` refuses any mixture that names it.

One decision = one episode. What the adapter does, and does NOT do:

* `state`: the case's JSON rendered to readable lines (`key: value`, one
  per line, nested keys dotted, lists of scalars joined with «; », lists
  of objects indexed `key[i].sub`). Nothing is summarised or invented.
* `question`: `questions[q].instructions` verbatim for `choice` and
  `score`. A `noul` instruction is a *statement* («This trace requires
  human review.») and the contract demands a question, so it is rendered
  as `<statement> Is this statement true?` — a rule of the converter,
  recorded per episode in `question_render`, never a hand edit.
* `candidates`: opaque ids `c1..cK` in an order shuffled by seed; `text`
  is the criterion's DESCRIPTION, never the label (Laya's own warning:
  `true`/`false` labels get followed by name). `score` levels read
  «<level> — <description>»; a `noul` without criteria gets the two
  default descriptions in `NOUL_DEFAULT_CRITERIA`.
* `evidence`: the contract requires a literal fragment of the state. The
  dataset annotates none, so the line of the rendered state that overlaps
  most with the case's generating `factors` (plus the gold description as
  tie-break) is taken, literally; `evidence_origin` says so. A case whose
  factors touch no line at all is a reject, not a guess.
* `family`: the contract's enumeration is closed, so `choice`/`score` map
  to `description_classification` (every option carries its own
  definition) and `noul` to `textual_inference_negation` (one hypothesis
  judged against the state). The task's finer key lives in `subfamily`:
  `external/typed-decisions/<workflow>/<type>`.
* `variant_group` = `td-<case id>`: the five questions of one state share
  it — «same situation, different question», the axis the model fails.
* `teacher_soft`: the gold distribution re-keyed by opaque id (Laya's
  soft CE target). Unused until `#T-loop-rl-jev`; kept, not thrown away.

Whatever fails `data.episode_contract.validate` goes to `rejects.jsonl`
with its reasons. Nothing is fixed by hand.

Pipeline (CPU, no torch, no training):

    # 1. download pinned by revision + sha256, decode parquet → raw jsonl
    #    (needs pyarrow, which the hash-locked .venv-train does not carry:
    #    any venv with `pyarrow huggingface_hub` does — the raw jsonl is
    #    then stdlib territory)
    <python-with-pyarrow> -m data.convert_typed_decisions fetch
    # 2. convert both splits (stdlib only)
    PYTHONPATH=. .venv-train/bin/python -m data.convert_typed_decisions convert
    # 3. measure the gate
    PYTHONPATH=. .venv-train/bin/python -m data.convert_typed_decisions gate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from typing import Any, Iterable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import episode_contract as EC  # noqa: E402

TASK = "T-ingest-laya"
CONVERTER_VERSION = "typed-decisions-v1"
ORIGIN = "laya-typed-decisions"
DATASET_ID = "LocalLLaMA/typed-decisions"
CONFIG = "all"
#: commit sha of the dataset repo, read from `HfApi().dataset_info()` on
#: 2026-09-27; a branch name would be a live pointer, this is not.
REVISION = "f7a2487edd7a043a5441a5e9ccc7fe5ddbd9ebe8"
#: declared in the dataset card at that revision (`license: apache-2.0`)
LICENSE = "Apache-2.0"
SPLITS = ("train", "test")
FILES = {
    "train": "all/train-00000-of-00001.parquet",
    "test": "all/test-00000-of-00001.parquet",
}
#: LFS sha256 the Hub reports for each parquet at REVISION
#: (`dataset_info(files_metadata=True)`); `fetch` re-hashes the bytes it
#: downloaded and refuses to decode anything else.
EXPECTED_SHA256 = {
    "train": "46a58d63edfd86e23229c78afe8b72307bb4ca9fb0e8df180cabb3c67ec9dcd5",
    "test": "4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c",
}
SEED = 20260927
LANG = "en"
QUESTION_TYPES = ("choice", "score", "noul")
FAMILY_OF = {
    "choice": EC.DESCRIPTION,
    "score": EC.DESCRIPTION,
    "noul": EC.INFERENCE,
}
NOUL_SUFFIX = "Is this statement true?"
NOUL_DEFAULT_CRITERIA = {
    "false": "The statement is false.",
    "true": "The statement is true.",
}
EVIDENCE_METHOD = "heuristic:factor-overlap"

RAW_DIR = os.path.join(ROOT, "artifacts", "data-raw", "typed-decisions")
OUT_DIR = os.path.join(ROOT, "artifacts", "episodes-external", "typed-decisions")
GATE_DIR = os.path.join(ROOT, "artifacts", "gates", TASK)
GATE_PATH = os.path.join(GATE_DIR, "gate.json")

#: what marks the eval-only cut wherever it travels: a path, a manifest, an
#: episode. `assert_trainable` looks for all three.
EVAL_ONLY_SPLIT = "test"
EVAL_ONLY_KEY = "typed-decisions/test"

_STOP = frozenset(
    "the and for with that this from into out not are was were has have had "
    "any all its their them they you your our can will may within than then "
    "there here about after before over under more most less".split()
)


#: keys whose STRING value is a location (a bare note is not scanned)
_PATH_KEYS = frozenset(
    {"path", "dir", "episodes", "manifest", "source", "file", "shards", "out"}
)


class EvalOnlyLeak(ValueError):
    """The typed-decisions test cut was named where training data goes."""


# ------------------------------------------------------------- utilities


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: str, rows: Iterable[dict]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def _rel(path: str) -> str:
    return os.path.relpath(path, ROOT) if path.startswith(ROOT) else path


# ------------------------------------------------------------- rendering


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        return re.sub(r"\s*\n\s*", " ", value).strip()
    return json.dumps(value)


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def state_lines(obj: Any, prefix: str = "") -> list[str]:
    """`key: value` per line; nested keys dotted; scalar lists «; »-joined;
    object lists indexed. The JSON's own key order is kept: this is a
    rendering, not a normalisation."""
    if isinstance(obj, dict):
        out: list[str] = []
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.extend(state_lines(value, path))
        return out
    if isinstance(obj, list):
        if not obj:
            return [f"{prefix}: []"]
        if all(_is_scalar(x) for x in obj):
            return [f"{prefix}: " + "; ".join(_scalar(x) for x in obj)]
        out = []
        for i, item in enumerate(obj):
            out.extend(state_lines(item, f"{prefix}[{i}]"))
        return out
    return [f"{prefix}: {_scalar(obj)}" if prefix else _scalar(obj)]


def render_state(state: Any) -> str:
    """The `state` column (JSON text or already-parsed) as readable prose."""
    if isinstance(state, str):
        try:
            state = json.loads(state)
        except ValueError:
            return _scalar(state)
    return "\n".join(state_lines(state))


# ------------------------------------------------------------ candidates


def _criteria_items(qtype: str, criteria: Any) -> list[tuple[str, str]]:
    """`(label, text)` pairs in the dataset's order, before shuffling."""
    if qtype == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError("choice criteria must be a non-empty {label: description}")
        return [(str(label), str(desc)) for label, desc in criteria.items()]
    if qtype == "score":
        if not isinstance(criteria, list) or not criteria:
            raise ValueError("score criteria must be a non-empty ordered list")
        return [(str(i), f"{i} — {desc}") for i, desc in enumerate(criteria)]
    if qtype == "noul":
        crit = criteria if criteria else NOUL_DEFAULT_CRITERIA
        if not isinstance(crit, dict) or set(crit) != {"false", "true"}:
            raise ValueError("noul criteria must map exactly false/true")
        return [("false", str(crit["false"])), ("true", str(crit["true"]))]
    raise ValueError(f"unknown question type {qtype!r}; known: {QUESTION_TYPES}")


def shuffle_rng(seed: int, case_id: str, question: str) -> random.Random:
    """One RNG per decision, a pure function of (seed, case, question): the
    order does not depend on how many rows were converted before."""
    return random.Random(f"{seed}\x00{case_id}\x00{question}")


def build_candidates(
    qtype: str, criteria: Any, rng: random.Random
) -> tuple[list[dict], dict[str, str]]:
    """Opaque ids `c1..cK` over a seeded permutation of the criteria.

    Returns the candidate list and `{label: opaque id}` so the gold can be
    pointed at AFTER shuffling — the id is positional in the shuffled
    order, so nothing about the label survives in it.
    """
    items = _criteria_items(qtype, criteria)
    order = list(range(len(items)))
    rng.shuffle(order)
    cands = [{"id": f"c{k + 1}", "text": items[j][1]} for k, j in enumerate(order)]
    label_to_id = {items[j][0]: f"c{k + 1}" for k, j in enumerate(order)}
    return cands, label_to_id


# -------------------------------------------------------------- evidence


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))
    return {w for w in words if (len(w) >= 3 or w.isdigit()) and w not in _STOP}


def factor_tokens(factors: Any) -> set[str]:
    """Tokens of the factor VALUES (strings and numbers; booleans and
    nulls carry no text to find). Keys are not used: they mirror state
    keys and would match every line."""
    if isinstance(factors, str):
        try:
            factors = json.loads(factors)
        except ValueError:
            return _tokens(factors)
    out: set[str] = set()
    if isinstance(factors, dict):
        for value in factors.values():
            out |= factor_tokens(value)
    elif isinstance(factors, list):
        for value in factors:
            out |= factor_tokens(value)
    elif isinstance(factors, bool) or factors is None:
        return out
    elif isinstance(factors, (int, float)):
        out.add(json.dumps(factors))
    return out


def pick_evidence(lines: list[str], factors: Any, label_text: str = "") -> str | None:
    """The state line that overlaps most with the generating factors
    (2 points per factor token) and the gold description (1 point per
    token); first line wins a tie; `None` when nothing overlaps."""
    ft, lt = factor_tokens(factors), _tokens(label_text)
    best, best_score = None, 0
    for line in lines:
        value = line.split(": ", 1)[1] if ": " in line else line
        toks = _tokens(value)
        score = 2 * len(ft & toks) + len(lt & toks)
        if score > best_score:
            best, best_score = line, score
    return best


# ------------------------------------------------------------ conversion


def _load_json(value: Any, what: str) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError as exc:
            raise ValueError(f"{what} is not JSON: {exc}") from exc
    return value


def render_question(qtype: str, instructions: str) -> tuple[str, str]:
    """`(question, how)`: verbatim, or the noul statement made a question."""
    text = (instructions or "").strip()
    if qtype == "noul":
        return f"{text} {NOUL_SUFFIX}", f"noul: statement + {NOUL_SUFFIX!r}"
    return text, "verbatim"


def convert_row(row: dict, split: str, seed: int = SEED) -> tuple[list[dict], list[dict]]:
    """One dataset case → its decisions as episodes, plus rejects."""
    episodes: list[dict] = []
    rejects: list[dict] = []
    case_id = str(row.get("id", ""))
    workflow = str(row.get("workflow", ""))
    group = f"td-{case_id}"

    def reject(question: str, reasons: list[str], stage: str) -> None:
        rejects.append(
            {
                "id": f"td-{case_id}-{question}",
                "case_id": case_id,
                "workflow": workflow,
                "question": question,
                "split": split,
                "stage": stage,
                "reasons": reasons,
            }
        )

    try:
        state_obj = _load_json(row.get("state"), "state")
        questions = _load_json(row.get("questions"), "questions")
        gold = _load_json(row.get("gold"), "gold")
        factors = _load_json(row.get("factors"), "factors")
    except ValueError as exc:
        reject("*", [str(exc)], "parse")
        return episodes, rejects
    if not isinstance(questions, dict) or not isinstance(gold, dict):
        reject("*", ["questions/gold are not objects"], "parse")
        return episodes, rejects
    agreement = row.get("label_agreement")
    if isinstance(agreement, str):
        try:
            agreement = json.loads(agreement)
        except ValueError:
            agreement = None
    state = render_state(state_obj)
    lines = state.split("\n")

    for qname, spec in questions.items():
        qtype = spec.get("type") if isinstance(spec, dict) else None
        g = gold.get(qname)
        if qtype not in QUESTION_TYPES:
            reject(qname, [f"unknown question type {qtype!r}"], "build")
            continue
        if not isinstance(g, dict) or "label" not in g:
            reject(qname, ["gold has no label for this question"], "build")
            continue
        try:
            cands, label_to_id = build_candidates(
                qtype, spec.get("criteria"), shuffle_rng(seed, case_id, qname)
            )
        except ValueError as exc:
            reject(qname, [str(exc)], "build")
            continue
        label = str(g["label"])
        if label not in label_to_id:
            reject(
                qname,
                [f"gold label {label!r} is not a criterion of {sorted(label_to_id)}"],
                "build",
            )
            continue
        answer = label_to_id[label]
        label_text = next(c["text"] for c in cands if c["id"] == answer)
        evidence = pick_evidence(lines, factors, label_text)
        question, how = render_question(qtype, spec.get("instructions", ""))
        probs = g.get("probabilities")
        teacher_soft = (
            {label_to_id[str(k)]: float(v) for k, v in probs.items() if str(k) in label_to_id}
            if isinstance(probs, dict)
            else None
        )
        ep = {
            "id": f"td-{case_id}-{qname}",
            "schema_version": EC.SCHEMA_VERSION,
            "state": state,
            "question": question,
            "candidates": cands,
            "answer": answer,
            "acceptable_answers": None,
            "preference": None,
            "evidence": evidence if evidence is not None else "",
            "evidence_origin": EVIDENCE_METHOD,
            "family": FAMILY_OF[qtype],
            "subfamily": f"external/typed-decisions/{workflow}/{qtype}",
            "question_type": qtype,
            "question_render": how,
            "lang": LANG,
            "origin": ORIGIN,
            "generator_seed": seed,
            "generator_version": CONVERTER_VERSION,
            "variant_group": group,
            "split": split,
            "eval_only": split == EVAL_ONLY_SPLIT,
            "teacher_soft": teacher_soft,
            "teacher_confidence": g.get("confidence"),
            "teacher_score": g.get("score"),
            "source": {
                "dataset": DATASET_ID,
                "config": CONFIG,
                "revision": REVISION,
                "split": split,
                "case_id": case_id,
                "workflow": workflow,
                "question": qname,
                "label": label,
                "label_agreement": (agreement or {}).get(qname)
                if isinstance(agreement, dict)
                else None,
            },
        }
        reasons = EC.validate(ep)
        if evidence is None:
            reasons = ["no state line overlaps the case factors (evidence)"] + [
                r for r in reasons if not r.startswith("missing evidence")
            ]
        if reasons:
            reject(qname, reasons, "validate")
            continue
        episodes.append(ep)
    return episodes, rejects


def convert_rows(rows: Iterable[dict], split: str, seed: int = SEED) -> tuple[list, list]:
    episodes: list[dict] = []
    rejects: list[dict] = []
    for row in rows:
        eps, rjs = convert_row(row, split, seed)
        episodes.extend(eps)
        rejects.extend(rjs)
    return episodes, rejects


def count_by(episodes: list[dict]) -> dict[str, dict[str, int]]:
    """`{workflow: {type: n}}`, measured from the episodes given."""
    out: dict[str, dict[str, int]] = {}
    for ep in episodes:
        wf = ep.get("source", {}).get("workflow", "?")
        qt = ep.get("question_type", "?")
        out.setdefault(wf, {})
        out[wf][qt] = out[wf].get(qt, 0) + 1
    return {wf: dict(sorted(v.items())) for wf, v in sorted(out.items())}


def reasons_histogram(rejects: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for rj in rejects:
        for reason in rj.get("reasons", []):
            key = re.sub(r"'[^']*'", "'…'", reason)
            out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


# ------------------------------------------------------------- the fetch


def _read_parquet(path: str) -> list[dict]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "decoding parquet needs pyarrow, which the hash-locked .venv-train "
            "does not carry: run `fetch` from any venv with `pyarrow "
            "huggingface_hub` installed (it only writes the raw jsonl); "
            "`convert` and `gate` are stdlib and run in .venv-train"
        ) from exc
    return pq.read_table(path).to_pylist()


def fetch(raw_dir: str = RAW_DIR) -> dict:
    """Download both parquet files pinned at REVISION, verify their sha256
    against the Hub's LFS metadata, decode to `<split>.jsonl`."""
    from huggingface_hub import hf_hub_download  # local import: optional dep

    os.makedirs(raw_dir, exist_ok=True)
    manifest: dict[str, Any] = {
        "task": TASK,
        "dataset": DATASET_ID,
        "config": CONFIG,
        "revision": REVISION,
        "license": LICENSE,
        "fetched_utc": _utcnow(),
        "decoder": sys.executable,
        "files": {},
    }
    for split in SPLITS:
        path = hf_hub_download(
            DATASET_ID, FILES[split], repo_type="dataset", revision=REVISION
        )
        digest = sha256_file(path)
        if digest != EXPECTED_SHA256[split]:
            raise ValueError(
                f"{FILES[split]} at {REVISION}: sha256 {digest} != expected "
                f"{EXPECTED_SHA256[split]}; refusing to decode"
            )
        rows = _read_parquet(path)
        out = os.path.join(raw_dir, f"{split}.jsonl")
        n = _write_jsonl(out, rows)
        manifest["files"][split] = {
            "parquet": FILES[split],
            "parquet_sha256": digest,
            "parquet_bytes": os.path.getsize(path),
            "raw": _rel(out),
            "raw_sha256": sha256_file(out),
            "rows": n,
        }
    with open(os.path.join(raw_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return manifest


def raw_manifest(raw_dir: str = RAW_DIR) -> dict:
    path = os.path.join(raw_dir, "manifest.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{_rel(path)} missing: run `python -m data.convert_typed_decisions fetch` "
            "from a venv with pyarrow first"
        )
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ----------------------------------------------------------- the convert


def convert_split(
    split: str,
    seed: int = SEED,
    raw_dir: str = RAW_DIR,
    out_dir: str = OUT_DIR,
    command: str = "",
) -> dict:
    """Read the raw jsonl (sha re-verified), write episodes/rejects/manifest."""
    raw = raw_manifest(raw_dir)
    entry = raw["files"][split]
    raw_path = os.path.join(raw_dir, f"{split}.jsonl")
    digest = sha256_file(raw_path)
    if digest != entry["raw_sha256"]:
        raise ValueError(
            f"{_rel(raw_path)}: sha256 {digest} != {entry['raw_sha256']} recorded "
            "at fetch; the raw rows moved"
        )
    if raw.get("revision") != REVISION:
        raise ValueError(
            f"raw rows were fetched at {raw.get('revision')}, converter pins {REVISION}"
        )
    rows = _read_jsonl(raw_path)
    t0 = time.time()
    episodes, rejects = convert_rows(rows, split, seed)
    base = os.path.join(out_dir, split)
    os.makedirs(base, exist_ok=True)
    ep_path = os.path.join(base, "episodes.jsonl")
    rj_path = os.path.join(base, "rejects.jsonl")
    n_ep = _write_jsonl(ep_path, episodes)
    n_rj = _write_jsonl(rj_path, rejects)
    n_decisions = sum(
        len(_load_json(r.get("questions"), "questions") or {}) for r in rows
    )
    manifest = {
        "task": TASK,
        "converter_version": CONVERTER_VERSION,
        "origin": ORIGIN,
        "dataset": DATASET_ID,
        "config": CONFIG,
        "revision": REVISION,
        "license": LICENSE,
        "split": split,
        "eval_only": split == EVAL_ONLY_SPLIT,
        "seed": seed,
        "command": command
        or f"PYTHONPATH=. .venv-train/bin/python -m data.convert_typed_decisions "
        f"convert --split {split} --seed {seed}",
        "source": {
            "parquet": entry["parquet"],
            "parquet_sha256": entry["parquet_sha256"],
            "parquet_bytes": entry["parquet_bytes"],
            "raw": entry["raw"],
            "raw_sha256": entry["raw_sha256"],
            "rows": entry["rows"],
        },
        "contract": EC.manifest_record(n_ep, seed, CONVERTER_VERSION, {split: n_ep}),
        "rules": {
            "question_render": {
                "choice": "verbatim",
                "score": "verbatim",
                "noul": f"statement + {NOUL_SUFFIX!r}",
            },
            "candidate_text": {
                "choice": "criteria[label] description",
                "score": "'<level> — <description>'",
                "noul": "criteria description, or NOUL_DEFAULT_CRITERIA when absent",
            },
            "noul_default_criteria": NOUL_DEFAULT_CRITERIA,
            "evidence": EVIDENCE_METHOD,
            "family_of": FAMILY_OF,
        },
        "n_cases": len(rows),
        "n_decisions": n_decisions,
        "n_episodes": n_ep,
        "n_rejects": n_rj,
        "per_workflow_type": count_by(episodes),
        "rejects_by_reason": reasons_histogram(rejects),
        "files": {
            "episodes.jsonl": {"rows": n_ep, "sha256": sha256_file(ep_path)},
            "rejects.jsonl": {"rows": n_rj, "sha256": sha256_file(rj_path)},
        },
        "elapsed_s": round(time.time() - t0, 2),
        "built_utc": _utcnow(),
        "status": "published",
    }
    with open(os.path.join(base, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return manifest


def load_split(split: str, out_dir: str = OUT_DIR) -> tuple[list[dict], dict]:
    """Episodes of a published split, verified against their manifest sha."""
    base = os.path.join(out_dir, split)
    with open(os.path.join(base, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    path = os.path.join(base, "episodes.jsonl")
    digest = sha256_file(path)
    recorded = manifest["files"]["episodes.jsonl"]["sha256"]
    if digest != recorded:
        raise ValueError(
            f"{_rel(path)}: sha256 {digest} != {recorded} in its manifest; "
            "the episodes moved under their seal"
        )
    return _read_jsonl(path), manifest


# --------------------------------------------------------------- the gate


def measure_gate(out_dir: str = OUT_DIR, seed: int = SEED) -> dict:
    """Counts measured from the published files, never from memory."""
    per_split: dict[str, Any] = {}
    episodes: dict[str, list[dict]] = {}
    rejects: dict[str, list[dict]] = {}
    for split in SPLITS:
        eps, man = load_split(split, out_dir)
        rjs_path = os.path.join(out_dir, split, "rejects.jsonl")
        rjs = _read_jsonl(rjs_path) if os.path.exists(rjs_path) else []
        episodes[split], rejects[split] = eps, rjs
        report = EC.batch_validate(eps)
        per_split[split] = {
            "manifest": _rel(os.path.join(out_dir, split, "manifest.json")),
            "revision": man.get("revision"),
            "episodes_sha256": man["files"]["episodes.jsonl"]["sha256"],
            "eval_only": bool(man.get("eval_only")),
            "n_cases": man.get("n_cases"),
            "n_decisions": man.get("n_decisions"),
            "n_episodes": len(eps),
            "n_rejects": len(rjs),
            "n_valid": report["n_valid"],
            "duplicate_ids": report["duplicate_ids"],
            "n_groups": len({ep["variant_group"] for ep in eps}),
            "per_workflow_type": count_by(eps),
            "per_type": _per_type(eps),
            "rejects_by_reason": reasons_histogram(rjs),
        }
    train_cases = {ep["source"]["case_id"] for ep in episodes["train"]}
    test_cases = {ep["source"]["case_id"] for ep in episodes["test"]}
    train_states = {ep["state"] for ep in episodes["train"]}
    test_states = {ep["state"] for ep in episodes["test"]}
    train_ids = {ep["id"] for ep in episodes["train"]}
    test_ids = {ep["id"] for ep in episodes["test"]}
    checks = [
        _check(
            "all_published_episodes_validate",
            all(per_split[s]["n_valid"] == per_split[s]["n_episodes"] for s in SPLITS),
            {s: f"{per_split[s]['n_valid']}/{per_split[s]['n_episodes']}" for s in SPLITS},
        ),
        _check(
            "rejects_carry_reasons",
            all(rj.get("reasons") for s in SPLITS for rj in rejects[s]),
            {s: per_split[s]["n_rejects"] for s in SPLITS},
        ),
        _check(
            "no_test_case_id_in_train",
            not (train_cases & test_cases),
            {"shared_case_ids": len(train_cases & test_cases)},
        ),
        _check(
            "no_test_state_in_train",
            not (train_states & test_states),
            {"shared_states": len(train_states & test_states)},
        ),
        _check(
            "no_episode_id_in_both",
            not (train_ids & test_ids),
            {"shared_episode_ids": len(train_ids & test_ids)},
        ),
        _check(
            "test_episodes_marked_eval_only",
            all(ep.get("eval_only") is True for ep in episodes["test"])
            and per_split["test"]["eval_only"],
            {"n_test": len(episodes["test"])},
        ),
        _check(
            "train_episodes_not_eval_only",
            all(ep.get("eval_only") is False for ep in episodes["train"]),
            {"n_train": len(episodes["train"])},
        ),
        _check(
            "no_duplicate_ids",
            all(not per_split[s]["duplicate_ids"] for s in SPLITS),
            {s: len(per_split[s]["duplicate_ids"]) for s in SPLITS},
        ),
        _check(
            "revision_pinned",
            all(per_split[s]["revision"] == REVISION for s in SPLITS),
            {"revision": REVISION},
        ),
        _check(
            "teacher_soft_kept",
            all(
                isinstance(ep.get("teacher_soft"), dict)
                and set(ep["teacher_soft"]) == {c["id"] for c in ep["candidates"]}
                for s in SPLITS
                for ep in episodes[s]
            ),
            {},
        ),
    ]
    gate = {
        "task": TASK,
        "status": "measured",
        "measured_utc": _utcnow(),
        "dataset": DATASET_ID,
        "config": CONFIG,
        "revision": REVISION,
        "license": LICENSE,
        "seed": seed,
        "converter_version": CONVERTER_VERSION,
        "schema_version": EC.SCHEMA_VERSION,
        "schema_sha": EC.schema_sha(),
        "eval_only": [EVAL_ONLY_SPLIT],
        "reference_cut": {
            "split": "test",
            "compare_against": "Jev 1.13.0 and laya-typed-decisions as published in "
            "TMP/laya/BENCHMARKS.md §typed-decisions (not measured here)",
            "reader": "eval.cuts.external_cut('typed-decisions', 'test')",
        },
        "accuracy": None,
        "note": "no accuracy is measured in this task: it is data only",
        "command": "PYTHONPATH=. .venv-train/bin/python -m data.convert_typed_decisions gate",
        "splits": per_split,
        "checks": checks,
        "verdict": "PASS" if all(c["pass"] for c in checks) else "FAIL",
    }
    return gate


def _per_type(episodes: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for ep in episodes:
        qt = ep.get("question_type", "?")
        out[qt] = out.get(qt, 0) + 1
    return dict(sorted(out.items()))


def _check(name: str, ok: bool, detail: dict) -> dict:
    return {"name": name, "pass": bool(ok), "detail": detail}


def write_gate(gate: dict, path: str = GATE_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(gate, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return path


# ------------------------------------------------- the eval-only barrier


def is_eval_only_ref(item: Any) -> bool:
    """Does this path / manifest / episode name the typed-decisions test cut?

    Three ways it can travel, all refused: a path under
    `…/typed-decisions/test`, a manifest (or mix entry) whose `eval_only`
    is true or whose `origin`+`split` name the cut, and an episode with
    `origin == "laya-typed-decisions"` and `split == "test"`.
    """
    if isinstance(item, str):
        norm = item.replace("\\", "/").rstrip("/")
        return EVAL_ONLY_KEY in norm or norm.endswith("typed-decisions-test")
    if isinstance(item, dict):
        if item.get("eval_only") is True:
            return True
        origin = item.get("origin") or item.get("dataset")
        if origin in (ORIGIN, DATASET_ID) and item.get("split") == EVAL_ONLY_SPLIT:
            return True
        for key in ("path", "dir", "episodes", "manifest", "source"):
            value = item.get(key)
            if isinstance(value, (str, dict)) and is_eval_only_ref(value):
                return True
    return False


def assert_trainable(sources: Any) -> list[str]:
    """Refuse a mixture that names the eval-only cut. Raises `EvalOnlyLeak`.

    `sources` is whatever a mix manifest carries: a path, a list of paths,
    a list of episodes, or a manifest dict with `sources`/`members`/
    `episodes`/`datasets` inside. Returns the (empty) list of hits so a
    caller can also use it as a check.
    """
    hits: list[str] = []

    def walk(node: Any, where: str) -> None:
        if isinstance(node, str):
            if is_eval_only_ref(node):
                hits.append(where)
        elif isinstance(node, dict):
            if is_eval_only_ref(node):
                hits.append(where)
                return
            for key, value in node.items():
                if key == "excluded":  # a manifest records WHY a cut is absent
                    continue
                if isinstance(value, (dict, list, tuple)) or (
                    isinstance(value, str) and key in _PATH_KEYS
                ):
                    walk(value, f"{where}.{key}")
        elif isinstance(node, (list, tuple)):
            for i, sub in enumerate(node):
                walk(sub, f"{where}[{i}]")

    walk(sources, "sources")
    if hits:
        raise EvalOnlyLeak(
            f"{DATASET_ID} split {EVAL_ONLY_SPLIT!r} is eval-only (Jev 0,727 / "
            f"Laya 0,766 are quoted on it) and was named as training data at: "
            + ", ".join(hits)
        )
    return hits


# ---------------------------------------------------------------- the CLI


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="data.convert_typed_decisions")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="download pinned parquet, verify sha, decode to raw jsonl")
    c = sub.add_parser("convert", help="raw jsonl → episode-v1 for one or both splits")
    c.add_argument("--split", choices=SPLITS, default=None)
    c.add_argument("--seed", type=int, default=SEED)
    sub.add_parser("gate", help="measure the gate from the published files")
    args = ap.parse_args(argv)
    if args.cmd == "fetch":
        man = fetch()
        print(json.dumps({k: v for k, v in man.items() if k != "files"}, indent=2))
        for split, entry in man["files"].items():
            print(f"[fetch] {split}: {entry['rows']} rows, parquet sha256 "
                  f"{entry['parquet_sha256'][:16]}…, raw {entry['raw']}")
        return 0
    if args.cmd == "convert":
        splits = (args.split,) if args.split else SPLITS
        command = "PYTHONPATH=. .venv-train/bin/python -m data.convert_typed_decisions " + " ".join(
            ["convert"] + argv[1:]
        )
        for split in splits:
            man = convert_split(split, seed=args.seed, command=command)
            print(
                f"[convert] {split}: {man['n_episodes']} episodes, {man['n_rejects']} "
                f"rejects from {man['n_cases']} cases / {man['n_decisions']} decisions "
                f"({man['elapsed_s']} s)"
            )
        return 0
    gate = measure_gate()
    path = write_gate(gate)
    for c in gate["checks"]:
        print(f"[gate] {'ok  ' if c['pass'] else 'FAIL'} {c['name']} {c['detail']}")
    print(f"[gate] {gate['verdict']} → {_rel(path)}")
    return 0 if gate["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
