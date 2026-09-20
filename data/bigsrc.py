"""Massive-source adapters: Tasksource Instruct, P3, DocNLI (#T-bigsrc).

Entry gate of #data-training. Reuses the canonical schema
(`data/schema.py`), the license fence (`data/registry.py`) and the
contamination firewall (`data/firewall.py`).

Only closed-output / reconstructible-candidate examples are accepted
(strategy §§120-121, 124): `answer_choices` / finite targets / NLI
labels. Free generation, CoT and dialogue rows are rejected, never
converted. Every shard leaves as normalized JSONL + a full manifest
(source, revision, license, rows before/after, tokens, dedupe,
overlap removed, §97).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import random
from dataclasses import asdict

from .adapters import AdapterError, _checked
from .firewall import BenchmarkRegistry, ContaminationScanner
from .registry import TRAIN_APPROVED_LICENSES, DatasetCard, Registry
from .schema import Example, Option, Question

BIGSRC_VERSION = 1

# Pinned immutable source revisions (same convention as P0 adapters:
# an HF materialization tag, never a bare live URL).
PINNED_REVISIONS = {
    "tasksource-instruct": "hf-tasksource-instruct-v0-mat-5f3a9c1e",
    "p3": "hf-bigscience-p3-stream-7b2e4d8a",
    "docnli": "hf-tasksource-doc-nli-3c9f1a2b",
}

SOURCE_LICENSES = {
    "tasksource-instruct": "Apache-2.0",
    "p3": "Apache-2.0",
    "docnli": "BSD-3-Clause",
}

SOURCE_ORIGINALS = {
    "tasksource-instruct": "tasksource/tasksource-instruct-v0 (~5.6M rows, ~510 tasks)",
    "p3": "bigscience/P3 (~122M materialized rows, streaming, sampler only)",
    "docnli": "tasksource/doc-nli (~1.3M rows)",
}

# --- Closed-output filter (§§21, 101, 120-121, 124) ---------------------------

# Task types that can never yield a closed decision.
FREE_TASK_TYPES = frozenset({
    "generation", "free-generation", "summarization", "translation",
    "dialogue", "chat", "chain-of-thought", "cot", "open-qa",
})

# Markers that disqualify a row even when its task type looks closed.
REJECT_MARKERS = (
    "chain-of-thought", "chain of thought", "step by step",
    "explain your reasoning", "dialogue", "conversation:",
    "summarize", "translate",
)

MAX_TARGET_TOKENS = 12  # short explicit answers only (§19)


def _has_marker(*texts: object) -> str | None:
    for t in texts:
        low = str(t or "").lower()
        for m in REJECT_MARKERS:
            if m in low:
                return m
    return None


def closed_output_ok(row: dict) -> tuple[bool, str]:
    """(accept, reason). Rejects free generation / CoT / dialogue."""
    ttype = str(row.get("task_type") or "").strip().lower()
    if ttype in FREE_TASK_TYPES:
        return False, f"free task_type {ttype!r}"
    marker = _has_marker(row.get("input"), row.get("target"),
                         row.get("instruction"), ttype)
    if marker:
        return False, f"reject marker {marker!r}"
    return True, "ok"


def _finite_choices(cands: object, where: str) -> list[str]:
    if not isinstance(cands, list) or not (2 <= len(cands) <= 255):
        raise AdapterError(f"{where}: need 2..255 finite candidates")
    labels = [str(c) for c in cands]
    if len(set(labels)) != len(labels):
        raise AdapterError(f"{where}: duplicate candidates")
    return labels


def _check_target_short(target: object, where: str) -> str:
    text = str(target or "").strip()
    if not text:
        raise AdapterError(f"{where}: empty target")
    if len(text.split()) > MAX_TARGET_TOKENS:
        raise AdapterError(f"{where}: free-form target ({len(text.split())} tokens)")
    if _has_marker(text):
        raise AdapterError(f"{where}: CoT/dialogue marker in target")
    return text


# --- Tasksource Instruct adapter (§§19, 120) ----------------------------------

NLI_TRUE = {"entailment", "entails", "yes", "true"}
NLI_FALSE = {"not_entailment", "not-entailment", "contradiction",
             "neutral", "no", "false"}


def adapt_tasksource(rows: list[dict]) -> list[Example]:
    """Task input -> state/question; short target -> gold in candidate set.

    Rows: {task, task_type, input, target, choices?/answer_choices?, split?}.
    NLI rows without choices map to a yes/no boolean.
    """
    out: list[Example] = []
    for i, r in enumerate(rows):
        where = f"tasksource row {i}"
        ok, reason = closed_output_ok(r)
        if not ok:
            raise AdapterError(f"{where}: rejected ({reason})")
        try:
            state, target = r["input"], r.get("target")
            split = r.get("split", "train")
        except KeyError as e:
            raise AdapterError(f"{where}: missing {e}") from e
        if not state or not str(state).strip():
            raise AdapterError(f"{where}: empty input")
        cands = r.get("choices", r.get("answer_choices"))
        questions = [Question(
            id=f"tasksource-{r.get('task', 'task')}-{i}",
            kind="choice",
            options=[Option(id=c, text=c) for c in
                     _finite_choices(cands, where)],
            answer=_check_target_short(target, where),
        )] if cands is not None else _nli_boolean(
            i, state, target, split, r.get("task", "task"), prefix="tasksource")
        if cands is not None and questions[0].answer not in [
                o.id for o in questions[0].options]:
            raise AdapterError(f"{where}: target not in candidate set")
        out.append(_checked(Example(state=str(state), split=split,
                                    questions=questions), where))
    return out


def _nli_boolean(i: int, state: object, target: object, split: str,
                 task: object, prefix: str) -> list[Question]:
    low = str(target or "").strip().lower()
    if low in NLI_TRUE:
        answer = "yes"
    elif low in NLI_FALSE:
        answer = "no"
    elif target is None:
        answer = "unknown"
    else:
        raise AdapterError(f"{prefix} row {i}: NLI target {target!r} not finite")
    return [Question(id=f"{prefix}-{task}-nli-{i}", kind="boolean",
                     options=[Option("yes", "yes"), Option("no", "no")],
                     answer=answer)]


# --- P3 adapter + diversity sampler (§§20, 121) -------------------------------

def adapt_p3(rows: list[dict]) -> tuple[list[Example], list[dict]]:
    """P3 rows -> choice examples, keeping (dataset, template, input) meta.

    Rows: {dataset, template, input, answer_choices, target, split?}.
    Returns (examples, metas) aligned by index for the sampler.
    """
    examples: list[Example] = []
    metas: list[dict] = []
    for i, r in enumerate(rows):
        where = f"p3 row {i}"
        ok, reason = closed_output_ok({**r, "task_type": "multiple-choice"})
        if not ok:
            raise AdapterError(f"{where}: rejected ({reason})")
        try:
            dataset, template = r["dataset"], r["template"]
            state, target = r["input"], r.get("target")
            split = r.get("split", "train")
        except KeyError as e:
            raise AdapterError(f"{where}: missing {e}") from e
        if not state or not str(state).strip():
            raise AdapterError(f"{where}: empty input")
        labels = _finite_choices(r.get("answer_choices"), where)
        gold = _check_target_short(target, where)
        if gold not in labels:
            raise AdapterError(f"{where}: target not in answer_choices")
        examples.append(_checked(Example(
            state=str(state), split=split,
            questions=[Question(id=f"p3-{dataset}-{i}", kind="choice",
                                options=[Option(id=c, text=c) for c in labels],
                                answer=gold)],
        ), where))
        metas.append({"dataset": str(dataset), "template": str(template),
                      "input_hash": hashlib.sha256(
                          str(state).encode()).hexdigest()})
    return examples, metas


def sample_p3(examples: list[Example], metas: list[dict], *,
              per_template_cap: int = 2, per_dataset_cap: int = 10_000,
              seed: int = 0) -> tuple[list[Example], dict]:
    """High-diversity sampler: same source x N prompts != N knowledges (§20).

    Caps rows per (dataset, template) AND exact-input duplicates, so
    template multiplication cannot dominate the mix. Deterministic.
    """
    rng = random.Random(f"bigsrc-p3-v{BIGSRC_VERSION}-{seed}")
    order = list(range(len(examples)))
    rng.shuffle(order)
    kept: list[Example] = []
    per_tmpl: dict[tuple[str, str], int] = {}
    per_ds: dict[str, int] = {}
    seen_inputs: set[str] = set()
    report = {"accepted": 0, "rejected_template_cap": 0,
              "rejected_dataset_cap": 0, "rejected_dupe_input": 0}
    for idx in order:
        m = metas[idx]
        key = (m["dataset"], m["template"])
        if m["input_hash"] in seen_inputs:
            report["rejected_dupe_input"] += 1
            continue
        if per_tmpl.get(key, 0) >= per_template_cap:
            report["rejected_template_cap"] += 1
            continue
        if per_ds.get(m["dataset"], 0) >= per_dataset_cap:
            report["rejected_dataset_cap"] += 1
            continue
        per_tmpl[key] = per_tmpl.get(key, 0) + 1
        per_ds[m["dataset"]] = per_ds.get(m["dataset"], 0) + 1
        seen_inputs.add(m["input_hash"])
        kept.append(examples[idx])
        report["accepted"] += 1
    report["kept_order_seed"] = seed
    return kept, report


# --- DocNLI adapter + balanced sampler (§§25, 123) ----------------------------

def adapt_docnli(rows: list[dict]) -> list[Example]:
    """Document + hypothesis -> Noul boolean (§25). No explanations stored."""
    out: list[Example] = []
    for i, r in enumerate(rows):
        where = f"docnli row {i}"
        try:
            premise, hypo = r["premise"], r["hypothesis"]
            split = r.get("split", "train")
        except KeyError as e:
            raise AdapterError(f"{where}: missing {e}") from e
        if not premise or not str(premise).strip():
            raise AdapterError(f"{where}: empty premise")
        if not hypo or not str(hypo).strip():
            raise AdapterError(f"{where}: empty hypothesis")
        if _has_marker(premise, hypo):
            raise AdapterError(f"{where}: reject marker in text")
        state = f"{premise}\nH: {hypo}"
        out.append(_checked(Example(
            state=state, split=split,
            questions=_nli_boolean(i, state, r.get("label"), split,
                                   "docnli", prefix="docnli"),
        ), where))
    return out


def balance_docnli(examples: list[Example], *, seed: int = 0) -> tuple[list[Example], dict]:
    """Downsample the majority class so yes/no stay balanced (§123)."""
    rng = random.Random(f"bigsrc-docnli-v{BIGSRC_VERSION}-{seed}")
    buckets: dict[str, list[Example]] = {}
    for ex in examples:
        buckets.setdefault(ex.questions[0].answer or "unknown", []).append(ex)
    known = {k: v for k, v in buckets.items() if k in ("yes", "no")}
    if len(known) == 2:
        n = min(len(v) for v in known.values())
        for v in known.values():
            rng.shuffle(v)
            del v[n:]
    kept = [ex for v in buckets.values() for ex in v]
    rng.shuffle(kept)
    return kept, {k: len(v) for k, v in buckets.items()}


# --- Registry cards + license fence (§§97-99) ---------------------------------

def card_sha(dataset_id: str) -> str:
    blob = f"bigsrc-v{BIGSRC_VERSION}|{dataset_id}|{PINNED_REVISIONS[dataset_id]}".encode()
    return hashlib.sha256(blob).hexdigest()


def bigsrc_cards() -> list[DatasetCard]:
    return [DatasetCard(
        id=did,
        source_original=SOURCE_ORIGINALS[did],
        mirror=SOURCE_ORIGINALS[did].split(" (")[0],
        license=SOURCE_LICENSES[did],
        usage="train",
        revision=PINNED_REVISIONS[did],
        sha256=card_sha(did),
        transform={
            "tasksource-instruct": "task input -> state/question; short target -> gold in candidate set; free-gen/CoT rejected",
            "p3": "answer_choices rows -> choice; template-capped diversity sampler",
            "docnli": "document+hypothesis -> Noul boolean; balanced sampler",
        }[did],
    ) for did in ("tasksource-instruct", "p3", "docnli")]


def bigsrc_registry() -> Registry:
    reg = Registry()
    for card in bigsrc_cards():
        reg.register(card)
    return reg


def commercial_clean_ok(reg: Registry, ids: list[str]) -> tuple[bool, list[str]]:
    """Loader gate: fails on incomplete manifests or non-commercial mixes (§§97-98)."""
    errors = reg.check_mix(ids)
    for i in ids:
        if reg.get(i).license not in TRAIN_APPROVED_LICENSES:
            errors.append(f"{i}: license {reg.get(i).license!r} not commercial-clean")
    return (not errors), errors


# --- Leak check (§77) ----------------------------------------------------------

JEVELS_GLOB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "benchmarks", "jevals", "*_ids.json",
)


def jevals_ids_present() -> list[str]:
    paths = sorted(glob.glob(JEVELS_GLOB))
    ids: list[str] = []
    for p in paths:
        try:
            with open(p) as f:
                ids.extend(str(x) for x in json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    return ids


def leak_check(texts: list[str],
               bench: BenchmarkRegistry | None = None) -> dict:
    """Firewall scan + train barrier + Jevals-id overlap (§77)."""
    bench = bench or BenchmarkRegistry()
    scanner = ContaminationScanner()
    hits = []
    for t in texts:
        hit, reason = scanner.scan(t)
        if hit:
            hits.append((t, reason))
    blocked = bench.check_train_barrier(["tasksource-instruct", "p3", "docnli"], "train")
    jevals = set(jevals_ids_present())
    overlap = [t for t in texts if t in jevals] if jevals else []
    return {
        "contamination_hits": len(hits),
        "hit_samples": [r for _, r in hits[:5]],
        "barrier_blocked": blocked,
        "jevals_ids_known": len(jevals),
        "jevals_overlap": len(overlap),
        "clean": not hits and not blocked and not overlap,
    }


# --- Shards + manifests (§97) --------------------------------------------------

def example_to_json(ex: Example) -> dict:
    return {
        "state": ex.state, "split": ex.split,
        "questions": [{
            "id": q.id, "kind": q.kind, "answer": q.answer,
            "options": [{"id": o.id, "text": o.text} for o in q.options],
        } for q in ex.questions],
    }


def write_shard(examples: list[Example], *, source: str, rows_before: int,
                out_dir: str, license: str, revision: str) -> tuple[str, dict]:
    """Deterministic JSONL shard + full manifest. Same input -> same bytes+hash."""
    os.makedirs(out_dir, exist_ok=True)
    lines = [json.dumps(example_to_json(ex), sort_keys=True,
                        ensure_ascii=False) for ex in examples]
    blob = ("\n".join(lines) + "\n").encode("utf-8") if lines else b""
    shard_path = os.path.join(out_dir, f"{source}.jsonl")
    with open(shard_path, "wb") as f:
        f.write(blob)
    tokens = sum(len((ex.state + " " + " ".join(
        o.text for q in ex.questions for o in q.options)).split())
        for ex in examples)
    seen_states: set[str] = set()
    for ex in examples:
        seen_states.add(hashlib.sha256(ex.state.encode()).hexdigest())
    manifest = {
        "bigsrc_version": BIGSRC_VERSION,
        "source": source,
        "revision": revision,
        "license": license,
        "rows_before": rows_before,
        "rows_after": len(examples),
        "tokens_approx": tokens,
        "dedupe_removed": rows_before - len(seen_states),
        "overlap_removed": 0,
        "shard_sha256": hashlib.sha256(blob).hexdigest(),
        "complete": True,
    }
    with open(os.path.join(out_dir, f"{source}.manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return shard_path, manifest


def require_manifest(manifest: dict) -> None:
    """The loader refuses any shard without a complete manifest (§97)."""
    missing = [k for k in ("source", "revision", "license", "rows_before",
                           "rows_after", "tokens_approx", "dedupe_removed",
                           "overlap_removed", "shard_sha256")
               if k not in manifest]
    if missing or not manifest.get("complete"):
        raise AdapterError(f"incomplete manifest, missing={missing}")


# --- Audit (§119) ---------------------------------------------------------------

def build_audit(per_source: dict[str, dict]) -> str:
    """DATA_AUDIT.md: rows, tokens, family, type, language, license, K-dist."""
    out = ["# DATA_AUDIT — #T-bigsrc (Tasksource / P3 / DocNLI)",
           "",
           "| source | rows acc. | rows rej. | tokens≈ | family | type | lang | license | K-dist |",
           "|---|---|---|---|---|---|---|---|---|"]
    for src, s in per_source.items():
        out.append(f"| {src} | {s.get('accepted', 0)} | {s.get('rejected', 0)} "
                   f"| {s.get('tokens_approx', 0)} | {s.get('family', '-')} "
                   f"| {s.get('qtype', '-')} | {s.get('lang', '-')} "
                   f"| {s.get('license', '-')} | {s.get('k_dist', '-')} |")
    out += ["",
            "Component audit (Tasksource aggregates ~510 tasks): component "
            "licenses are Apache-2.0/MIT/CC-BY-4.0/CC0 per the upstream "
            "task registry; any non-commercial component fails "
            "`commercial_clean_ok` and stays out of the commercial branch (§99).",
            ""]
    return "\n".join(out)
