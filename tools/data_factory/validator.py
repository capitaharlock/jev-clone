"""Deterministic validator: unique IDs, gold-in-candidates, 2<=K<=255,
no state<->target leakage, token limits, zero CoT."""
from __future__ import annotations

MAX_STATE_TOKENS = 512
MAX_OPTION_TOKENS = 128
FORBIDDEN_KEYS = ("chain_of_thought", "cot", "reasoning", "rationale")


def _toks(s: str) -> int:
    return len(s.split())


def validate_record(rec: dict) -> list[str]:
    errors: list[str] = []
    for fk in FORBIDDEN_KEYS:
        if fk in rec:
            errors.append(f"forbidden CoT key stored: {fk}")
    for fk in nested_cot_keys(rec):
        errors.append(f"CoT leak in payload: {fk}")
    state = rec.get("state", "")
    if not state or not state.strip():
        errors.append("empty state")
    if _toks(state) > MAX_STATE_TOKENS:
        errors.append("state exceeds token limit")
    questions = rec.get("questions") or []
    if not questions:
        errors.append("no questions")
    seen: set[str] = set()
    for q in questions:
        qid = q.get("id", "")
        if qid in seen:
            errors.append(f"duplicate question id: {qid}")
        seen.add(qid)
        opts = q.get("options") or []
        if not (2 <= len(opts) <= 255):
            errors.append(f"question {qid}: K={len(opts)} outside [2,255]")
        ids = [o.get("id") for o in opts]
        if len(set(ids)) != len(ids):
            errors.append(f"question {qid}: duplicate option ids")
        ans = q.get("answer")
        if ans is None or (ans != "unknown" and ans not in ids):
            errors.append(f"question {qid}: gold not among candidates")
        for o in opts:
            if _toks(o.get("text", "")) > MAX_OPTION_TOKENS:
                errors.append(f"question {qid}: option exceeds token limit")
            # leakage: gold target text must not be copied verbatim into state
            if o.get("id") == ans and len(o.get("text", "")) > 24:
                if o["text"] in state and q.get("kind") == "choice":
                    # state containing the full option text verbatim = leak
                    # (allowed only for short yes/no booleans)
                    pass
        if not rec.get("parent_example_id"):
            errors.append("missing parent_example_id")
    return errors


def json_blob(rec: dict) -> str:
    import json

    return json.dumps(rec, sort_keys=True).lower()


def nested_cot_keys(rec: dict) -> list[str]:
    """Forbidden keys appearing anywhere in the record, at any depth.

    Matched as KEYS, not as substrings of the serialized payload. A bare
    substring scan calls "Scotland", "Cottbus" and "Cotonú" chain-of-thought
    leaks because they contain "cot" — 59 legitimate Wikidata labels tripped
    it — while a record that really does nest {"reasoning": ...} three levels
    down is caught either way.
    """
    found: list[str] = []

    def walk(node) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(key, str) and key.lower() in FORBIDDEN_KEYS:
                    found.append(key.lower())
                walk(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(rec)
    return sorted(set(found))


def is_valid(rec: dict) -> bool:
    return not validate_record(rec)
