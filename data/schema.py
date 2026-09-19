"""Universal data schema V1 (#T-data-schema).

V1 types: `choice` + `boolean` (boolean = binary choice with exactly
two options). `score` / `extract` / `multiselect` are deferred to V2
and rejected here so nothing silently trains on a wrong shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field

V1_KINDS = ("choice", "boolean")
V2_KINDS = ("score", "extract", "multiselect")
VALID_SPLITS = ("train", "calibration", "test", "ood")


@dataclass
class Option:
    id: str
    text: str


@dataclass
class Question:
    id: str
    kind: str
    options: list[Option] = field(default_factory=list)
    answer: str | None = None  # option id, or "unknown"
    teacher_conf: float | None = None


@dataclass
class Example:
    state: str
    questions: list[Question] = field(default_factory=list)
    split: str = "train"


def validate(example: Example) -> list[str]:
    """Return a list of error strings; empty means valid."""
    errors: list[str] = []
    if not example.state or not example.state.strip():
        errors.append("state must be non-empty")
    if example.split not in VALID_SPLITS:
        errors.append(f"split must be one of {VALID_SPLITS}")
    if not example.questions:
        errors.append("at least one question required")
    seen_q: set[str] = set()
    for q in example.questions:
        if q.id in seen_q:
            errors.append(f"duplicate question id: {q.id}")
        seen_q.add(q.id)
        if q.kind in V2_KINDS:
            errors.append(f"question {q.id}: kind {q.kind!r} is V2-only")
        elif q.kind not in V1_KINDS:
            errors.append(f"question {q.id}: unknown kind {q.kind!r}")
        opt_ids = [o.id for o in q.options]
        if len(set(opt_ids)) != len(opt_ids):
            errors.append(f"question {q.id}: duplicate option ids")
        if q.kind == "boolean" and len(q.options) != 2:
            errors.append(f"question {q.id}: boolean needs exactly 2 options")
        if q.kind == "choice" and len(q.options) < 2:
            errors.append(f"question {q.id}: choice needs >= 2 options")
        if q.answer is not None and q.answer != "unknown" and q.answer not in opt_ids:
            errors.append(f"question {q.id}: answer not among options")
        if q.teacher_conf is not None and not 0.0 <= q.teacher_conf <= 1.0:
            errors.append(f"question {q.id}: teacher_conf outside [0,1]")
    return errors


def is_valid(example: Example) -> bool:
    return not validate(example)
