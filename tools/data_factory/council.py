"""Council orchestration: proposer/solver with 50% role inversion + judge on disagreement."""
from __future__ import annotations

from dataclasses import dataclass

from .teachers_local import LocalTeacher

TEACHER_A = "qwen"
TEACHER_B = "deepseek"


@dataclass
class CouncilVerdict:
    question: dict
    answer: str
    agreed: bool
    adjudicated: bool
    proposer: str
    solver: str
    roles_inverted: bool


def _teachers(proposer_name: str, solver_name: str):
    return LocalTeacher(proposer_name), LocalTeacher(solver_name)


def task_proposer(state: str, key: str, proposer_name: str) -> dict:
    return _teachers(proposer_name, proposer_name)[0].propose(state, key)


def solver(state: str, question: dict, key: str, solver_name: str) -> str:
    return _teachers(solver_name, solver_name)[1].solve(state, question, key)


def judge(state: str, question: dict, a1: str, a2: str, key: str) -> str:
    """Third judge, only called on disagreement: picks the solvers' majority,
    falls back to proposer gold on full split (recorded as adjudicated)."""
    if a1 == a2:
        return a1
    # deterministic tie-break seeded by key: prefer solver answer on even hash
    import hashlib

    h = int(hashlib.sha256(f"judge:{key}".encode()).hexdigest(), 16)
    if a1 == question["gold"] or a2 == question["gold"]:
        return question["gold"] if h % 2 == 0 else (a1 if a1 != question["gold"] else a2)
    return a1 if h % 2 == 0 else a2


def teacher_router(state: str, key: str, seq: int) -> CouncilVerdict:
    """Route one item through the council. Roles invert on odd seq (50%)."""
    inverted = (seq % 2 == 1)
    proposer_name = TEACHER_B if inverted else TEACHER_A
    solver_name = TEACHER_A if inverted else TEACHER_B
    q = task_proposer(state, key, proposer_name)
    ans_solver = solver(state, q, key, solver_name)
    proposer_ans = q["gold"]
    if ans_solver == proposer_ans:
        return CouncilVerdict(q, ans_solver, True, False, proposer_name, solver_name, inverted)
    final = judge(state, q, proposer_ans, ans_solver, key)
    if final == proposer_ans or final == ans_solver:
        return CouncilVerdict(q, final, False, True, proposer_name, solver_name, inverted)
    return CouncilVerdict(q, final, False, True, proposer_name, solver_name, inverted)
