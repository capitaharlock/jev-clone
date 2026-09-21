"""Quality score per example (§67) and difficulty banding (§68).

§67 fixes the price of every label source: a human-checked label and a
graph-derived one are both worth 1.0, two agreeing teachers 0.8, a lone
teacher 0.55, and an UNVERIFIED example is not cheap — it is rejected. The
rejection is the point: a corpus with no floor is what produced the
synth-loop contamination (#T-halt-contam), so `score_for` returns None and
the caller must drop the row and count it.

Programmatic rows reach 1.0 because the graph is the verifier. A teacher
that only redacted the wording does NOT lower that score — it never had a
say in the label (§35).
"""
from __future__ import annotations

#: source -> score. `unverified` is deliberately absent: see `score_for`.
QUALITY_TIERS = {
    "human": 1.0,
    "programmatic": 1.0,
    "two_teacher": 0.8,
    "one_teacher": 0.55,
}
QUALITY_PROGRAMMATIC = QUALITY_TIERS["programmatic"]

#: below this an example never enters the corpus
MIN_ACCEPTED_SCORE = 0.55

#: §68 — the global easy/medium/hard split of the corpus
DIFFICULTY_BANDS = (("easy", 0.25), ("medium", 0.75), ("hard", 1.0))


def score_for(source: str, teachers_agreeing: int = 0) -> float | None:
    """Quality score for a label source. None means REJECT the example.

    `source` is who decided the ANSWER, not who wrote the prose. Anything
    that is not a recognised verified source — including the explicit
    "unverified" — scores None.
    """
    if source in ("human", "programmatic"):
        return QUALITY_TIERS[source]
    if source == "teacher":
        if teachers_agreeing >= 2:
            return QUALITY_TIERS["two_teacher"]
        if teachers_agreeing == 1:
            return QUALITY_TIERS["one_teacher"]
        return None
    return None


def accepts(source: str, teachers_agreeing: int = 0) -> bool:
    score = score_for(source, teachers_agreeing)
    return score is not None and score >= MIN_ACCEPTED_SCORE


def quality_block(source: str = "programmatic", teachers_agreeing: int = 0,
                  decider: str = "graph") -> dict | None:
    """The `quality` field of one question, or None if the row must be dropped."""
    score = score_for(source, teachers_agreeing)
    if score is None:
        return None
    return {"source": source, "score": score, "decider": decider}


def assign_difficulty(scores: list[float]) -> list[str]:
    """Rank-band `scores` into easy 25 / medium 50 / hard 25 (§68).

    Banding is by RANK, not by an absolute cut, so the mix holds whatever
    the hardness scale of a particular run looks like. Ties break by index,
    which keeps a rebuild from the same corpus identical.
    """
    n = len(scores)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: (scores[i], i))
    out = [""] * n
    for rank, i in enumerate(order):
        frac = rank / n
        for name, upper in DIFFICULTY_BANDS:
            if frac < upper:
                out[i] = name
                break
        else:
            out[i] = "hard"
    return out


def difficulty_histogram(diffs: list[str]) -> dict[str, float]:
    n = max(1, len(diffs))
    return {name: round(sum(1 for d in diffs if d == name) / n, 4)
            for name, _ in DIFFICULTY_BANDS}
