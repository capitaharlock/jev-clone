"""Scores shared by training and the benchmark runner."""

from __future__ import annotations

import math

Z_95 = 1.959964


def wilson_lower(successes: int, n: int, z: float = Z_95) -> float:
    """Lower bound of the 95% Wilson score interval for a binomial rate."""
    if n <= 0:
        return 0.0
    p = successes / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - half) / (1 + z * z / n))


def chance(rows: list[dict]) -> float:
    """Expected forced accuracy of a uniform random pick over each row's options."""
    if not rows:
        return 0.0
    return sum(1 / len(row["candidates"]) for row in rows) / len(rows)
