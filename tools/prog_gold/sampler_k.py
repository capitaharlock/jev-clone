"""Variable-K cardinality sampler (§§47, 131).

`data.optset.OptionSetSampler` is the shipped option-set sampler and this
module does NOT replace it: that sampler draws K uniformly in
`[k_min, k_max]` over the prefetched label pools, and it is what the mix
consumes. What §131 asks for here is a different draw — a fixed
35/25/20/10/6/4 mix over K = 2,3,4,5,6,8 against the Wikidata pools — so
only the draw lives here. Everything else about an option set (hard
negatives, wording, ordering) comes from `miner` and `pipeline`.
"""
from __future__ import annotations

import random

#: K -> percentage of rows (§131). Must sum to 100.
K_MIX = [(2, 35), (3, 25), (4, 20), (5, 10), (6, 6), (8, 4)]
K_VALUES = [k for k, _ in K_MIX]
K_WEIGHTS = [w for _, w in K_MIX]

assert sum(K_WEIGHTS) == 100, "K mix must sum to 100"


def draw_k(rng: random.Random) -> int:
    """One K from the guideline mix."""
    return rng.choices(K_VALUES, weights=K_WEIGHTS)[0]


def k_histogram(ks: list[int]) -> dict[int, float]:
    n = max(1, len(ks))
    return {k: round(sum(1 for x in ks if x == k) / n, 4) for k in K_VALUES}


def mix_deviation(ks: list[int]) -> float:
    """Largest absolute gap between the observed share and the target mix."""
    hist = k_histogram(ks)
    return max(abs(hist[k] - w / 100) for k, w in K_MIX)


def hardness(k: int, sim_rank: float) -> float:
    """Hardness of one row: more options and more confusable distractors.

    `sim_rank` is the nearest-label score of the closest distractor, already
    in [0, 1] — it is what the miner ranked the pool by, so the difficulty
    banding and the miner agree on what "close" means.
    """
    span = max(K_VALUES) - min(K_VALUES)
    return (k - min(K_VALUES)) / span * 0.6 + max(0.0, min(1.0, sim_rank)) * 0.4
