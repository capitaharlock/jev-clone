"""Corpus mixer for `decision-mix-clean-1m` (#T-mix-1m, data-training §86).

A thin §86 layer over `data/mix.py`, which is the only place that decides
what enters a training mixture and in what proportion. This package adds
what the §86 corpus needs on top of that registry: the benchmark fence
(§§18, 77), the layer plan and its measured gap, the §48 type mix, the
§65 diversity dashboard, the §63 token ledger per shard, and the gate
that publishes all of it — pass or fail — with the numbers that decide it.
"""
