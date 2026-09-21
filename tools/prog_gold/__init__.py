"""Programmatic gold V1 (#T-prog-gold): the GRAPH gives the gold.

Teachers only redact wording — they never decide the correct answer.
Modules: fetch (Wikidata CC0), graph (triple store + checker), miner
(same-taxon hard negatives), sampler_k (K mix), quality (score+difficulty),
packer (multi-Q), transforms (consistency/counterfactual), pipeline, run_pilot.
"""
from __future__ import annotations
