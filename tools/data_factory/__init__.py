"""Synthetic factory V1 (#T-synth-factory): teacher-council grounded generator.

Pipeline: source_streamer -> task_proposer -> teacher_router (proposer/solver
with 50% role inversion) -> solver -> judge (on disagreement only) ->
adversary (distractor/ambiguity variants) -> validator -> deduper ->
shard_writer. No CoT is ever stored; every variant records parent_example_id.
"""
from __future__ import annotations
