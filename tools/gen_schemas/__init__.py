"""Decision-schema generator (#T-gen-schemas).

Replaces the email-template loop of #T-gen-loop: the unit is a complete
decision schema (domain, state, question, K options with plausible
distractors, answer or `unknown`), bounded by a per-domain quota, deduped by
3-gram Jaccard against everything already generated, and measured on the
axes that decide whether the corpus teaches anything.
"""
from .budget import Budget
from .dedup import SchemaDeduper
from .diversity import diversity, skeleton_curve
from .domains import DOMAIN_IDS, DOMAINS, MAX_K, MIN_K, STATE_FORMATS, domain
from .generator import (DecisionSchema, GenerationResult, build, generate,
                        normalized_text, skeleton_of)
from .llm import RecordProposer
from .vocab import LANGS

__all__ = [
    "Budget", "SchemaDeduper", "DecisionSchema", "GenerationResult",
    "RecordProposer", "DOMAINS", "DOMAIN_IDS", "LANGS", "MIN_K", "MAX_K",
    "STATE_FORMATS", "build", "generate", "diversity", "skeleton_curve",
    "domain", "normalized_text", "skeleton_of",
]
