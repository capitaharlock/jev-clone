"""Decision-schema generator (#T-gen-schemas).

The generated unit is a COMPLETE DECISION SCHEMA — domain, state format,
state, question, K options (K variable, 3..8) with plausible distractors,
the correct answer, and a controlled fraction where the correct answer is
``unknown`` because the state does not carry it. That is the fix for
audit-2026-09-21 finding D: the old loop asked for email *templates*, so
after 11 h it could still only do one task in one language.

Acceptance is a quality bar, not a string check:

* the answer must be derivable from the state (or the case is an `unknown`
  case, and then the answer must NOT be derivable from it — both directions
  are verified per schema, and a violation is a counted rejection);
* the schema must be under its (domain, language, K) quota;
* it must not be a near-duplicate of anything already generated;
* it must not carry benchmark content (the #T-halt-contam firewall).
"""
from __future__ import annotations

import dataclasses
import hashlib
import random
import re
from collections import Counter, deque
from dataclasses import dataclass, field

from data.leakage import normalize
from data.schema import Example, Option, Question, validate
from eval.splits import group_split_of

from .budget import Budget, Cell
from .dedup import SchemaDeduper
from .domains import Domain, domain as get_domain, question_for, render_state, STATE_FORMATS
from .llm import RecordProposer
from .vocab import ORGS, PEOPLE, REF_PREFIXES, t

DEFAULT_UNKNOWN_RATE = 0.15
DEFAULT_DEDUP_THRESHOLD = 0.7
_DIGITS = re.compile(r"\d+")


@dataclass
class DecisionSchema:
    schema_id: str
    domain: str
    language: str
    state_format: str
    queried_field: str
    state: str
    question: str
    options: list[dict]  # [{"id": <value key>, "text": <localized>}]
    answer: str  # option id, or "unknown"
    unknown: bool
    k: int
    hard_negatives: list[str]
    skeleton: str
    split: str
    source: str  # "local" | "llm"
    norm_text: str  # state+question with proper nouns and figures masked
    entities: tuple  # the proper nouns masked out of norm_text

    def cell(self) -> Cell:
        return (self.domain, self.language, self.k)

    def texts(self) -> list[str]:
        return [self.state, self.question] + [o["text"] for o in self.options]

    def to_example(self) -> Example:
        """Universal-schema V1 row. The question rides at the end of the
        state because the V1 Example has no separate question field."""
        return Example(
            state=f"{self.state}\n\n{self.question}",
            questions=[Question(
                id=self.schema_id,
                kind="choice",
                options=[Option(id=o["id"], text=o["text"]) for o in self.options],
                answer=self.answer,
            )],
            split=self.split,
        )

    def to_row(self) -> dict:
        row = dataclasses.asdict(self.to_example())
        row["gen"] = {
            "domain": self.domain, "language": self.language,
            "state_format": self.state_format, "queried_field": self.queried_field,
            "question": self.question, "k": self.k, "unknown": self.unknown,
            "hard_negatives": self.hard_negatives, "skeleton": self.skeleton,
            "source": self.source,
        }
        return row


# --- normalization --------------------------------------------------------

def normalized_text(state: str, question: str, entities: tuple[str, ...]) -> str:
    """State+question with proper nouns and figures masked.

    This is what both the skeleton count and the dedup see, so churning a
    name or an amount can never pass as new content — the exact illusion
    that turned 190 skeletons into 258 700 rows (finding C).
    """
    s = f"{state}\n{question}"
    for e in sorted(entities, key=len, reverse=True):
        if e:
            s = s.replace(e, "@")
    return normalize(_DIGITS.sub("0", s))


def skeleton_of(state: str, question: str, entities: tuple[str, ...]) -> str:
    return hashlib.sha256(
        normalized_text(state, question, entities).encode()).hexdigest()[:16]


# --- candidate construction ----------------------------------------------

def _header(dom: Domain, lang: str, rng: random.Random) -> tuple[str, tuple[str, ...]]:
    ref = f"{rng.choice(REF_PREFIXES)}-{rng.randrange(1000, 99999)}"
    who = rng.choice(PEOPLE + ORGS)
    entities = (ref, who)
    text = t(f"hdr.{dom.id}", lang).format(
        ref=ref, who=who, day=rng.randrange(1, 29), num=rng.randrange(2, 400))
    return text, entities


def sample_record(dom: Domain, rng: random.Random) -> dict[str, str]:
    return {f.name: rng.choice([k for k, _ in f.values()]) for f in dom.fields}


def build(dom: Domain, lang: str, k: int, rng: random.Random, *,
          unknown_rate: float = DEFAULT_UNKNOWN_RATE,
          record: dict[str, str] | None = None,
          source: str = "local") -> tuple[DecisionSchema | None, str]:
    """(schema, reason). A None schema comes with the rejection reason."""
    fld = rng.choice(dom.fields)
    values = fld.values()
    if not 2 <= k <= len(values):
        return None, "k_out_of_range"
    rec = dict(record) if record else {}
    for f in dom.fields:  # an external proposer may under-fill; never coerce
        if rec.get(f.name) not in {key for key, _ in f.values()}:
            rec[f.name] = rng.choice([key for key, _ in f.values()])
    correct = rec[fld.name]
    group = fld.group_of(correct)
    hard = [key for key, g in values if g == group and key != correct]
    easy = [key for key, g in values if g != group]
    rng.shuffle(hard)
    rng.shuffle(easy)
    picked = (hard + easy)[:k - 1]
    if len(picked) < k - 1:
        return None, "not_enough_distractors"
    option_keys = [correct] + picked
    rng.shuffle(option_keys)

    unknown = rng.random() < unknown_rate
    fmt = rng.choice(STATE_FORMATS)
    header, entities = _header(dom, lang, rng)
    state = render_state(dom, rec, lang, fmt, header,
                         omit=fld.name if unknown else None)
    question = question_for(fld, lang)
    answer_text = t(correct, lang)
    visible = answer_text.lower() in state.lower()
    if unknown and visible:
        return None, "answer_visible_in_unknown_case"
    if not unknown and not visible:
        return None, "answer_absent_from_state"

    norm = normalized_text(state, question, entities)
    schema = DecisionSchema(
        schema_id="",  # assigned by the caller once accepted
        domain=dom.id, language=lang, state_format=fmt, queried_field=fld.name,
        state=state, question=question,
        options=[{"id": key, "text": t(key, lang)} for key in option_keys],
        answer="unknown" if unknown else correct,
        unknown=unknown, k=k,
        hard_negatives=[key for key in option_keys if key in hard],
        skeleton=hashlib.sha256(norm.encode()).hexdigest()[:16],
        split=group_split_of(norm, dom.id, lang),
        source=source,
        norm_text=norm,
        entities=entities,
    )
    return schema, "ok"


# --- the budgeted loop ----------------------------------------------------

@dataclass
class GenerationResult:
    schemas: list[DecisionSchema] = field(default_factory=list)
    rejections: Counter = field(default_factory=Counter)
    attempts: int = 0
    budget: Budget | None = None
    deduper: SchemaDeduper | None = None
    proposer: RecordProposer | None = None
    stopped: str = ""

    def normalized_texts(self) -> list[str]:
        return [s.skeleton for s in self.schemas]


def generate(budget: Budget, *, seed: int = 20260921,
             unknown_rate: float = DEFAULT_UNKNOWN_RATE,
             dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
             scanner=None, proposer: RecordProposer | None = None,
             id_prefix: str = "gen-schema", attempt_factor: int = 60,
             log=print) -> GenerationResult:
    """Fill ``budget`` with accepted schemas. Stops when the quota is full.

    ``scanner`` is a ``data.firewall.ContaminationScanner`` (or anything with
    ``.scan(text) -> (hit, reason)``); pass None to skip the firewall only in
    tests that build their own.
    """
    rng = random.Random(f"gen-schemas-v1-{seed}")
    deduper = SchemaDeduper(dedup_threshold)
    proposer = proposer if proposer is not None else RecordProposer(log=log)
    res = GenerationResult(budget=budget, deduper=deduper, proposer=proposer)
    pools: dict[tuple[str, str], deque] = {}
    max_attempts = max(1000, budget.total() * attempt_factor)

    while not budget.full() and res.attempts < max_attempts:
        cells = budget.open_cells()
        rng.shuffle(cells)
        progressed = False
        for cell in cells:
            dom_id, lang, k = cell
            dom = get_domain(dom_id)
            res.attempts += 1
            pool = pools.get((dom_id, lang))
            if pool is None:
                pool = deque(proposer.propose(dom, lang, 32))
                pools[(dom_id, lang)] = pool
            record, source = (pool.popleft(), "llm") if pool else (None, "local")
            schema, reason = build(dom, lang, k, rng, unknown_rate=unknown_rate,
                                   record=record, source=source)
            if schema is None:
                res.rejections[reason] += 1
                continue
            if scanner is not None:
                for text in schema.texts():
                    hit, why = scanner.scan(text)
                    if hit:
                        res.rejections["firewall"] += 1
                        log(f"[gen-schemas] firewall rejected a schema: {why}")
                        schema = None
                        break
                if schema is None:
                    continue
            accepted, _sim = deduper.add(schema.norm_text)
            if not accepted:
                res.rejections["duplicate"] += 1
                continue
            schema.schema_id = f"{id_prefix}-{seed}-{len(res.schemas):06d}"
            errs = validate(schema.to_example())
            if errs:
                res.rejections["invalid_schema"] += 1
                continue
            if not budget.take(cell):
                res.rejections["quota_full"] += 1
                continue
            res.schemas.append(schema)
            progressed = True
        if not progressed:
            res.stopped = "no candidate accepted in a full sweep"
            break
    if not res.stopped:
        res.stopped = "quota full" if budget.full() else "attempt limit reached"
    return res
