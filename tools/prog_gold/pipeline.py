"""Build packs from cached graph triples. The gold is the graph, always.

Flow per entity: read its triples, draw K from the §131 mix, mine same-taxon
hard negatives, let the teacher adversary reorder them, verify every built
answer back against the graph, band the difficulty, then pack 3-10 questions
per state so the backbone encodes the state once (§§8, 58, 132).

Two things are load-bearing rather than decorative:

* `verify_record` runs on EVERY question before it is kept, not only in the
  tests. A question whose answer does not resolve in the graph is
  unverified, and §67 says unverified is rejected, so it is dropped and
  counted. That is the same check a hand-altered shard fails.
* Adversarial-context rows (§57) carry ~2 000 tokens of filler and cannot
  fit the 512-token state cap. They are written to SEPARATE probe shards
  with the cap explicitly waived and counted, instead of being quietly
  dropped by the validator or quietly relaxing the cap for everyone.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tools.data_factory.validator import validate_record  # noqa: E402

from .fetch import TAXONS  # noqa: E402
from .graph import TripleStore, verify_question  # noqa: E402
from .miner import (  # noqa: E402
    PoolTooSmall,
    adversarial_reorder,
    build_indexes,
    build_pools,
    make_adversary,
)
from .packer import pack  # noqa: E402
from .quality import assign_difficulty, quality_block  # noqa: E402
from .sampler_k import draw_k, hardness  # noqa: E402
from .transforms import (  # noqa: E402
    counterfactual_pair,
    irrelevant_context,
    paraphrase_state,
    shuffle_options,
    weight_of,
)

SEED = 20260920
YES, NO = "yes", "no"
STATE_CAP_ERROR = "state exceeds token limit"
#: choice rows drawn per property
DRAWS_PER_PROP = 2
#: questions of a pack cloned into each layer-D variant (§105 weight control)
LAYER_D_QUESTIONS = 3
#: layer D is built for every Nth entity, not for every one
LAYER_D_EVERY = 2
#: the packer's floor, restated so layer D never emits a thin pack
MIN_PACK_QUESTIONS = 3


def _bool_options() -> list[dict]:
    return [{"id": YES, "text": "Sí"}, {"id": NO, "text": "No"}]


def _choice_options(values: list[str], gold: str, store: TripleStore,
                    rng: random.Random) -> tuple[list[dict], str]:
    """Shuffle values into an option set; ids follow FINAL position.

    Option ids are assigned after the shuffle and the answer is looked up by
    VALUE, never by rendered text: two Wikidata items can share a Spanish
    label, and matching on text would silently point the answer at the wrong
    option.
    """
    order = list(values)
    rng.shuffle(order)
    options = [{"id": f"o{i}", "text": store.labels[v]} for i, v in enumerate(order)]
    answer = options[order.index(gold)]["id"]
    return options, answer


def build_entity_qs(taxon: str, e: str, store: TripleStore, indexes: dict,
                    rng: random.Random, seq: list[int], teacher=None,
                    stats: dict | None = None) -> tuple[list[dict], list[float], dict]:
    """Questions for one entity, each carrying its graph provenance pointer."""
    spec = TAXONS[taxon]
    stats = stats if stats is not None else {}
    label = store.labels[e]
    qs: list[dict] = []
    scores: list[float] = []

    def _bump(key: str, n: int = 1) -> None:
        stats[key] = stats.get(key, 0) + n

    def _next_id() -> str:
        qid = f"g{seq[0]:07d}"
        seq[0] += 1
        return qid

    for prop, pname, part in spec["props"]:
        values = store.labelled_values(e, prop)
        if not values:
            continue
        gold = values[0]
        glabel = store.labels[gold]
        index = indexes.get((taxon, prop))

        # -- choice rows: variable K over same-taxon hard negatives ---------
        if index is not None and index.distinct_labels > 2:
            for _ in range(DRAWS_PER_PROP):
                k = min(draw_k(rng), index.distinct_labels)
                if k < 2:
                    continue
                try:
                    negs = sample_negatives_for(index, gold, k - 1, rng)
                except PoolTooSmall:
                    _bump("pool_too_small")
                    continue
                if teacher is not None:
                    negs, changed = adversarial_reorder(
                        index, gold, negs, teacher, f"{e}:{prop}:{seq[0]}")
                    _bump("adversary_reordered" if changed else "adversary_kept")
                options, answer = _choice_options([gold] + negs, gold, store, rng)
                sim = max((index.score(gold, n) for n in negs), default=0.0)
                qs.append({
                    "id": _next_id(), "kind": "choice",
                    "question": f"¿Cuál es {part} {pname} de {label}?",
                    "options": options, "answer": answer,
                    "variant": "base", "weight": weight_of("base"),
                    "gold": {"kind": "graph", "entity": e, "prop": prop, "value": gold},
                })
                scores.append(hardness(k, sim))

        # -- boolean row + its counterfactual half (§56) ---------------------
        true_q = {
            "id": _next_id(), "kind": "boolean",
            "question": f"¿Es {glabel} {part} {pname} de {label}?",
            "options": _bool_options(), "answer": YES,
            "variant": "base", "weight": weight_of("base"),
            "_subject_label": glabel,
            "gold": {"kind": "graph", "entity": e, "prop": prop, "value": gold},
        }
        qs.append(true_q)
        scores.append(hardness(2, 0.0))

        held = store.values(e, prop)
        false_pool = [v for v in (index.ids if index is not None else [])
                      if v not in held and store.labels[v] != glabel]
        if false_pool:
            false_value = false_pool[rng.randrange(len(false_pool))]
            qs.append(counterfactual_pair(true_q, gold, false_value,
                                          store.labels[false_value], _next_id()))
            scores.append(hardness(2, 0.0))
            _bump("counterfactual_pairs")

    # -- elevation booleans, answered from the graph quantity ---------------
    if taxon == "mountain" and e in store.amounts:
        amt = store.amounts[e]
        others = sorted(a for x, a in store.amounts.items() if x != e)
        alt = int(round(others[rng.randrange(len(others))])) if others else int(amt) + 500
        for thresh in (int(round(amt)) - 1, alt):
            above = amt > thresh
            qs.append({
                "id": _next_id(), "kind": "boolean",
                "question": f"¿Supera {label} los {thresh} metros de altitud?",
                "options": _bool_options(), "answer": YES if above else NO,
                "variant": "base", "weight": weight_of("base"),
                "gold": {"kind": "graph-amount", "entity": e,
                         "threshold": thresh, "above": above},
            })
            scores.append(hardness(2, 0.0))

    # -- §67: an answer the graph does not confirm is unverified -> reject ---
    kept_qs, kept_scores = [], []
    for q, sc in zip(qs, scores):
        errors = verify_question(q, store)
        if errors:
            _bump("rejected_unverified")
            continue
        block = quality_block("programmatic", decider="graph")
        if block is None:  # unreachable for graph rows; §67 says drop, not guess
            _bump("rejected_quality")
            continue
        q["quality"] = block
        kept_qs.append(q)
        kept_scores.append(sc)
    return kept_qs, kept_scores, {"label": label}


def sample_negatives_for(index, gold: str, k: int, rng: random.Random) -> list[str]:
    from .miner import sample_negatives

    return sample_negatives(index, gold, k, rng)


def state_text(taxon: str, e: str, store: TripleStore) -> str:
    spec = TAXONS[taxon]
    label = store.labels[e]
    sents = [f"{label} es {spec['indef']} {spec['es']}."]
    for prop, pname, _part in spec["props"]:
        for v in sorted(store.values(e, prop))[:3]:
            sents.append(f"Su {pname} es {store.labels[v]}.")
    if e in store.amounts:
        sents.append(f"Su altitud es {int(round(store.amounts[e]))} metros.")
    return " ".join(sents)


def strip_for_validation(rec: dict) -> dict:
    """The record as the validator's contract sees it — parent id INCLUDED.

    Dropping `parent_example_id` here is what made the first pilot reject
    100 % of its rows: the validator requires it per question.
    """
    return {
        "state": rec["state"],
        "split": rec["split"],
        "parent_example_id": rec.get("parent_example_id"),
        "questions": [{k: v for k, v in q.items()
                       if k in ("id", "kind", "options", "answer")}
                      for q in rec["questions"]],
    }


def validate_pack(rec: dict) -> list[str]:
    """Validate a pack; waive only the state cap, only for probe rows."""
    errors = validate_record(strip_for_validation(rec))
    if rec.get("probe"):
        errors = [e for e in errors if e != STATE_CAP_ERROR]
    return errors


def make_layer_d(pack_rec: dict, parent: str, seq: list[int],
                 rng: random.Random) -> list[dict]:
    """Layer D: the §105 consistency transforms and the §57 robustness probe.

    Three variants of the SAME questions and the SAME answers — options
    reordered, state reworded, state buried in 2 000 irrelevant tokens. Only
    `LAYER_D_QUESTIONS` of the pack are cloned into each, because a transform
    that copies every row is how a corpus ends up training mostly on its own
    paraphrases; `TRANSFORM_WEIGHTS` prices the rest and the gate holds the
    total under `MAX_TRANSFORM_SHARE`.

    Every variant keeps the pack's split and `parent_example_id`, so no half
    of a transform group can meet the model in a different split.
    """
    source = pack_rec["questions"][:LAYER_D_QUESTIONS]
    if len(source) < MIN_PACK_QUESTIONS:
        return []

    def _clone(variant: str, questions: list[dict] | None = None) -> list[dict]:
        out = []
        for q in (questions or source):
            copy = dict(q)
            copy["id"] = f"g{seq[0]:07d}"
            seq[0] += 1
            copy["variant"] = variant
            copy["weight"] = weight_of(variant)
            copy["parent_question_id"] = q["id"]
            out.append(copy)
        return out

    shuffled = []
    for q in source:
        copy = dict(q)
        copy["options"] = shuffle_options(q["options"], rng)
        shuffled.append(copy)

    para, redactor = paraphrase_state(pack_rec["state"], parent)
    base = {"split": pack_rec["split"], "parent_example_id": parent, "layer": "D"}
    return [
        {**base, "state": pack_rec["state"],
         "questions": _clone("option-shuffle", shuffled),
         "variant": "option-shuffle",
         "teacher": {"redactor": "none", "gold_decider": "graph"}},
        {**base, "state": para, "questions": _clone("paraphrase"),
         "variant": "paraphrase",
         "teacher": {"redactor": redactor, "gold_decider": "graph"}},
        {**base, "state": irrelevant_context(pack_rec["state"]),
         "questions": _clone("irrelevant-context"),
         "variant": "irrelevant-context", "probe": True,
         "teacher": {"redactor": "none", "gold_decider": "graph"}},
    ]


def build_records(store: TripleStore, target_questions: int = 100000,
                  log=print) -> tuple[list[dict], dict]:
    """Build packs until `target_questions` questions are kept."""
    pools = build_pools(store)
    log(f"[prog-gold] indexing {len(pools)} same-taxon pools…")
    indexes = build_indexes(store, pools)
    teacher, teacher_mode = make_adversary(log=log)

    entities = sorted(store.taxon_of.items())
    records: list[dict] = []
    all_scores: list[float] = []
    score_ref: list[dict] = []
    stats: dict = {}
    meta = {
        "entities_seen": 0, "entities_used": 0, "k_draws": [],
        "layers": {}, "taxons": {}, "variants": {}, "teacher_mode": teacher_mode,
        "pools": {f"{t}/{p}": len(v) for (t, p), v in sorted(pools.items())},
    }
    seq = [0]
    n_q = 0

    for idx, (e, taxon) in enumerate(entities):
        if n_q >= target_questions:
            break
        meta["entities_seen"] += 1
        if not store.has_label(e):
            stats["skipped_unlabelled_entity"] = stats.get("skipped_unlabelled_entity", 0) + 1
            continue
        rng = random.Random(SEED + idx)
        qs, scores, _ = build_entity_qs(taxon, e, store, indexes, rng, seq,
                                        teacher=teacher, stats=stats)
        if len(qs) < 3:
            continue
        parent = f"wd:{e}"
        state = state_text(taxon, e, store)
        packs = pack([dict(q, _score=s) for q, s in zip(qs, scores)],
                     state, parent, seq[0])
        if not packs:
            continue
        for pck in packs:
            pck["layer"] = "B"
            pck["variant"] = "base"
        batch = list(packs)
        if n_q < target_questions and idx % LAYER_D_EVERY == 0:
            batch.extend(make_layer_d(packs[0], parent, seq, rng))

        accepted = []
        for rec in batch:
            errors = validate_pack(rec)
            if errors:
                stats["rejected_validator"] = stats.get("rejected_validator", 0) + 1
                stats.setdefault("validator_reasons", {})
                for err in errors[:1]:
                    reasons = stats["validator_reasons"]
                    reasons[err] = reasons.get(err, 0) + 1
                continue
            accepted.append(rec)
        if not accepted:
            continue

        meta["entities_used"] += 1
        meta["taxons"][taxon] = meta["taxons"].get(taxon, 0) + 1
        for rec in accepted:
            layer = rec["layer"]
            meta["layers"][layer] = meta["layers"].get(layer, 0) + len(rec["questions"])
            for q in rec["questions"]:
                sc = q.pop("_score", 0.5)
                variant = q.get("variant", "base")
                meta["variants"][variant] = meta["variants"].get(variant, 0) + 1
                if q["kind"] == "choice" and variant == "base":
                    meta["k_draws"].append(len(q["options"]))
                all_scores.append(sc)
                score_ref.append(q)
            n_q += len(rec["questions"])
            records.append(rec)

    # §68 banding, once, across the whole corpus
    for q, band in zip(score_ref, assign_difficulty(all_scores)):
        q["quality"]["difficulty"] = band
    for rec in records:
        for q in rec["questions"]:
            q.pop("_score", None)
            q.pop("_subject_label", None)

    meta.update(stats)
    meta["questions"] = n_q
    return records, meta
